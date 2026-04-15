from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame

from config.pipeline_runner import console, print_done, print_stage
from config.spark_manager import SparkManager
from src.ml.models_ej2.common.io import ensure_project_dir, resolve_project_path, save_json
from src.ml.models_ej2.common.split import split_model_stress_spark
from src.ml.models_ej2.common.spark import (
    apply_time_target_not_null_filters,
    evaluate_regression_predictions,
    get_spark_session,
    infer_feature_columns,
    materialize_on_disk,
    read_parquet,
)
from src.ml.models_ej2.common.tuning import (
    build_param_grid_candidates,
    is_higher_better_regression_metric,
    parse_csv_values,
)


def _build_pipeline_with_config(
    num_cols: list[str],
    cat_cols: list[str],
    *,
    one_hot_encode_cats: bool,
    rf_num_trees: int,
    rf_max_depth: int,
    rf_min_instances_per_node: int,
    rf_subsampling_rate: float,
    rf_feature_subset_strategy: str,
    rf_max_bins: int,
    rf_max_memory_mb: int,
    rf_seed: int,
) -> tuple[Pipeline, list[str]]:
    stages = []
    encoded_cat_cols = []

    for c in cat_cols:
        idx = f"{c}__idx"
        stages.append(StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep"))
        if one_hot_encode_cats:
            ohe = f"{c}__ohe"
            stages.append(OneHotEncoder(inputCols=[idx], outputCols=[ohe], handleInvalid="keep"))
            encoded_cat_cols.append(ohe)
        else:
            # En árboles, usar índices evita expandir mucho el vector en memoria.
            encoded_cat_cols.append(idx)

    feature_cols = num_cols + encoded_cat_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    rf = RandomForestRegressor(
        labelCol="label",
        featuresCol="features",
        numTrees=rf_num_trees,
        maxDepth=rf_max_depth,
        minInstancesPerNode=rf_min_instances_per_node,
        subsamplingRate=rf_subsampling_rate,
        featureSubsetStrategy=rf_feature_subset_strategy,
        maxBins=rf_max_bins,
        maxMemoryInMB=rf_max_memory_mb,
        seed=rf_seed,
    )
    stages.append(rf)

    return Pipeline(stages=stages), feature_cols


def _resolve_rf_training_params(
    *,
    rf_num_trees: int | None,
    rf_max_depth: int | None,
    rf_subsampling_rate: float | None,
    rf_max_bins: int | None,
    rf_max_memory_mb: int | None,
) -> dict[str, Any]:
    defaults = {
        # Defaults fijos orientados a WSL: conservadores pero algo más exigentes.
        "rf_num_trees": 100,
        "rf_max_depth": 8,
        "rf_subsampling_rate": 0.6,
        "rf_max_bins": 16,
        "rf_max_memory_mb": 32,
    }
    return {
        "rf_num_trees": defaults["rf_num_trees"] if rf_num_trees is None else rf_num_trees,
        "rf_max_depth": defaults["rf_max_depth"] if rf_max_depth is None else rf_max_depth,
        "rf_subsampling_rate": (
            defaults["rf_subsampling_rate"] if rf_subsampling_rate is None else rf_subsampling_rate
        ),
        "rf_max_bins": defaults["rf_max_bins"] if rf_max_bins is None else rf_max_bins,
        "rf_max_memory_mb": defaults["rf_max_memory_mb"] if rf_max_memory_mb is None else rf_max_memory_mb,
    }


def _build_tuning_candidates(
    *,
    max_trials: int,
    num_trees_values: list[int] | None,
    max_depth_values: list[int] | None,
    subsampling_values: list[float] | None,
    max_bins_values: list[int] | None,
    max_memory_values: list[int] | None,
    base_params: dict[str, Any],
) -> list[dict[str, Any]]:
    defaults = {
        # Grid por defecto estable para WSL (evita picos típicos de OOM).
        "rf_num_trees": [80, 100, 120],
        "rf_max_depth": [7, 8],
        "rf_subsampling_rate": [0.5, 0.6],
        "rf_max_bins": [16],
        "rf_max_memory_mb": [32],
    }
    grid = {
        "rf_num_trees": num_trees_values or defaults["rf_num_trees"],
        "rf_max_depth": max_depth_values or defaults["rf_max_depth"],
        "rf_subsampling_rate": subsampling_values or defaults["rf_subsampling_rate"],
        "rf_max_bins": max_bins_values or defaults["rf_max_bins"],
        "rf_max_memory_mb": max_memory_values or defaults["rf_max_memory_mb"],
    }

    return build_param_grid_candidates(
        grid=grid,
        max_trials=max_trials,
        base_params=base_params,
    )


def _feature_names_from_metadata(
    transformed_df: DataFrame,
    n_features: int,
    fallback_feature_cols: list[str],
) -> list[str]:
    if len(fallback_feature_cols) == n_features:
        names = list(fallback_feature_cols)
    else:
        names = [f"f_{i}" for i in range(n_features)]

    try:
        features_meta = transformed_df.schema["features"].metadata
        attrs_by_type = features_meta.get("ml_attr", {}).get("attrs", {})
        extracted: list[tuple[int, str]] = []

        for attr_type in ("numeric", "binary", "nominal"):
            for attr in attrs_by_type.get(attr_type, []):
                idx = attr.get("idx")
                name = attr.get("name")
                if isinstance(idx, int) and 0 <= idx < n_features and isinstance(name, str) and name:
                    extracted.append((idx, name))

        if extracted:
            names = [f"f_{i}" for i in range(n_features)]
            for idx, name in extracted:
                names[idx] = name
    except Exception:
        pass

    return names


def _feature_importance(model, transformed_df: DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # Extraer importancia usando metadata cuando hay OHE
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances

    n_features = len(importances)
    names = _feature_names_from_metadata(
        transformed_df=transformed_df,
        n_features=n_features,
        fallback_feature_cols=feature_cols,
    )

    data = {
        "feature": names,
        "importance": [float(x) for x in importances],
    }
    df = pd.DataFrame(data).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def run_random_forest_stress(
    input_dir: str = "data/aggregated/ex_stress/df_stress_zone_hour_day",
    outputs_dir: str = "outputs/ejercicio2/random_forest",
    target_col: str = "target_stress_t1",
    time_col: str = "timestamp_hour",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    extra_drop_cols: list[str] | None = None,
    shuffle_partitions: int = 200,
    one_hot_encode_cats: bool = False,
    rf_num_trees: int | None = None,
    rf_max_depth: int | None = None,
    rf_subsampling_rate: float | None = None,
    rf_max_bins: int | None = None,
    rf_max_memory_mb: int | None = None,
    tune_hyperparams: bool = False,
    tune_metric: str = "rmse",
    tune_max_trials: int = 8,
    tune_train_sample_frac: float = 0.25,
    tune_val_sample_frac: float = 0.50,
    tune_num_trees_values: list[int] | None = None,
    tune_max_depth_values: list[int] | None = None,
    tune_subsampling_values: list[float] | None = None,
    tune_max_bins_values: list[int] | None = None,
    tune_max_memory_values: list[int] | None = None,
    include_train_metrics: bool = True,
    refit_train_val: bool = True,
    fit_all_data: bool = False,
) -> dict[str, Any]:
    print_stage("ML RANDOM FOREST (SPARK)", "Regresion de stress con Spark", color="green")

    input_path = resolve_project_path(input_dir)
    outputs_path = ensure_project_dir(outputs_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"No existe el dataset de entrada: {input_path}")

    spark = get_spark_session(shuffle_partitions=shuffle_partitions)
    rf_params = _resolve_rf_training_params(
        rf_num_trees=rf_num_trees,
        rf_max_depth=rf_max_depth,
        rf_subsampling_rate=rf_subsampling_rate,
        rf_max_bins=rf_max_bins,
        rf_max_memory_mb=rf_max_memory_mb,
    )

    train_df = None
    val_df = None
    test_df = None
    cached_dfs: list[DataFrame] = []
    try:
        console.print(f"[cyan]Leyendo dataset[/cyan] -> {input_path}")
        df = read_parquet(spark, input_path)
        df = apply_time_target_not_null_filters(df, time_col=time_col, target_col=target_col)

        train_df, val_df, test_df, bounds = split_model_stress_spark(
            df=df,
            target_col=target_col,
            time_col=time_col,
            train_frac=train_frac,
            val_frac=val_frac,
            extra_drop_cols=extra_drop_cols,
            label_col="label",
            return_bounds=True,
        )

        # Materializamos en disco para evitar re-cálculo + picos de RAM en local mode.
        train_df = materialize_on_disk(train_df)
        val_df = materialize_on_disk(val_df)
        test_df = materialize_on_disk(test_df)
        cached_dfs.extend([train_df, val_df, test_df])

        # Preparar columnas
        num_cols, cat_cols = infer_feature_columns(train_df, label_col="label")
        if cat_cols:
            cat_encoding = "OneHotEncoder" if one_hot_encode_cats else "StringIndexer (sin one-hot)"
            console.print(
                "[cyan]Columnas categóricas detectadas[/cyan] "
                f"({len(cat_cols)}): {', '.join(cat_cols)} | encoding={cat_encoding}"
            )
        else:
            console.print("[cyan]Columnas categóricas detectadas[/cyan]: 0")
        df_train = train_df
        df_val = val_df
        df_test = test_df

        if tune_hyperparams:
            if tune_max_trials <= 0:
                raise ValueError("--tune-max-trials debe ser > 0.")
            if not (0 < tune_train_sample_frac <= 1):
                raise ValueError("--tune-train-sample-frac debe estar en (0, 1].")
            if not (0 < tune_val_sample_frac <= 1):
                raise ValueError("--tune-val-sample-frac debe estar en (0, 1].")

        selected_rf_params = rf_params.copy()
        tuning_summary: dict[str, Any] | None = None

        if tune_hyperparams:
            candidates = _build_tuning_candidates(
                max_trials=tune_max_trials,
                num_trees_values=tune_num_trees_values,
                max_depth_values=tune_max_depth_values,
                subsampling_values=tune_subsampling_values,
                max_bins_values=tune_max_bins_values,
                max_memory_values=tune_max_memory_values,
                base_params=rf_params,
            )
            console.print(
                "[cyan]Tuning de hiperparámetros (validación temporal)[/cyan] "
                f"| metric={tune_metric} | trials={len(candidates)} "
                f"| train_sample_frac={tune_train_sample_frac} | val_sample_frac={tune_val_sample_frac}"
            )

            tune_train_df = df_train
            tune_val_df = df_val
            if tune_train_sample_frac < 1.0:
                tune_train_df = materialize_on_disk(
                    df_train.sample(withReplacement=False, fraction=tune_train_sample_frac, seed=42)
                )
                cached_dfs.append(tune_train_df)
            if tune_val_sample_frac < 1.0:
                tune_val_df = materialize_on_disk(
                    df_val.sample(withReplacement=False, fraction=tune_val_sample_frac, seed=42)
                )
                cached_dfs.append(tune_val_df)

            higher_is_better = is_higher_better_regression_metric(tune_metric)
            best_trial: dict[str, Any] | None = None
            trials: list[dict[str, Any]] = []

            for i, params in enumerate(candidates, start=1):
                console.print(
                    f"[cyan]Trial {i}/{len(candidates)}[/cyan] "
                    f"| trees={params['rf_num_trees']} depth={params['rf_max_depth']} "
                    f"subsample={params['rf_subsampling_rate']} bins={params['rf_max_bins']} "
                    f"memMB={params['rf_max_memory_mb']}"
                )
                trial_pipeline, _ = _build_pipeline_with_config(
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    one_hot_encode_cats=one_hot_encode_cats,
                    rf_num_trees=int(params["rf_num_trees"]),
                    rf_max_depth=int(params["rf_max_depth"]),
                    rf_min_instances_per_node=2,
                    rf_subsampling_rate=float(params["rf_subsampling_rate"]),
                    rf_feature_subset_strategy="sqrt",
                    rf_max_bins=int(params["rf_max_bins"]),
                    rf_max_memory_mb=int(params["rf_max_memory_mb"]),
                    rf_seed=42,
                )

                try:
                    trial_model = trial_pipeline.fit(tune_train_df)
                    trial_pred_val = trial_model.transform(tune_val_df)
                    trial_metrics_val = evaluate_regression_predictions(trial_pred_val)
                    trial_score = float(trial_metrics_val[tune_metric])
                    trial_row = {
                        "trial": i,
                        "status": "ok",
                        "score_metric": tune_metric,
                        "score_value": trial_score,
                        "rf_num_trees": int(params["rf_num_trees"]),
                        "rf_max_depth": int(params["rf_max_depth"]),
                        "rf_subsampling_rate": float(params["rf_subsampling_rate"]),
                        "rf_max_bins": int(params["rf_max_bins"]),
                        "rf_max_memory_mb": int(params["rf_max_memory_mb"]),
                        "val_mae": float(trial_metrics_val["mae"]),
                        "val_rmse": float(trial_metrics_val["rmse"]),
                        "val_r2": float(trial_metrics_val["r2"]),
                    }
                    trials.append(trial_row)

                    is_better = (
                        best_trial is None
                        or (
                            trial_score > best_trial["score_value"]
                            if higher_is_better
                            else trial_score < best_trial["score_value"]
                        )
                    )
                    if is_better:
                        best_trial = trial_row
                        selected_rf_params = {
                            "rf_num_trees": int(params["rf_num_trees"]),
                            "rf_max_depth": int(params["rf_max_depth"]),
                            "rf_subsampling_rate": float(params["rf_subsampling_rate"]),
                            "rf_max_bins": int(params["rf_max_bins"]),
                            "rf_max_memory_mb": int(params["rf_max_memory_mb"]),
                        }
                except Exception as e:
                    err = str(e)
                    trials.append(
                        {
                            "trial": i,
                            "status": "failed",
                            "error": err,
                            "rf_num_trees": int(params["rf_num_trees"]),
                            "rf_max_depth": int(params["rf_max_depth"]),
                            "rf_subsampling_rate": float(params["rf_subsampling_rate"]),
                            "rf_max_bins": int(params["rf_max_bins"]),
                            "rf_max_memory_mb": int(params["rf_max_memory_mb"]),
                        }
                    )
                    err_lower = err.lower()
                    if (
                        "java heap space" in err_lower
                        or "connection refused" in err_lower
                        or "answer from java side is empty" in err_lower
                    ):
                        raise RuntimeError(
                            "Spark JVM cayó durante el tuning (OOM/conexión). "
                            "Reduce --tune-max-trials o usa parámetros más conservadores."
                        ) from e

            if best_trial is None:
                raise RuntimeError("Ningún trial del tuning terminó correctamente.")

            tuning_trials_fp = outputs_path / "random_forest_stress_spark_tuning_trials.csv"
            pd.DataFrame(trials).to_csv(tuning_trials_fp, index=False)

            tuning_summary = {
                "enabled": True,
                "metric": tune_metric,
                "higher_is_better": bool(higher_is_better),
                "n_trials": len(candidates),
                "train_sample_frac": float(tune_train_sample_frac),
                "val_sample_frac": float(tune_val_sample_frac),
                "best_trial": best_trial,
                "best_params": selected_rf_params,
                "trials_csv": str(tuning_trials_fp),
            }
            tuning_report_fp = outputs_path / "random_forest_stress_spark_tuning_report.json"
            save_json(tuning_summary, tuning_report_fp)

            console.print(
                "[green]Mejor trial[/green] "
                f"| {tune_metric}={best_trial['score_value']:.6f} "
                f"| params={selected_rf_params}"
            )
        else:
            tuning_summary = {
                "enabled": False,
                "selected_params": selected_rf_params,
            }

        def _build_pipeline_selected() -> tuple[Pipeline, list[str]]:
            return _build_pipeline_with_config(
                num_cols=num_cols,
                cat_cols=cat_cols,
                one_hot_encode_cats=one_hot_encode_cats,
                rf_num_trees=selected_rf_params["rf_num_trees"],
                rf_max_depth=selected_rf_params["rf_max_depth"],
                rf_min_instances_per_node=2,
                rf_subsampling_rate=selected_rf_params["rf_subsampling_rate"],
                rf_feature_subset_strategy="sqrt",
                rf_max_bins=selected_rf_params["rf_max_bins"],
                rf_max_memory_mb=selected_rf_params["rf_max_memory_mb"],
                rf_seed=42,
            )

        console.print(
            "[cyan]Entrenando modelo base (fit en train)[/cyan] "
            f"| numTrees={selected_rf_params['rf_num_trees']} "
            f"| maxDepth={selected_rf_params['rf_max_depth']} "
            f"| subsamplingRate={selected_rf_params['rf_subsampling_rate']} "
            f"| maxBins={selected_rf_params['rf_max_bins']} "
            f"| maxMemoryInMB={selected_rf_params['rf_max_memory_mb']} "
            f"| one_hot_encode_cats={one_hot_encode_cats}"
        )
        train_pipeline, feature_cols = _build_pipeline_selected()
        model_train = train_pipeline.fit(df_train)

        pred_train = model_train.transform(df_train) if include_train_metrics else None
        pred_val = model_train.transform(df_val)
        pred_test = model_train.transform(df_test)

        metrics_train = evaluate_regression_predictions(pred_train) if pred_train is not None else None
        metrics_val = evaluate_regression_predictions(pred_val)
        metrics_test = evaluate_regression_predictions(pred_test)
        overfit_gaps_train_fit = None
        if metrics_train is not None:
            overfit_gaps_train_fit = {
                "val_minus_train_rmse": float(metrics_val["rmse"] - metrics_train["rmse"]),
                "test_minus_train_rmse": float(metrics_test["rmse"] - metrics_train["rmse"]),
                "train_minus_val_r2": float(metrics_train["r2"] - metrics_val["r2"]),
                "train_minus_test_r2": float(metrics_train["r2"] - metrics_test["r2"]),
            }

        train_fit_metrics = {
            "train_seen": metrics_train,
            "val_unseen": metrics_val,
            "test_unseen": metrics_test,
            "overfit_gaps": overfit_gaps_train_fit,
        }

        # Refit en train+val tras elegir hiperparámetros (buena práctica antes de evaluar en test final).
        train_val_df = materialize_on_disk(df_train.unionByName(df_val))
        cached_dfs.append(train_val_df)

        model_selected = model_train
        pred_for_importance = pred_val
        final_model_training_data = "train"
        train_val_refit_metrics = None

        if refit_train_val:
            console.print("[cyan]Refit final con train+val[/cyan] (test permanece no visto hasta aquí)")
            train_val_pipeline, _ = _build_pipeline_selected()
            model_train_val = train_val_pipeline.fit(train_val_df)

            pred_train_val_on_train = model_train_val.transform(df_train) if include_train_metrics else None
            pred_train_val_on_val = model_train_val.transform(df_val) if include_train_metrics else None
            pred_train_val_on_train_val = model_train_val.transform(train_val_df)
            pred_train_val_on_test = model_train_val.transform(df_test)

            metrics_train_seen = (
                evaluate_regression_predictions(pred_train_val_on_train)
                if pred_train_val_on_train is not None
                else None
            )
            metrics_val_seen = (
                evaluate_regression_predictions(pred_train_val_on_val)
                if pred_train_val_on_val is not None
                else None
            )
            metrics_train_val_seen = evaluate_regression_predictions(pred_train_val_on_train_val)
            metrics_test_unseen = evaluate_regression_predictions(pred_train_val_on_test)

            overfit_gaps_train_val_fit = {
                "test_minus_train_val_rmse": float(metrics_test_unseen["rmse"] - metrics_train_val_seen["rmse"]),
                "train_val_minus_test_r2": float(metrics_train_val_seen["r2"] - metrics_test_unseen["r2"]),
            }

            train_val_refit_metrics = {
                "train_seen": metrics_train_seen,
                "val_seen": metrics_val_seen,
                "train_val_seen": metrics_train_val_seen,
                "test_unseen": metrics_test_unseen,
                "overfit_gaps": overfit_gaps_train_val_fit,
            }

            model_selected = model_train_val
            pred_for_importance = pred_train_val_on_test
            final_model_training_data = "train_plus_val"

        all_data_metrics = None
        model_all_fp = None
        if fit_all_data:
            console.print(
                "[cyan]Entrenando modelo para despliegue con train+val+test[/cyan] "
                "(métricas de este modelo son solo in-sample)"
            )
            all_data_df = materialize_on_disk(train_val_df.unionByName(df_test))
            cached_dfs.append(all_data_df)
            all_pipeline, _ = _build_pipeline_selected()
            model_all = all_pipeline.fit(all_data_df)
            pred_all = model_all.transform(all_data_df)
            all_data_metrics = {
                "all_data_seen": evaluate_regression_predictions(pred_all),
                "warning": (
                    "Métricas optimistas (entrenado y evaluado sobre los mismos datos). "
                    "No usar para estimar generalización."
                ),
            }
            model_all_fp = outputs_path / "random_forest_stress_spark_model_all_data"
            model_all.write().overwrite().save(str(model_all_fp))

        # Guardar modelo seleccionado (por defecto, el refit train+val si está habilitado).
        model_fp = outputs_path / "random_forest_stress_spark_model"
        model_selected.write().overwrite().save(str(model_fp))

        # Importancias del modelo seleccionado
        importance_df = _feature_importance(model_selected, pred_for_importance, feature_cols)
        importance_fp = outputs_path / "random_forest_stress_spark_feature_importance.csv"
        importance_df.to_csv(importance_fp, index=False)

        report = {
            "model": "RandomForestRegressor_spark",
            "input_dir": str(input_path),
            "target_col": target_col,
            "time_col": time_col,
            "extra_drop_cols": extra_drop_cols,
            "split_params": {
                "train_frac": float(train_frac),
                "val_frac": float(val_frac),
                "bounds": bounds,
            },
            "features": {
                "numeric": num_cols,
                "categorical": cat_cols,
                "n_features": len(feature_cols),
                "one_hot_encode_cats": bool(one_hot_encode_cats),
                "categorical_encoding": "one_hot" if one_hot_encode_cats else "string_index",
            },
            "training_params": {
                "rf_num_trees": int(selected_rf_params["rf_num_trees"]),
                "rf_max_depth": int(selected_rf_params["rf_max_depth"]),
                "rf_subsampling_rate": float(selected_rf_params["rf_subsampling_rate"]),
                "rf_max_bins": int(selected_rf_params["rf_max_bins"]),
                "rf_max_memory_mb": int(selected_rf_params["rf_max_memory_mb"]),
            },
            "training_protocol": {
                "tuning": "train -> val",
                "post_tuning_refit": "train+val -> test" if refit_train_val else "train -> test",
                "final_model_training_data": final_model_training_data,
                "fit_all_data_enabled": bool(fit_all_data),
            },
            "metrics": {
                "train_fit": train_fit_metrics,
                "train_val_refit": train_val_refit_metrics,
                "all_data_fit": all_data_metrics,
            },
            "tuning": tuning_summary,
            "artifacts": {
                "model_dir": str(model_fp),
                "all_data_model_dir": None if model_all_fp is None else str(model_all_fp),
                "feature_importance_csv": str(importance_fp),
            },
        }

        report_fp = outputs_path / "random_forest_stress_spark_report.json"
        save_json(report, report_fp)

        console.print(f"[green]Modelo guardado[/green] -> {model_fp}")
        if model_all_fp is not None:
            console.print(f"[green]Modelo all-data guardado[/green] -> {model_all_fp}")
        console.print(f"[green]Importancias guardadas[/green] -> {importance_fp}")
        console.print(f"[green]Reporte guardado[/green] -> {report_fp}")

        print_done("RANDOM FOREST SPARK COMPLETADO")
        return report
    finally:
        for split_df in cached_dfs:
            if split_df is not None:
                split_df.unpersist(blocking=False)
        SparkManager.stop_session()


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena RandomForestRegressor con Spark ML.")
    p.add_argument(
        "--input-dir",
        default="data/aggregated/ex_stress/df_stress_zone_hour_day",
        help="Directorio del dataset de stress (parquets particionados).",
    )
    p.add_argument(
        "--outputs-dir",
        default="outputs/ejercicio2/random_forest",
        help="Directorio de salida para modelo, metricas e importancias.",
    )
    p.add_argument("--target-col", default="target_stress_t1")
    p.add_argument("--time-col", default="timestamp_hour")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument(
        "--drop-col",
        action="append",
        default=None,
        help="Columna extra a eliminar de X. Repetible. Se anade a las columnas base del split.",
    )
    p.add_argument(
        "--shuffle-partitions",
        type=int,
        default=200,
        help="Numero de particiones de shuffle en Spark.",
    )
    p.add_argument(
        "--one-hot-cats",
        action="store_true",
        help=(
            "Aplica OneHotEncoder a categoricas string. "
            "En arboles suele consumir mas memoria; por defecto se usan indices."
        ),
    )
    p.add_argument(
        "--rf-num-trees",
        type=int,
        default=None,
        help="Numero de arboles del RandomForest. Si no se pasa, usa el valor por defecto del script.",
    )
    p.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Profundidad maxima. Si no se pasa, usa el valor por defecto del script.",
    )
    p.add_argument(
        "--rf-subsampling-rate",
        type=float,
        default=None,
        help="Subsampling por arbol. Si no se pasa, usa el valor por defecto del script.",
    )
    p.add_argument(
        "--rf-max-memory-mb",
        type=int,
        default=None,
        help="Memoria del algoritmo por nodo/particion en MB. Si no se pasa, usa el valor por defecto del script.",
    )
    p.add_argument(
        "--rf-max-bins",
        type=int,
        default=None,
        help="Numero de bins para features continuas. Si no se pasa, usa el valor por defecto del script.",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Activa búsqueda de hiperparámetros sobre validación temporal.",
    )
    p.add_argument(
        "--tune-metric",
        default="rmse",
        choices=["rmse", "mae", "r2"],
        help="Métrica objetivo para elegir el mejor trial.",
    )
    p.add_argument(
        "--tune-max-trials",
        type=int,
        default=8,
        help="Máximo de combinaciones evaluadas durante el tuning.",
    )
    p.add_argument(
        "--tune-train-sample-frac",
        type=float,
        default=0.25,
        help="Fracción de train usada para tuning (1.0 usa todo train).",
    )
    p.add_argument(
        "--tune-val-sample-frac",
        type=float,
        default=0.50,
        help="Fracción de val usada para tuning (1.0 usa todo val).",
    )
    p.add_argument(
        "--tune-num-trees-values",
        default=None,
        help="Lista CSV para tuning de numTrees (ej: 80,100,120).",
    )
    p.add_argument(
        "--tune-max-depth-values",
        default=None,
        help="Lista CSV para tuning de maxDepth (ej: 7,8).",
    )
    p.add_argument(
        "--tune-subsampling-values",
        default=None,
        help="Lista CSV para tuning de subsamplingRate (ej: 0.5,0.6).",
    )
    p.add_argument(
        "--tune-max-bins-values",
        default=None,
        help="Lista CSV para tuning de maxBins (ej: 16).",
    )
    p.add_argument(
        "--tune-max-memory-values",
        default=None,
        help="Lista CSV para tuning de maxMemoryInMB (ej: 32).",
    )
    p.add_argument(
        "--skip-train-metrics",
        action="store_true",
        help="No calcula métricas de train (más rápido/menos coste).",
    )
    p.add_argument(
        "--no-refit-train-val",
        action="store_true",
        help=(
            "Desactiva el refit final con train+val tras el tuning. "
            "Por defecto se realiza refit con train+val."
        ),
    )
    p.add_argument(
        "--fit-all-data",
        action="store_true",
        help=(
            "Entrena además un modelo de despliegue con train+val+test "
            "(sin validez para medir generalización)."
        ),
    )

    args = p.parse_args()
    tune_num_trees_values = parse_csv_values(args.tune_num_trees_values, int)
    tune_max_depth_values = parse_csv_values(args.tune_max_depth_values, int)
    tune_subsampling_values = parse_csv_values(args.tune_subsampling_values, float)
    tune_max_bins_values = parse_csv_values(args.tune_max_bins_values, int)
    tune_max_memory_values = parse_csv_values(args.tune_max_memory_values, int)

    run_random_forest_stress(
        input_dir=args.input_dir,
        outputs_dir=args.outputs_dir,
        target_col=args.target_col,
        time_col=args.time_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        extra_drop_cols=args.drop_col,
        shuffle_partitions=args.shuffle_partitions,
        one_hot_encode_cats=args.one_hot_cats,
        rf_num_trees=args.rf_num_trees,
        rf_max_depth=args.rf_max_depth,
        rf_subsampling_rate=args.rf_subsampling_rate,
        rf_max_bins=args.rf_max_bins,
        rf_max_memory_mb=args.rf_max_memory_mb,
        tune_hyperparams=args.tune,
        tune_metric=args.tune_metric,
        tune_max_trials=args.tune_max_trials,
        tune_train_sample_frac=args.tune_train_sample_frac,
        tune_val_sample_frac=args.tune_val_sample_frac,
        tune_num_trees_values=tune_num_trees_values,
        tune_max_depth_values=tune_max_depth_values,
        tune_subsampling_values=tune_subsampling_values,
        tune_max_bins_values=tune_max_bins_values,
        tune_max_memory_values=tune_max_memory_values,
        include_train_metrics=not args.skip_train_metrics,
        refit_train_val=not args.no_refit_train_val,
        fit_all_data=args.fit_all_data,
    )


if __name__ == "__main__":
    main()
