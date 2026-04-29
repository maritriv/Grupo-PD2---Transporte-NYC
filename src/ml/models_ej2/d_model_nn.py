from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class NNPreparedData:
    x_num: np.ndarray
    x_cat: np.ndarray
    y: np.ndarray


class TabularTorchDataset(Dataset):
    def __init__(self, data: NNPreparedData):
        self.x_num = torch.tensor(data.x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(data.x_cat, dtype=torch.long)
        self.y = torch.tensor(data.y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class EmbeddingMLPRegressor(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list[int],
        embedding_dim_cap: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.15,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.num_numeric = int(num_numeric)
        self.cat_cardinalities = list(cat_cardinalities)
        self.embedding_dim_cap = int(embedding_dim_cap)

        self.embeddings = nn.ModuleList()
        emb_total_dim = 0
        for card in self.cat_cardinalities:
            emb_dim = min(self.embedding_dim_cap, max(4, int(round(1.6 * math.sqrt(card)))))
            self.embeddings.append(nn.Embedding(num_embeddings=card, embedding_dim=emb_dim))
            emb_total_dim += emb_dim

        layers: list[nn.Module] = []
        in_dim = self.num_numeric + emb_total_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            cat_parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num, *cat_parts], dim=1)
        else:
            x = x_num
        return self.mlp(x)


class TorchRegressorWrapper:
    def __init__(
        self,
        model: EmbeddingMLPRegressor,
        device: str,
        lr: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        grad_clip_norm: float,
        verbose: bool = False,
    ):
        self.model = model
        self.device = device
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.grad_clip_norm = float(grad_clip_norm)
        self.verbose = verbose

        self.model.to(self.device)
        self.best_state: dict[str, torch.Tensor] | None = None
        self.training_history: list[dict[str, float]] = []

    @staticmethod
    def _to_loader(data: NNPreparedData, batch_size: int, shuffle: bool) -> DataLoader:
        ds = TabularTorchDataset(data)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    @staticmethod
    def _loss(pred_log1p: torch.Tensor, y_log1p: torch.Tensor) -> torch.Tensor:
        return nn.functional.smooth_l1_loss(pred_log1p, y_log1p)

    @torch.no_grad()
    def predict(self, data: NNPreparedData) -> np.ndarray:
        self.model.eval()
        loader = self._to_loader(data, batch_size=max(1024, self.batch_size), shuffle=False)
        preds: list[np.ndarray] = []
        for x_num, x_cat, _ in loader:
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device)
            out = self.model(x_num, x_cat)
            pred = torch.expm1(out).clamp(min=0.0)
            preds.append(pred.cpu().numpy().reshape(-1))
        return np.concatenate(preds) if preds else np.empty(0, dtype=float)

    @torch.no_grad()
    def evaluate_loss(self, data: NNPreparedData) -> float:
        self.model.eval()
        loader = self._to_loader(data, batch_size=max(1024, self.batch_size), shuffle=False)
        losses = []
        for x_num, x_cat, y in loader:
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device)
            y_log1p = torch.log1p(y.to(self.device))
            out = self.model(x_num, x_cat)
            losses.append(float(self._loss(out, y_log1p).detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    def fit(self, train: NNPreparedData, val: NNPreparedData | None = None) -> None:
        train_loader = self._to_loader(train, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, self.patience // 3),
        )

        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            batch_losses = []
            for x_num, x_cat, y in train_loader:
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                y_log1p = torch.log1p(y.to(self.device))

                optimizer.zero_grad(set_to_none=True)
                out = self.model(x_num, x_cat)
                loss = self._loss(out, y_log1p)
                loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            val_loss = self.evaluate_loss(val) if val is not None else train_loss
            scheduler.step(val_loss)

            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )

            if self.verbose and (epoch == 1 or epoch % 10 == 0):
                console.print(
                    f"[cyan]Epoch {epoch:03d}[/cyan] | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | lr={optimizer.param_groups[0]['lr']:.6f}"
                )

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


# -----------------------------------------------------------------------------
# Preprocessing for Torch
# -----------------------------------------------------------------------------
def _infer_nn_feature_columns(df_train: DataFrame, label_col: str = "label") -> tuple[list[str], list[str]]:
    return infer_feature_columns(df_train, label_col=label_col)


def _collect_spark_to_pandas(df: DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.select(*cols).toPandas()


def _fit_numeric_stats(df_train: pd.DataFrame, num_cols: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for c in num_cols:
        s = pd.to_numeric(df_train[c], errors="coerce")
        med = float(s.median()) if not s.dropna().empty else 0.0
        mean = float(s.fillna(med).mean())
        std = float(s.fillna(med).std())
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        stats[c] = {"median": med, "mean": mean, "std": std}
    return stats


def _apply_numeric_stats(df: pd.DataFrame, num_cols: list[str], stats: dict[str, dict[str, float]]) -> np.ndarray:
    arrs = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        meta = stats[c]
        s = s.fillna(meta["median"])
        s = (s - meta["mean"]) / meta["std"]
        arrs.append(s.to_numpy(dtype=np.float32))
    if not arrs:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.column_stack(arrs).astype(np.float32)


def _fit_category_maps(df_train: pd.DataFrame, cat_cols: list[str]) -> tuple[dict[str, dict[str, int]], list[int]]:
    cat_maps: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    for c in cat_cols:
        vals = df_train[c].astype("string").fillna("<MISSING>")
        uniq = sorted(vals.unique().tolist())
        mapping = {"<UNK>": 0}
        for i, v in enumerate(uniq, start=1):
            mapping[str(v)] = i
        cat_maps[c] = mapping
        cardinalities.append(len(mapping))
    return cat_maps, cardinalities


def _apply_category_maps(df: pd.DataFrame, cat_cols: list[str], cat_maps: dict[str, dict[str, int]]) -> np.ndarray:
    cols = []
    for c in cat_cols:
        mapping = cat_maps[c]
        vals = df[c].astype("string").fillna("<MISSING>").astype(str)
        encoded = vals.map(lambda x: mapping.get(x, 0)).to_numpy(dtype=np.int64)
        cols.append(encoded)
    if not cols:
        return np.zeros((len(df), 0), dtype=np.int64)
    return np.column_stack(cols).astype(np.int64)


def _prepare_nn_data(
    df: pd.DataFrame,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    target_col: str,
    num_stats: dict[str, dict[str, float]],
    cat_maps: dict[str, dict[str, int]],
) -> NNPreparedData:
    x_num = _apply_numeric_stats(df, num_cols, num_stats)
    x_cat = _apply_category_maps(df, cat_cols, cat_maps)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return NNPreparedData(x_num=x_num, x_cat=x_cat, y=y)


# -----------------------------------------------------------------------------
# NN config / tuning
# -----------------------------------------------------------------------------
def _resolve_nn_training_params(
    *,
    nn_hidden_dims: str | None,
    nn_dropout: float | None,
    nn_learning_rate: float | None,
    nn_weight_decay: float | None,
    nn_batch_size: int | None,
    nn_max_epochs: int | None,
    nn_patience: int | None,
    nn_embedding_dim_cap: int | None,
) -> dict[str, Any]:
    defaults = {
        "nn_hidden_dims": "64,32",
        "nn_dropout": 0.15,
        "nn_learning_rate": 1e-3,
        "nn_weight_decay": 1e-5,
        "nn_batch_size": 2048,
        "nn_max_epochs": 30,
        "nn_patience": 5,
        "nn_embedding_dim_cap": 12,
    }
    return {
        "nn_hidden_dims": defaults["nn_hidden_dims"] if nn_hidden_dims is None else nn_hidden_dims,
        "nn_dropout": defaults["nn_dropout"] if nn_dropout is None else nn_dropout,
        "nn_learning_rate": defaults["nn_learning_rate"] if nn_learning_rate is None else nn_learning_rate,
        "nn_weight_decay": defaults["nn_weight_decay"] if nn_weight_decay is None else nn_weight_decay,
        "nn_batch_size": defaults["nn_batch_size"] if nn_batch_size is None else nn_batch_size,
        "nn_max_epochs": defaults["nn_max_epochs"] if nn_max_epochs is None else nn_max_epochs,
        "nn_patience": defaults["nn_patience"] if nn_patience is None else nn_patience,
        "nn_embedding_dim_cap": defaults["nn_embedding_dim_cap"] if nn_embedding_dim_cap is None else nn_embedding_dim_cap,
    }


def _build_tuning_candidates(
    *,
    max_trials: int,
    hidden_dims_values: list[str] | None,
    dropout_values: list[float] | None,
    learning_rate_values: list[float] | None,
    weight_decay_values: list[float] | None,
    batch_size_values: list[int] | None,
    embedding_cap_values: list[int] | None,
    base_params: dict[str, Any],
) -> list[dict[str, Any]]:
    defaults = {
        "nn_hidden_dims": ["256,128,64", "512,256,128"],
        "nn_dropout": [0.15, 0.25],
        "nn_learning_rate": [1e-3, 5e-4],
        "nn_weight_decay": [1e-5, 5e-5],
        "nn_batch_size": [1024, 2048],
        "nn_embedding_dim_cap": [24, 32],
    }
    grid = {
        "nn_hidden_dims": hidden_dims_values or defaults["nn_hidden_dims"],
        "nn_dropout": dropout_values or defaults["nn_dropout"],
        "nn_learning_rate": learning_rate_values or defaults["nn_learning_rate"],
        "nn_weight_decay": weight_decay_values or defaults["nn_weight_decay"],
        "nn_batch_size": batch_size_values or defaults["nn_batch_size"],
        "nn_embedding_dim_cap": embedding_cap_values or defaults["nn_embedding_dim_cap"],
    }
    return build_param_grid_candidates(grid=grid, max_trials=max_trials, base_params=base_params)


def _parse_hidden_dims(s: str) -> tuple[int, ...]:
    vals = [int(v.strip()) for v in str(s).split(",") if v.strip()]
    if not vals:
        raise ValueError("nn_hidden_dims no puede estar vacio.")
    return tuple(vals)


def _build_wrapper(
    *,
    num_numeric: int,
    cat_cardinalities: list[int],
    nn_hidden_dims: str,
    nn_dropout: float,
    nn_learning_rate: float,
    nn_weight_decay: float,
    nn_batch_size: int,
    nn_max_epochs: int,
    nn_patience: int,
    nn_embedding_dim_cap: int,
    seed: int,
    verbose: bool,
) -> TorchRegressorWrapper:
    _set_seed(seed)
    model = EmbeddingMLPRegressor(
        num_numeric=num_numeric,
        cat_cardinalities=cat_cardinalities,
        embedding_dim_cap=nn_embedding_dim_cap,
        hidden_dims=_parse_hidden_dims(nn_hidden_dims),
        dropout=nn_dropout,
        use_batch_norm=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TorchRegressorWrapper(
        model=model,
        device=device,
        lr=nn_learning_rate,
        weight_decay=nn_weight_decay,
        batch_size=nn_batch_size,
        max_epochs=nn_max_epochs,
        patience=nn_patience,
        grad_clip_norm=1.0,
        verbose=verbose,
    )


# -----------------------------------------------------------------------------
# Feature importance by permutation
# -----------------------------------------------------------------------------
def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _permutation_importance(
    wrapper: TorchRegressorWrapper,
    pdf_eval: pd.DataFrame,
    prepared_eval: NNPreparedData,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    target_col: str,
    num_stats: dict[str, dict[str, float]],
    cat_maps: dict[str, dict[str, int]],
    max_features: int = 40,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base_pred = wrapper.predict(prepared_eval)
    y_true = prepared_eval.y
    base_mae = _compute_basic_metrics(y_true, base_pred)["mae"]

    feature_candidates = num_cols + cat_cols
    rows = []
    for feature in feature_candidates[:max_features]:
        shuffled = pdf_eval.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        prepared = _prepare_nn_data(
            shuffled,
            num_cols=num_cols,
            cat_cols=cat_cols,
            target_col=target_col,
            num_stats=num_stats,
            cat_maps=cat_maps,
        )
        pred = wrapper.predict(prepared)
        mae = _compute_basic_metrics(y_true, pred)["mae"]
        rows.append({"feature": feature, "importance": float(mae - base_mae)})

    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------
def run_neural_network_stress(
    input_dir: str = "data/aggregated/ex_stress/df_stress_zone_hour_day",
    outputs_dir: str = "outputs/ejercicio2/neural_network",
    target_col: str = "target_stress_t1",
    time_col: str = "timestamp_hour",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    extra_drop_cols: list[str] | None = None,
    shuffle_partitions: int = 200,
    nn_hidden_dims: str | None = None,
    nn_dropout: float | None = None,
    nn_learning_rate: float | None = None,
    nn_weight_decay: float | None = None,
    nn_batch_size: int | None = None,
    nn_max_epochs: int | None = None,
    nn_patience: int | None = None,
    nn_embedding_dim_cap: int | None = None,
    tune_hyperparams: bool = False,
    tune_metric: str = "rmse",
    tune_max_trials: int = 6,
    tune_train_sample_frac: float = 0.25,
    tune_val_sample_frac: float = 0.50,
    tune_hidden_dims_values: list[str] | None = None,
    tune_dropout_values: list[float] | None = None,
    tune_learning_rate_values: list[float] | None = None,
    tune_weight_decay_values: list[float] | None = None,
    tune_batch_size_values: list[int] | None = None,
    tune_embedding_cap_values: list[int] | None = None,
    include_train_metrics: bool = True,
    refit_train_val: bool = False,
    fit_all_data: bool = False,
    compute_importance: bool = False,
    seed: int = 42,
    verbose_epochs: bool = False,
) -> dict[str, Any]:
    print_stage("ML NEURAL NETWORK (SPARK+TORCH)", "Regresion de stress con embeddings", color="green")

    input_path = resolve_project_path(input_dir)
    outputs_path = ensure_project_dir(outputs_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el dataset de entrada: {input_path}")

    spark = get_spark_session(shuffle_partitions=shuffle_partitions)
    nn_params = _resolve_nn_training_params(
        nn_hidden_dims=nn_hidden_dims,
        nn_dropout=nn_dropout,
        nn_learning_rate=nn_learning_rate,
        nn_weight_decay=nn_weight_decay,
        nn_batch_size=nn_batch_size,
        nn_max_epochs=nn_max_epochs,
        nn_patience=nn_patience,
        nn_embedding_dim_cap=nn_embedding_dim_cap,
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

        train_df = materialize_on_disk(train_df)
        val_df = materialize_on_disk(val_df)
        test_df = materialize_on_disk(test_df)
        cached_dfs.extend([train_df, val_df, test_df])

        num_cols, cat_cols = _infer_nn_feature_columns(train_df, label_col="label")
        if cat_cols:
            console.print(
                "[cyan]Columnas categóricas detectadas[/cyan] "
                f"({len(cat_cols)}): {', '.join(cat_cols)} | encoding=embeddings"
            )
        else:
            console.print("[cyan]Columnas categóricas detectadas[/cyan]: 0")

        # El split ya elimina columnas auxiliares como timestamp_hour de X.
        # Por eso NO forzamos time_col aquí: si no existe tras el split, Spark falla
        # con UNRESOLVED_COLUMN. Para entrenar Torch solo hacen falta features + target.
        available_train_cols = set(train_df.columns)
        select_cols = [c for c in [*num_cols, *cat_cols, target_col] if c in available_train_cols]
        missing_required = [c for c in [target_col] if c not in select_cols]
        if missing_required:
            raise ValueError(f"Faltan columnas requeridas tras el split: {missing_required}")

        pdf_train = _collect_spark_to_pandas(train_df, select_cols)
        pdf_val = _collect_spark_to_pandas(val_df, select_cols)
        pdf_test = _collect_spark_to_pandas(test_df, select_cols)

        num_stats = _fit_numeric_stats(pdf_train, num_cols)
        cat_maps, cat_cardinalities = _fit_category_maps(pdf_train, cat_cols)

        train_prepared = _prepare_nn_data(
            pdf_train,
            num_cols=num_cols,
            cat_cols=cat_cols,
            target_col=target_col,
            num_stats=num_stats,
            cat_maps=cat_maps,
        )
        val_prepared = _prepare_nn_data(
            pdf_val,
            num_cols=num_cols,
            cat_cols=cat_cols,
            target_col=target_col,
            num_stats=num_stats,
            cat_maps=cat_maps,
        )
        test_prepared = _prepare_nn_data(
            pdf_test,
            num_cols=num_cols,
            cat_cols=cat_cols,
            target_col=target_col,
            num_stats=num_stats,
            cat_maps=cat_maps,
        )

        if tune_hyperparams:
            if tune_max_trials <= 0:
                raise ValueError("--tune-max-trials debe ser > 0.")
            if not (0 < tune_train_sample_frac <= 1):
                raise ValueError("--tune-train-sample-frac debe estar en (0, 1].")
            if not (0 < tune_val_sample_frac <= 1):
                raise ValueError("--tune-val-sample-frac debe estar en (0, 1].")

        selected_params = nn_params.copy()
        tuning_summary: dict[str, Any] | None = None

        if tune_hyperparams:
            candidates = _build_tuning_candidates(
                max_trials=tune_max_trials,
                hidden_dims_values=tune_hidden_dims_values,
                dropout_values=tune_dropout_values,
                learning_rate_values=tune_learning_rate_values,
                weight_decay_values=tune_weight_decay_values,
                batch_size_values=tune_batch_size_values,
                embedding_cap_values=tune_embedding_cap_values,
                base_params=nn_params,
            )
            console.print(
                "[cyan]Tuning de hiperparámetros (validación temporal)[/cyan] "
                f"| metric={tune_metric} | trials={len(candidates)} "
                f"| train_sample_frac={tune_train_sample_frac} | val_sample_frac={tune_val_sample_frac}"
            )

            higher_is_better = is_higher_better_regression_metric(tune_metric)
            best_trial: dict[str, Any] | None = None
            trials: list[dict[str, Any]] = []

            n_train_tune = max(256, int(len(pdf_train) * tune_train_sample_frac))
            n_val_tune = max(256, int(len(pdf_val) * tune_val_sample_frac))
            pdf_train_tune = pdf_train.iloc[:n_train_tune].copy()
            pdf_val_tune = pdf_val.iloc[:n_val_tune].copy()
            train_tune = _prepare_nn_data(
                pdf_train_tune, num_cols=num_cols, cat_cols=cat_cols, target_col=target_col, num_stats=num_stats, cat_maps=cat_maps
            )
            val_tune = _prepare_nn_data(
                pdf_val_tune, num_cols=num_cols, cat_cols=cat_cols, target_col=target_col, num_stats=num_stats, cat_maps=cat_maps
            )

            for i, params in enumerate(candidates, start=1):
                console.print(
                    f"[cyan]Trial {i}/{len(candidates)}[/cyan] | hidden={params['nn_hidden_dims']} | dropout={params['nn_dropout']} | lr={params['nn_learning_rate']} | wd={params['nn_weight_decay']} | bs={params['nn_batch_size']} | emb_cap={params['nn_embedding_dim_cap']}"
                )
                try:
                    wrapper = _build_wrapper(
                        num_numeric=len(num_cols),
                        cat_cardinalities=cat_cardinalities,
                        nn_hidden_dims=str(params["nn_hidden_dims"]),
                        nn_dropout=float(params["nn_dropout"]),
                        nn_learning_rate=float(params["nn_learning_rate"]),
                        nn_weight_decay=float(params["nn_weight_decay"]),
                        nn_batch_size=int(params["nn_batch_size"]),
                        nn_max_epochs=int(nn_params["nn_max_epochs"]),
                        nn_patience=int(nn_params["nn_patience"]),
                        nn_embedding_dim_cap=int(params["nn_embedding_dim_cap"]),
                        seed=seed,
                        verbose=False,
                    )
                    wrapper.fit(train_tune, val_tune)
                    pred_val = wrapper.predict(val_tune)
                    metrics_val = _compute_basic_metrics(val_tune.y, pred_val)
                    trial_score = float(metrics_val[tune_metric])
                    trial_row = {
                        "trial": i,
                        "status": "ok",
                        "score_metric": tune_metric,
                        "score_value": trial_score,
                        "nn_hidden_dims": str(params["nn_hidden_dims"]),
                        "nn_dropout": float(params["nn_dropout"]),
                        "nn_learning_rate": float(params["nn_learning_rate"]),
                        "nn_weight_decay": float(params["nn_weight_decay"]),
                        "nn_batch_size": int(params["nn_batch_size"]),
                        "nn_embedding_dim_cap": int(params["nn_embedding_dim_cap"]),
                        "val_mae": float(metrics_val["mae"]),
                        "val_rmse": float(metrics_val["rmse"]),
                        "val_r2": float(metrics_val["r2"]),
                    }
                    trials.append(trial_row)
                    is_better = (
                        best_trial is None
                        or (trial_score > best_trial["score_value"] if higher_is_better else trial_score < best_trial["score_value"])
                    )
                    if is_better:
                        best_trial = trial_row
                        selected_params = {
                            **selected_params,
                            "nn_hidden_dims": str(params["nn_hidden_dims"]),
                            "nn_dropout": float(params["nn_dropout"]),
                            "nn_learning_rate": float(params["nn_learning_rate"]),
                            "nn_weight_decay": float(params["nn_weight_decay"]),
                            "nn_batch_size": int(params["nn_batch_size"]),
                            "nn_embedding_dim_cap": int(params["nn_embedding_dim_cap"]),
                        }
                except Exception as e:
                    trials.append(
                        {
                            "trial": i,
                            "status": "failed",
                            "error": str(e),
                            "nn_hidden_dims": str(params["nn_hidden_dims"]),
                            "nn_dropout": float(params["nn_dropout"]),
                            "nn_learning_rate": float(params["nn_learning_rate"]),
                            "nn_weight_decay": float(params["nn_weight_decay"]),
                            "nn_batch_size": int(params["nn_batch_size"]),
                            "nn_embedding_dim_cap": int(params["nn_embedding_dim_cap"]),
                        }
                    )

            if best_trial is None:
                raise RuntimeError("Ningún trial del tuning terminó correctamente.")

            tuning_trials_fp = outputs_path / "neural_network_stress_tuning_trials.csv"
            pd.DataFrame(trials).to_csv(tuning_trials_fp, index=False)
            tuning_summary = {
                "enabled": True,
                "metric": tune_metric,
                "higher_is_better": bool(higher_is_better),
                "n_trials": len(candidates),
                "train_sample_frac": float(tune_train_sample_frac),
                "val_sample_frac": float(tune_val_sample_frac),
                "best_trial": best_trial,
                "best_params": selected_params,
                "trials_csv": str(tuning_trials_fp),
            }
            save_json(tuning_summary, outputs_path / "neural_network_stress_tuning_report.json")
            console.print(
                "[green]Mejor trial[/green] "
                f"| {tune_metric}={best_trial['score_value']:.6f} | params={selected_params}"
            )
        else:
            tuning_summary = {"enabled": False, "selected_params": selected_params}

        console.print(
            "[cyan]Entrenando modelo base (fit en train)[/cyan] "
            f"| hidden={selected_params['nn_hidden_dims']} "
            f"| dropout={selected_params['nn_dropout']} "
            f"| lr={selected_params['nn_learning_rate']} "
            f"| wd={selected_params['nn_weight_decay']} "
            f"| batch_size={selected_params['nn_batch_size']} "
            f"| emb_cap={selected_params['nn_embedding_dim_cap']}"
        )

        wrapper_train = _build_wrapper(
            num_numeric=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            nn_hidden_dims=str(selected_params["nn_hidden_dims"]),
            nn_dropout=float(selected_params["nn_dropout"]),
            nn_learning_rate=float(selected_params["nn_learning_rate"]),
            nn_weight_decay=float(selected_params["nn_weight_decay"]),
            nn_batch_size=int(selected_params["nn_batch_size"]),
            nn_max_epochs=int(selected_params["nn_max_epochs"]),
            nn_patience=int(selected_params["nn_patience"]),
            nn_embedding_dim_cap=int(selected_params["nn_embedding_dim_cap"]),
            seed=seed,
            verbose=verbose_epochs,
        )
        wrapper_train.fit(train_prepared, val_prepared)

        pred_train = wrapper_train.predict(train_prepared) if include_train_metrics else None
        pred_val = wrapper_train.predict(val_prepared)
        pred_test = wrapper_train.predict(test_prepared)

        metrics_train = _compute_basic_metrics(train_prepared.y, pred_train) if pred_train is not None else None
        metrics_val = _compute_basic_metrics(val_prepared.y, pred_val)
        metrics_test = _compute_basic_metrics(test_prepared.y, pred_test)
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

        train_val_refit_metrics = None
        final_wrapper = wrapper_train
        final_model_training_data = "train"
        pdf_importance_ref = pdf_val
        prepared_importance_ref = val_prepared

        if refit_train_val:
            console.print("[cyan]Refit final con train+val[/cyan] (test permanece no visto hasta aquí)")
            pdf_train_val = pd.concat([pdf_train, pdf_val], ignore_index=True)
            train_val_prepared = _prepare_nn_data(
                pdf_train_val,
                num_cols=num_cols,
                cat_cols=cat_cols,
                target_col=target_col,
                num_stats=num_stats,
                cat_maps=cat_maps,
            )
            wrapper_train_val = _build_wrapper(
                num_numeric=len(num_cols),
                cat_cardinalities=cat_cardinalities,
                nn_hidden_dims=str(selected_params["nn_hidden_dims"]),
                nn_dropout=float(selected_params["nn_dropout"]),
                nn_learning_rate=float(selected_params["nn_learning_rate"]),
                nn_weight_decay=float(selected_params["nn_weight_decay"]),
                nn_batch_size=int(selected_params["nn_batch_size"]),
                nn_max_epochs=int(selected_params["nn_max_epochs"]),
                nn_patience=int(selected_params["nn_patience"]),
                nn_embedding_dim_cap=int(selected_params["nn_embedding_dim_cap"]),
                seed=seed,
                verbose=verbose_epochs,
            )
            wrapper_train_val.fit(train_val_prepared, test_prepared)

            pred_train_seen = wrapper_train_val.predict(train_prepared) if include_train_metrics else None
            pred_val_seen = wrapper_train_val.predict(val_prepared) if include_train_metrics else None
            pred_train_val_seen = wrapper_train_val.predict(train_val_prepared)
            pred_test_unseen = wrapper_train_val.predict(test_prepared)

            metrics_train_seen = _compute_basic_metrics(train_prepared.y, pred_train_seen) if pred_train_seen is not None else None
            metrics_val_seen = _compute_basic_metrics(val_prepared.y, pred_val_seen) if pred_val_seen is not None else None
            metrics_train_val_seen = _compute_basic_metrics(train_val_prepared.y, pred_train_val_seen)
            metrics_test_unseen = _compute_basic_metrics(test_prepared.y, pred_test_unseen)

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
            final_wrapper = wrapper_train_val
            final_model_training_data = "train_plus_val"
            pdf_importance_ref = pdf_test
            prepared_importance_ref = test_prepared

        all_data_metrics = None
        model_all_fp = None
        if fit_all_data:
            console.print(
                "[cyan]Entrenando modelo para despliegue con train+val+test[/cyan] "
                "(métricas de este modelo son solo in-sample)"
            )
            pdf_all = pd.concat([pdf_train, pdf_val, pdf_test], ignore_index=True)
            all_prepared = _prepare_nn_data(
                pdf_all, num_cols=num_cols, cat_cols=cat_cols, target_col=target_col, num_stats=num_stats, cat_maps=cat_maps
            )
            wrapper_all = _build_wrapper(
                num_numeric=len(num_cols),
                cat_cardinalities=cat_cardinalities,
                nn_hidden_dims=str(selected_params["nn_hidden_dims"]),
                nn_dropout=float(selected_params["nn_dropout"]),
                nn_learning_rate=float(selected_params["nn_learning_rate"]),
                nn_weight_decay=float(selected_params["nn_weight_decay"]),
                nn_batch_size=int(selected_params["nn_batch_size"]),
                nn_max_epochs=int(selected_params["nn_max_epochs"]),
                nn_patience=int(selected_params["nn_patience"]),
                nn_embedding_dim_cap=int(selected_params["nn_embedding_dim_cap"]),
                seed=seed,
                verbose=verbose_epochs,
            )
            wrapper_all.fit(all_prepared, None)
            pred_all = wrapper_all.predict(all_prepared)
            all_data_metrics = {
                "all_data_seen": _compute_basic_metrics(all_prepared.y, pred_all),
                "warning": (
                    "Métricas optimistas (entrenado y evaluado sobre los mismos datos). "
                    "No usar para estimar generalización."
                ),
            }
            model_all_fp = outputs_path / "neural_network_stress_model_all_data.pt"
            torch.save(
                {
                    "state_dict": wrapper_all.model.state_dict(),
                    "num_cols": num_cols,
                    "cat_cols": cat_cols,
                    "num_stats": num_stats,
                    "cat_maps": cat_maps,
                    "selected_params": selected_params,
                    "target_col": target_col,
                },
                model_all_fp,
            )

        model_fp = outputs_path / "neural_network_stress_model.pt"
        torch.save(
            {
                "state_dict": final_wrapper.model.state_dict(),
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "num_stats": num_stats,
                "cat_maps": cat_maps,
                "selected_params": selected_params,
                "target_col": target_col,
                "training_history": final_wrapper.training_history,
            },
            model_fp,
        )

        importance_fp = None
        if compute_importance:
            importance_df = _permutation_importance(
                final_wrapper,
                pdf_importance_ref,
                prepared_importance_ref,
                num_cols=num_cols,
                cat_cols=cat_cols,
                target_col=target_col,
                num_stats=num_stats,
                cat_maps=cat_maps,
                max_features=25,
                random_state=seed,
            )
            importance_fp = outputs_path / "neural_network_stress_feature_importance.csv"
            importance_df.to_csv(importance_fp, index=False)

        history_fp = outputs_path / "neural_network_training_history.csv"
        pd.DataFrame(final_wrapper.training_history).to_csv(history_fp, index=False)

        report = {
            "model": "EmbeddingMLPRegressor_torch",
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
                "n_numeric": len(num_cols),
                "n_categorical": len(cat_cols),
                "categorical_encoding": "learned_embeddings",
                "cat_cardinalities": cat_cardinalities,
            },
            "training_params": {
                "nn_hidden_dims": str(selected_params["nn_hidden_dims"]),
                "nn_dropout": float(selected_params["nn_dropout"]),
                "nn_learning_rate": float(selected_params["nn_learning_rate"]),
                "nn_weight_decay": float(selected_params["nn_weight_decay"]),
                "nn_batch_size": int(selected_params["nn_batch_size"]),
                "nn_max_epochs": int(selected_params["nn_max_epochs"]),
                "nn_patience": int(selected_params["nn_patience"]),
                "nn_embedding_dim_cap": int(selected_params["nn_embedding_dim_cap"]),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
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
                "model_pt": str(model_fp),
                "all_data_model_pt": None if model_all_fp is None else str(model_all_fp),
                "feature_importance_csv": None if importance_fp is None else str(importance_fp),
                "training_history_csv": str(history_fp),
            },
        }
        report_fp = outputs_path / "neural_network_stress_report.json"
        save_json(report, report_fp)

        console.print(f"[green]Modelo guardado[/green] -> {model_fp}")
        if model_all_fp is not None:
            console.print(f"[green]Modelo all-data guardado[/green] -> {model_all_fp}")
        if importance_fp is not None:
            console.print(f"[green]Importancias guardadas[/green] -> {importance_fp}")
        console.print(f"[green]Historial guardado[/green] -> {history_fp}")
        console.print(f"[green]Reporte guardado[/green] -> {report_fp}")

        print_done("NEURAL NETWORK SPARK+TORCH COMPLETADO")
        return report
    finally:
        for split_df in cached_dfs:
            if split_df is not None:
                split_df.unpersist(blocking=False)
        SparkManager.stop_session()


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena una red neuronal tabular con embeddings usando Spark + PyTorch.")
    p.add_argument(
        "--input-dir",
        default="data/aggregated/ex_stress/df_stress_zone_hour_day",
        help="Directorio del dataset de stress (parquets particionados).",
    )
    p.add_argument(
        "--outputs-dir",
        default="outputs/ejercicio2/neural_network",
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
    p.add_argument("--shuffle-partitions", type=int, default=200)
    p.add_argument("--nn-hidden-dims", default=None, help="Capas ocultas CSV, ej: 64,32")
    p.add_argument("--nn-dropout", type=float, default=None)
    p.add_argument("--nn-learning-rate", type=float, default=None)
    p.add_argument("--nn-weight-decay", type=float, default=None)
    p.add_argument("--nn-batch-size", type=int, default=None)
    p.add_argument("--nn-max-epochs", type=int, default=None)
    p.add_argument("--nn-patience", type=int, default=None)
    p.add_argument("--nn-embedding-dim-cap", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose-epochs", action="store_true")
    p.add_argument("--tune", action="store_true", help="Activa busqueda de hiperparametros sobre validacion temporal.")
    p.add_argument("--tune-metric", default="rmse", choices=["rmse", "mae", "r2"])
    p.add_argument("--tune-max-trials", type=int, default=6)
    p.add_argument("--tune-train-sample-frac", type=float, default=0.25)
    p.add_argument("--tune-val-sample-frac", type=float, default=0.50)
    p.add_argument("--tune-hidden-dims-values", default=None, help="Lista CSV para tuning (ej: 256,128,64|512,256,128) separada por ';'")
    p.add_argument("--tune-dropout-values", default=None, help="Lista CSV de dropout (ej: 0.15,0.25)")
    p.add_argument("--tune-learning-rate-values", default=None, help="Lista CSV de learning rates")
    p.add_argument("--tune-weight-decay-values", default=None, help="Lista CSV de weight decay")
    p.add_argument("--tune-batch-size-values", default=None, help="Lista CSV de batch size")
    p.add_argument("--tune-embedding-cap-values", default=None, help="Lista CSV de embedding dim cap")
    p.add_argument("--skip-train-metrics", action="store_true")
    p.add_argument("--refit-train-val", action="store_true", help="Opcional: reentrena con train+val antes de evaluar test. Más pesado.")
    p.add_argument("--fit-all-data", action="store_true")
    p.add_argument("--compute-importance", action="store_true", help="Calcula importancias por permutacion. Es lento; desactivado por defecto.")

    args = p.parse_args()
    tune_dropout_values = parse_csv_values(args.tune_dropout_values, float)
    tune_learning_rate_values = parse_csv_values(args.tune_learning_rate_values, float)
    tune_weight_decay_values = parse_csv_values(args.tune_weight_decay_values, float)
    tune_batch_size_values = parse_csv_values(args.tune_batch_size_values, int)
    tune_embedding_cap_values = parse_csv_values(args.tune_embedding_cap_values, int)
    tune_hidden_dims_values = None
    if args.tune_hidden_dims_values:
        tune_hidden_dims_values = [v.strip() for v in args.tune_hidden_dims_values.split(";") if v.strip()]

    run_neural_network_stress(
        input_dir=args.input_dir,
        outputs_dir=args.outputs_dir,
        target_col=args.target_col,
        time_col=args.time_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        extra_drop_cols=args.drop_col,
        shuffle_partitions=args.shuffle_partitions,
        nn_hidden_dims=args.nn_hidden_dims,
        nn_dropout=args.nn_dropout,
        nn_learning_rate=args.nn_learning_rate,
        nn_weight_decay=args.nn_weight_decay,
        nn_batch_size=args.nn_batch_size,
        nn_max_epochs=args.nn_max_epochs,
        nn_patience=args.nn_patience,
        nn_embedding_dim_cap=args.nn_embedding_dim_cap,
        tune_hyperparams=args.tune,
        tune_metric=args.tune_metric,
        tune_max_trials=args.tune_max_trials,
        tune_train_sample_frac=args.tune_train_sample_frac,
        tune_val_sample_frac=args.tune_val_sample_frac,
        tune_hidden_dims_values=tune_hidden_dims_values,
        tune_dropout_values=tune_dropout_values,
        tune_learning_rate_values=tune_learning_rate_values,
        tune_weight_decay_values=tune_weight_decay_values,
        tune_batch_size_values=tune_batch_size_values,
        tune_embedding_cap_values=tune_embedding_cap_values,
        include_train_metrics=not args.skip_train_metrics,
        refit_train_val=args.refit_train_val,
        fit_all_data=args.fit_all_data,
        compute_importance=args.compute_importance,
        seed=args.seed,
        verbose_epochs=args.verbose_epochs,
    )


if __name__ == "__main__":
    main()
