from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPORTS_ROOT = Path("outputs/ejercicio2/reports")
OUT_DIR = Path("outputs/ejercicio2/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_standard_metrics(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("rmse_test") is not None:
        return {
            "rmse_test": data.get("rmse_test"),
            "mae_test": data.get("mae_test"),
            "r2_test": data.get("r2_test"),
            "accuracy_test": data.get("accuracy_test"),
            "f1_test": data.get("f1_test"),
        }

    metrics = data.get("metrics", {})
    test_metrics = (
        metrics.get("train_val_refit", {}).get("test_unseen")
        or metrics.get("train_fit", {}).get("test_unseen")
        or {}
    )

    return {
        "rmse_test": test_metrics.get("rmse"),
        "mae_test": test_metrics.get("mae"),
        "r2_test": test_metrics.get("r2"),
        "accuracy_test": None,
        "f1_test": None,
    }


rows = []

for path in REPORTS_ROOT.rglob("*report*.json"):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "baselines" in data:
        for baseline_name, baseline_data in data["baselines"].items():
            reg = baseline_data.get("test", {}).get("regression", {})
            clf = baseline_data.get("test", {}).get("classification", {})

            rows.append({
                "model": f"baseline_{baseline_name}",
                "target": data.get("target") or data.get("target_col") or data.get("target_regression"),
                "rmse_test": reg.get("rmse"),
                "mae_test": reg.get("mae"),
                "r2_test": reg.get("r2"),
                "accuracy_test": clf.get("accuracy"),
                "f1_test": clf.get("f1"),
                "report_file": str(path),
            })
        continue

    metrics = extract_standard_metrics(data)

    rows.append({
        "model": data.get("model", path.stem),
        "target": data.get("target") or data.get("target_col") or data.get("target_regression"),
        "rmse_test": metrics["rmse_test"],
        "mae_test": metrics["mae_test"],
        "r2_test": metrics["r2_test"],
        "accuracy_test": metrics["accuracy_test"],
        "f1_test": metrics["f1_test"],
        "report_file": str(path),
    })

df = pd.DataFrame(rows)

df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

best = df.dropna(subset=["rmse_test"]).sort_values("rmse_test").head(1)

result = {
    "models": rows,
    "best_model_by_rmse": best.to_dict(orient="records")[0] if not best.empty else None,
}

with (OUT_DIR / "model_comparison.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("Comparación exportada en:", OUT_DIR)