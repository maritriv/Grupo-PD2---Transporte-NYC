from __future__ import annotations

from typing import Literal

FeatureMode = Literal["operational", "predictive"]

TARGET_REG = "stress_score"
TARGET_CLF = "is_stress"

TARGET_COLUMNS = [TARGET_REG, TARGET_CLF]

# Variables contemporáneas o demasiado cercanas al target.
# Fuera en modo predictivo.
PREDICTIVE_EXCLUDED_COLUMNS = [
    "num_trips",
    "avg_price",
    "std_price",
    "price_variability",
    "price_variability_abs",
    "price_variability_rel",
    "biz_score",
    "biz_score_iqr",
    "biz_score_zsum",
    "biz_score_zproduct",
]

# Variables que no son útiles para modelado directo
COMMON_FORCED_DROP = [
    "date",
]


def get_forced_drop_columns(mode: FeatureMode) -> list[str]:
    if mode == "operational":
        return COMMON_FORCED_DROP.copy()

    if mode == "predictive":
        return COMMON_FORCED_DROP + PREDICTIVE_EXCLUDED_COLUMNS

    raise ValueError(f"Modo no soportado: {mode}")