from __future__ import annotations

import pandas as pd

from src.ml.dataset.modules.display import print_step_status


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera lags, rolling y variables de calendario."""
    df = df.sort_values(["service_type", "pu_location_id", "date", "hour"])
    grp = df.groupby(["service_type", "pu_location_id"], sort=False)

    df["lag_1h_trips"] = grp["num_trips"].shift(1)
    df["lag_24h_trips"] = grp["num_trips"].shift(24)
    df["roll_3h_trips"] = grp["num_trips"].rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.astype(int)
    df["month"] = pd.to_datetime(df["date"]).dt.month.astype(int)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    df[["lag_1h_trips", "lag_24h_trips"]] = df[["lag_1h_trips", "lag_24h_trips"]].fillna(0)

    print_step_status("Features temporales", f"generadas | {len(df.columns)} columnas actuales")
    return df