from __future__ import annotations

from itertools import product
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def parse_csv_values(raw: str | None, caster: Callable[[str], T]) -> list[T] | None:
    if raw is None:
        return None
    values: list[T] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(caster(token))
    if not values:
        raise ValueError("La lista de valores está vacía.")
    return values


def is_higher_better_regression_metric(metric_name: str) -> bool:
    if metric_name not in {"mae", "rmse", "r2"}:
        raise ValueError(f"Métrica de tuning no soportada: {metric_name}")
    return metric_name == "r2"


def take_evenly_spaced(items: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k <= 0:
        raise ValueError("k debe ser > 0.")
    if len(items) <= k:
        return items
    if k == 1:
        return [items[0]]

    idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    uniq_idxs: list[int] = []
    seen: set[int] = set()
    for idx in idxs:
        if idx not in seen:
            uniq_idxs.append(idx)
            seen.add(idx)
    return [items[idx] for idx in uniq_idxs]


def build_param_grid_candidates(
    *,
    grid: dict[str, list[Any]],
    max_trials: int,
    base_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    combos = [dict(zip(keys, values_tuple)) for values_tuple in product(*(grid[k] for k in keys))]

    if base_params is not None and base_params not in combos:
        combos.insert(0, base_params.copy())

    candidates = take_evenly_spaced(combos, max_trials)
    if base_params is not None and base_params not in candidates:
        candidates = [base_params.copy()] + candidates[:-1]
    return candidates
