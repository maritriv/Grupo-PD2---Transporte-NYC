from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # Fallback conservador para mantener compatibilidad si no se encuentran marcadores.
    return current.parents[4]


def resolve_project_path(path_from_root: str | Path) -> Path:
    return (get_project_root() / Path(path_from_root)).resolve()


def ensure_project_dir(path_from_root: str | Path) -> Path:
    out_dir = resolve_project_path(path_from_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(data: dict[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
