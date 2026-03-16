from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.table import Table

from src.pipeline_runner import console


def print_step_status(step: str, message: str) -> None:
    console.print(f"[cyan]{step}[/cyan] -> {message}")


def print_rich_table_preview(
    df: pd.DataFrame,
    title: str,
    cols: list[str] | None = None,
    n: int = 10,
    max_col_width: int = 16,
) -> None:
    """Muestra una preview del dataframe como tabla Rich, limitada y legible."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")

    if df.empty:
        console.print("[yellow]DataFrame vacío[/yellow]")
        return

    preview = df.copy()
    if cols is not None:
        existing_cols = [c for c in cols if c in preview.columns]
        if existing_cols:
            preview = preview[existing_cols]

    preview = preview.head(n)

    table = Table(show_header=True, header_style="bold magenta")
    for col in preview.columns:
        table.add_column(str(col), overflow="fold", max_width=max_col_width)

    for _, row in preview.iterrows():
        values = []
        for val in row:
            if pd.isna(val):
                values.append("NA")
            else:
                text = str(val)
                if len(text) > max_col_width:
                    text = text[: max_col_width - 3] + "..."
                values.append(text)
        table.add_row(*values)

    console.print(f"[white]Rows:[/white] {len(df):,} | [white]Cols:[/white] {len(df.columns):,}")
    console.print(table)


def print_build_summary(df: pd.DataFrame, out_fp: Path) -> None:
    summary = Table(title="Resumen build_dataset", header_style="bold cyan")
    summary.add_column("Métrica", style="bold white")
    summary.add_column("Valor")
    summary.add_row("Rows", f"{len(df):,}")
    summary.add_row("Cols", f"{len(df.columns):,}")
    summary.add_row("Output", str(out_fp))
    console.print(summary)