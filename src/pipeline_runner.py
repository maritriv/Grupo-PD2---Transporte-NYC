from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def print_stage(title: str, subtitle: str | None = None, color: str = "cyan") -> None:
    text = title if subtitle is None else f"{title}\n[dim]{subtitle}[/dim]"
    console.print()
    console.print(Panel.fit(f"[bold white]{text}[/bold white]", border_style=color))


def ejecutar_modulo(modulo: str, project_root: Path) -> None:
    """
    Ejecuta un modulo Python (python -m ...) con el interprete activo,
    desde la raiz del proyecto para que las rutas relativas funcionen.
    """
    console.print()
    console.print(f"[bold white][INFO][/bold white] Ejecutando: [bold]{modulo}[/bold]")
    console.print("[dim]" + "-" * 60 + "[/dim]")

    result = subprocess.run(
        [sys.executable, "-m", modulo],
        cwd=str(project_root),
    )

    if result.returncode != 0:
        console.print(f"[bold red][ERROR][/bold red] Error ejecutando {modulo}")
        raise RuntimeError(f"Error ejecutando {modulo}")


def print_done(message: str) -> None:
    console.print()
    console.print(Panel.fit(f"[bold green]{message}[/bold green]", border_style="green"))
