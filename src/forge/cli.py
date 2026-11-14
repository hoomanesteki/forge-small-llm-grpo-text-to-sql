"""Command line entry point: ``forge <group> <command>``."""

from __future__ import annotations

import typer

from forge import __version__

app = typer.Typer(
    name="forge",
    help="Forge: GRPO post-training for text-to-SQL on free GPUs.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Forge command line tools."""


@app.command()
def version() -> None:
    """Print the installed Forge version."""
    typer.echo(f"forge {__version__}")


if __name__ == "__main__":
    app()
