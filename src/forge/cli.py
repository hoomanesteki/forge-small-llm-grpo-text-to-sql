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
ledger_app = typer.Typer(help="The GPU compute ledger.", no_args_is_help=True)
app.add_typer(ledger_app, name="ledger")


@app.callback()
def main() -> None:
    """Forge command line tools."""


@app.command()
def version() -> None:
    """Print the installed Forge version."""
    typer.echo(f"forge {__version__}")


@app.command()
def doctor() -> None:
    """Check Python, uv, Kaggle and Hugging Face auth, the run store and free disk."""
    from forge.ops.doctor import Context, exit_code, render, run_doctor

    ctx = Context()
    checks = run_doctor(ctx)
    typer.echo(render(checks, ctx.env))
    raise typer.Exit(exit_code(checks))


@ledger_app.command("add")
def ledger_add(
    platform: str = typer.Option(..., help="kaggle, colab or modal"),
    hours: float = typer.Option(..., help="wall-clock GPU hours"),
    run_id: str = typer.Option(..., "--run-id"),
    purpose: str = typer.Option(...),
    gpu: str = typer.Option("T4"),
    cost: float = typer.Option(0.0, help="cash spent in USD"),
    on: str = typer.Option("", help="session date, YYYY-MM-DD (default today)"),
) -> None:
    """Add a row by hand, for sessions the CLI didn't launch (for example Colab)."""
    from datetime import date

    from forge.ops.ledger import LedgerRow, append

    day = date.fromisoformat(on) if on else date.today()
    append(LedgerRow(day, platform, gpu, hours, run_id, purpose, cost))
    typer.echo(f"added {hours:.2f} h on {platform} for {run_id}")


@ledger_app.command("summary")
def ledger_summary(days: int = typer.Option(7, help="quota window in days")) -> None:
    """Hours this week against the Kaggle quota, total GPU hours and cash spent."""
    from forge.ops.ledger import summary

    typer.echo(summary(window_days=days).render())


if __name__ == "__main__":
    app()
