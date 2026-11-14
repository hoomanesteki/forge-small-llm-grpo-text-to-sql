"""Command line entry point: ``forge <group> <command>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from forge import __version__

app = typer.Typer(
    name="forge",
    help="Forge: GRPO post-training for text-to-SQL on free GPUs.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(
    help="Spider data: fetch, verify, split and summarize.", no_args_is_help=True
)
app.add_typer(data_app, name="data")
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
    typer.echo(render(checks, ctx.env, sorted(ctx.secrets)))
    raise typer.Exit(exit_code(checks))


# Spelled out rather than imported from forge.data.fetch, which would load the data
# stack for every command; a unit test keeps the two in step.
DataSource = Literal["hf", "mirror", "gdrive"]


@data_app.command("fetch")
def data_fetch(
    from_zip: Annotated[
        Path | None,
        typer.Option("--from-zip", exists=True, dir_okay=False, help="Use this Spider zip."),
    ] = None,
    source: Annotated[
        DataSource,
        typer.Option(help="Download from the pinned Hub dataset, its mirror, or Google Drive."),
    ] = "hf",
    force: Annotated[
        bool, typer.Option("--force", help="Extract again even if every file verifies.")
    ] = False,
) -> None:
    """Get Spider into FORGE_DATA_DIR (default data/spider) and verify every file."""
    from forge.data.fetch import FetchError, fetch
    from forge.data.spider import resolve_data_dir
    from forge.ops.paths import RepoRootNotFound

    try:
        data_dir = resolve_data_dir()
        result = fetch(data_dir, zip_path=from_zip, source=source, force=force)
    except (FetchError, RepoRootNotFound, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if result.zip_path is None:
        typer.echo(f"{data_dir}: all {result.files} files match spider.sha256, nothing to do")
    else:
        typer.echo(
            f"extracted {result.files} files from {result.zip_path} into {data_dir}; "
            "all match spider.sha256"
        )


@data_app.command("stats")
def data_stats() -> None:
    """Questions by split, database and hardness, and the data manifest hash."""
    from forge.data.manifest import data_manifest_hash
    from forge.data.spider import DataNotFound, load_raw, resolve_data_dir
    from forge.data.splits import split_examples
    from forge.data.stats import render_stats
    from forge.ops.paths import RepoRootNotFound

    try:
        data_dir = resolve_data_dir()
        splits = split_examples(load_raw(data_dir))
        digest = data_manifest_hash(data_dir)
    except (DataNotFound, RepoRootNotFound, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"data: {data_dir}\n")
    typer.echo(render_stats([example for split in splits.values() for example in split]))
    typer.echo(f"\ndata manifest hash: {digest}")


@data_app.command("splits")
def data_splits(
    write: Annotated[
        bool,
        typer.Option(
            "--write", help="Rewrite src/forge/data/splits/ from the full, verified download."
        ),
    ] = False,
) -> None:
    """Questions and databases per split, checked against the committed split manifests."""
    from forge.data.spider import (
        ChecksumMismatch,
        DataNotFound,
        load_checksums,
        load_raw,
        resolve_data_dir,
        verify_files,
    )
    from forge.data.splits import (
        MANIFEST_DIR,
        SPLITS,
        VAL_SEED,
        render_manifests,
        split_examples,
        write_manifests,
    )
    from forge.ops.paths import RepoRootNotFound, repo_root

    try:
        data_dir = resolve_data_dir()
        splits = split_examples(load_raw(data_dir))
        if write:
            # The committed split is the full download's: a subset must never replace it.
            verify_files(data_dir, load_checksums())
            out_dir = repo_root() / "src" / "forge" / "data" / "splits"
            written = write_manifests(splits, out_dir)
    except ChecksumMismatch as exc:
        typer.echo(
            f"error: --write needs the full Spider download, but {len(exc.files)} file(s) "
            "are missing or differ from spider.sha256; run `forge data fetch` or point "
            "FORGE_DATA_DIR at it",
            err=True,
        )
        raise typer.Exit(1) from exc
    except (DataNotFound, RepoRootNotFound, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"data: {data_dir}\n")
    typer.echo(f"{'split':<5}  {'questions':>9}  {'databases':>9}")
    for name in SPLITS:
        databases = {example.db_id for example in splits[name]}
        typer.echo(f"{name:<5}  {len(splits[name]):>9}  {len(databases):>9}")
    val_dbs = sorted({example.db_id for example in splits["val"]})
    typer.echo(f"\nval databases (seed {VAL_SEED}): {', '.join(val_dbs)}")

    if write:
        typer.echo(f"\nwrote {', '.join(path.name for path in written)} to {out_dir}")
        return
    stale = [
        name
        for name, text in render_manifests(splits).items()
        if not (MANIFEST_DIR / name).is_file()
        or (MANIFEST_DIR / name).read_bytes() != text.encode()
    ]
    if stale:
        typer.echo(
            f"error: {', '.join(stale)} differ from this split; the committed manifests hold "
            "the full download's split, so point FORGE_DATA_DIR at it, or rewrite them with "
            "--write if the split rule changed",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo("\nmatches the committed split manifests")


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
