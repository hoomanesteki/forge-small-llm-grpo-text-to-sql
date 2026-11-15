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
sql_app = typer.Typer(
    help="Run SQL the way evaluation and rewards do: read-only, one statement, with limits.",
    no_args_is_help=True,
)
app.add_typer(sql_app, name="sql")
eval_app = typer.Typer(
    help="Check the checkers: gold scored against itself, and fast against official EX.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")


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


# Spelled out rather than imported from forge.data, which would load the data stack
# for every command; a unit test keeps each one in step with its source.
DataSource = Literal["hf", "mirror", "gdrive"]
SplitName = Literal["train", "val", "dev"]
PromptVersionName = Literal["p1", "p2"]
# The agreement check mutates training-side gold queries only (plan FG-108).
AgreementSplitName = Literal["train", "val"]


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
        bool,
        typer.Option(
            "--force",
            help="Extract every file again, replacing any that differ from spider.sha256.",
        ),
    ] = False,
) -> None:
    """Get Spider into FORGE_DATA_DIR (default data/spider) and verify every file.

    The zip is kept in data/download. Files that are there but differ are only replaced
    with --force.
    """
    from forge.data.fetch import FetchError, fetch
    from forge.data.spider import resolve_data_dir
    from forge.ops.paths import RepoRootNotFound

    try:
        data_dir = resolve_data_dir()
        result = fetch(data_dir, zip_path=from_zip, source=source, force=force)
    # Downloads and extraction raise FetchError; OSError is reading files that are there.
    except (FetchError, OSError, RepoRootNotFound, ValueError) as exc:
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


@data_app.command("prompt-stats")
def data_prompt_stats(
    tokenizer: Annotated[
        str,
        typer.Option(
            help="Model on the Hub, or a local directory, whose tokenizer and chat template "
            "count the tokens."
        ),
    ],
    split: Annotated[SplitName, typer.Option(help="Whose prompts to measure.")] = "train",
    prompt_version: Annotated[
        PromptVersionName, typer.Option(help="p1, or p2 with sample rows.")
    ] = "p1",
) -> None:
    """Prompt token lengths, and what the overflow policy does past 1,536 tokens.

    It reads questions and schemas only, so it is not a dev evaluation.
    """
    from forge.data.prompts import (
        MAX_PROMPT_TOKENS,
        chat_token_counter,
        fit_prompt,
        load_tokenizer,
        prompt_stats,
        render_prompt_stats,
    )
    from forge.data.spider import load_raw, resolve_data_dir
    from forge.data.splits import split_examples
    from forge.ops.paths import RepoRootNotFound

    try:
        data_dir = resolve_data_dir()
        examples = split_examples(load_raw(data_dir))[split]
        if not examples:
            raise ValueError(f"the {split} split has no questions in {data_dir}")
        count = chat_token_counter(load_tokenizer(tokenizer))
        fitted = [
            fit_prompt(example, prompt_version, count, data_dir=data_dir) for example in examples
        ]
    # OSError covers missing data (DataNotFound) and a tokenizer that can't be loaded.
    except (ImportError, OSError, RepoRootNotFound, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"data: {data_dir}")
    typer.echo(f"split {split}, prompt {prompt_version}, tokenizer {tokenizer}\n")
    typer.echo(render_prompt_stats(prompt_stats(fitted, max_tokens=MAX_PROMPT_TOKENS)))


@sql_app.command("run")
def sql_run(
    sql: Annotated[str, typer.Argument(help="One SQL statement.")],
    db: Annotated[str, typer.Option("--db", help="Spider database id, such as concert_singer.")],
    timeout_ms: Annotated[int, typer.Option(min=1, help="Stop the query after this long.")] = 2000,
    max_rows: Annotated[int, typer.Option(min=0, help="Return at most this many rows.")] = 10_000,
) -> None:
    """Run SQL on a database in FORGE_DATA_DIR and print the result; exit 1 if it fails."""
    from forge.data.spider import DataNotFound
    from forge.ops.paths import RepoRootNotFound
    from forge.sql.executor import SqliteExecutor

    try:
        with SqliteExecutor() as executor:
            result = executor.run(db, sql, timeout_ms=timeout_ms, max_rows=max_rows)
    except (DataNotFound, RepoRootNotFound) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    timing = f"({result.ms:.1f} ms)"
    if not result.ok:
        typer.echo(f"{result.error_kind}: {result.error} {timing}")
        raise typer.Exit(1)
    rows = result.rows or []
    if result.columns:
        typer.echo(" | ".join(result.columns))
    for row in rows:
        typer.echo(" | ".join(map(_sql_value, row)))
    summary = f"{len(rows)} row{'' if len(rows) == 1 else 's'}"
    if result.truncated:
        summary += f", truncated at --max-rows {max_rows}"
    typer.echo(f"{summary} {timing}")


def _sql_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return f"X'{value.hex()}'"
    return str(value)


@eval_app.command("gold")
def eval_gold(
    split: Annotated[SplitName, typer.Option(help="Whose gold queries to score.")] = "val",
) -> None:
    """Score each gold query against itself with both checkers; exit 1 unless both give 100%.

    The known failing golds are skipped. It runs no model, so a dev check is not one of
    the six dev evaluations.
    """
    from forge.data.spider import FAILING_GOLD_IDS, DataNotFound, load_raw, resolve_data_dir
    from forge.data.splits import split_examples
    from forge.eval.agreement import GOLD_TIMEOUT_MS, gold_mutations, score
    from forge.eval.testsuite import OfficialEvalTimeout
    from forge.ops.paths import RepoRootNotFound

    try:
        data_dir = resolve_data_dir()
        examples = split_examples(load_raw(data_dir))[split]
        kept = [example for example in examples if example.id not in FAILING_GOLD_IDS]
        if not kept:
            raise ValueError(f"the {split} split has no questions in {data_dir}")
        scores = score(gold_mutations(kept), data_dir=data_dir, pred_timeout_ms=GOLD_TIMEOUT_MS)
    # ValueError covers GoldQueryFailed, a gold query that fails on its database.
    except (DataNotFound, OfficialEvalTimeout, RepoRootNotFound, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    databases = len({example.db_id for example in kept})
    skipped = len(examples) - len(kept)
    typer.echo(f"data: {data_dir}")
    typer.echo(
        f"split {split}: {len(kept)} gold queries over {databases} "
        f"database{'' if databases == 1 else 's'}"
        + (f", {skipped} known failing golds skipped" if skipped else "")
    )
    for name, hits in (
        ("fast EX", sum(s.fast for s in scores)),
        ("official EX", sum(s.official for s in scores)),
    ):
        typer.echo(f"{name:<11}  {hits:>5}/{len(scores)}  {hits / len(scores):.2%}")
    misses = [s for s in scores if not (s.fast and s.official)]
    if misses:
        typer.echo(f"error: {len(misses)} gold queries don't match themselves:", err=True)
        for s in misses[:10]:
            typer.echo(
                f"  {s.mutation.example_id} ({s.mutation.db_id}): fast {s.fast}, "
                f"official {s.official}: {s.mutation.gold_sql}",
                err=True,
            )
        raise typer.Exit(1)


@eval_app.command("agreement")
def eval_agreement(
    mutations: Annotated[
        int, typer.Option(min=1, help="How many predictions to make from gold queries.")
    ] = 2000,
    split: Annotated[
        AgreementSplitName, typer.Option(help="Whose gold queries to mutate.")
    ] = "val",
    seed: Annotated[int, typer.Option(help="Seed for picking and changing the queries.")] = 0,
    out: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False, help="Write the report here, not docs/audit/agreement-<date>.md."
        ),
    ] = None,
) -> None:
    """Fast against official EX on mutated gold queries; exit 1 under 98% agreement (I5)."""
    from datetime import UTC, datetime

    from forge.data.manifest import data_manifest_hash
    from forge.data.spider import FAILING_GOLD_IDS, load_raw, resolve_data_dir
    from forge.data.splits import split_examples
    from forge.eval.agreement import (
        AGREEMENT_TARGET,
        AgreementRun,
        agreement_mutations,
        agreement_rate,
        render_report,
        score,
    )
    from forge.eval.testsuite import OfficialEvalTimeout
    from forge.ops.paths import RepoRootNotFound, repo_root

    try:
        data_dir = resolve_data_dir()
        examples = [
            example
            for example in split_examples(load_raw(data_dir))[split]
            if example.id not in FAILING_GOLD_IDS
        ]
        if not examples:
            raise ValueError(f"the {split} split has no questions in {data_dir}")
        scores = score(
            agreement_mutations(examples, mutations, seed=seed, data_dir=data_dir),
            data_dir=data_dir,
        )
        day = datetime.now(UTC).date()
        report = out or repo_root() / "docs" / "audit" / f"agreement-{day.isoformat()}.md"
        run = AgreementRun(
            command=f"forge eval agreement --mutations {mutations} --split {split} --seed {seed}",
            split=split,
            examples=len(examples),
            databases=len({example.db_id for example in examples}),
            seed=seed,
            day=day,
            git_sha=_report_git_sha(),
            data_hash=data_manifest_hash(data_dir),
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(scores, run), encoding="utf-8")
    # OSError covers a missing database (DataNotFound) and a report that can't be written;
    # ValueError covers GoldQueryFailed.
    except (OSError, OfficialEvalTimeout, RepoRootNotFound, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    rate = agreement_rate(scores)
    agreeing = sum(s.agree for s in scores)
    typer.echo(f"data: {data_dir}")
    typer.echo(
        f"split {split}: {run.examples} questions over {run.databases} "
        f"database{'' if run.databases == 1 else 's'}, {len(scores)} predictions, seed {seed}"
    )
    typer.echo(
        f"fast EX {sum(s.fast for s in scores)}, official EX {sum(s.official for s in scores)}"
    )
    typer.echo(f"agreement {rate:.2%} ({agreeing} of {len(scores)})")
    typer.echo(f"report: {report}")
    if rate < AGREEMENT_TARGET:
        typer.echo(f"error: agreement is under the {AGREEMENT_TARGET:.0%} that I5 needs", err=True)
        raise typer.Exit(1)


def _report_git_sha() -> str:
    """The commit a report came from, marked dirty when the tree has changes."""
    import subprocess

    from forge.ops.manifest import git_sha
    from forge.ops.paths import RepoRootNotFound, repo_root

    try:
        return git_sha(repo_root(), allow_dirty=True)
    except (OSError, RepoRootNotFound, subprocess.CalledProcessError):
        # A wheel installed outside a checkout (a Kaggle kernel) has no history to name.
        return "unknown (not run from a git checkout)"


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
