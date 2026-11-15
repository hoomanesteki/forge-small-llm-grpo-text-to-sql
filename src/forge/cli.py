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
    help="Evaluate a model on a split, and check the checkers: gold scored against itself, "
    "and fast against official EX.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")
report_app = typer.Typer(help="Reports from runs' stored files.", no_args_is_help=True)
app.add_typer(report_app, name="report")


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
BackendName = Literal["hf", "vllm", "llamacpp", "fake"]
DeviceName = Literal["cpu", "mps"]
StoreName = Literal["local", "hub"]

# Printed by the commands that read dev without a model, so nobody logs them as dev uses
# or holds back from running them.
NOT_A_DEV_USE = (
    "note: no model output on dev, so this is not one of the six dev evaluations (ADR-0004)"
)


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

    It reads questions and schemas only, so --split dev is not a dev evaluation.
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
    if split == "dev":
        typer.echo(NOT_A_DEV_USE)
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


def _eval_config_file(ctx: typer.Context, path: Path | None) -> Path | None:
    """Make the options an eval config file sets the defaults of ``forge eval run``.

    click's default map holds them, so an option given on the command line still wins,
    and each value is checked by its option's type as if it had been typed.
    """
    if path is None:
        return None
    from forge.eval.config import read_config_file

    try:
        options = read_config_file(path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    ctx.default_map = {**(ctx.default_map or {}), **options}
    return path


@eval_app.command("run")
def eval_run(
    model: Annotated[
        str,
        typer.Option(
            help="Hub repo id (with @revision to pin it) or local directory; for --backend "
            "fake, gold or select-1."
        ),
    ],
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            is_eager=True,
            callback=_eval_config_file,
            help="YAML file of options, such as configs/eval/*.yaml, keyed by option name "
            "(prompt_version for --prompt-version); options given here override it.",
        ),
    ] = None,
    adapter: Annotated[
        str | None, typer.Option(help="LoRA adapter to load on the model, Hub id or directory.")
    ] = None,
    backend: Annotated[
        BackendName, typer.Option(help="hf runs transformers; fake answers without a model.")
    ] = "hf",
    device: Annotated[
        DeviceName,
        typer.Option(
            help="Where hf runs the model: cpu, or mps for Apple silicon's GPU, which makes "
            "the run a local-mps pilot (ADR-0016)."
        ),
    ] = "cpu",
    split: Annotated[SplitName, typer.Option(help="Which questions to answer.")] = "val",
    reason: Annotated[
        str | None,
        typer.Option(help="Why dev is evaluated: required for --split dev, and logged."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Answer only the first N questions: by id on val and dev, in file order on "
            "train (train_spider before the rowless train_others).",
        ),
    ] = None,
    prompt_version: Annotated[
        PromptVersionName, typer.Option(help="p1, or p2 with sample rows.")
    ] = "p1",
    n: Annotated[int, typer.Option("--n", min=1, help="Samples per question.")] = 1,
    temperature: Annotated[
        float, typer.Option(min=0.0, help="0 decodes greedily; --n above 1 needs more.")
    ] = 0.0,
    retry: Annotated[
        int,
        typer.Option(
            min=0, max=1, help="1 asks once more, with the error, when a query fails to run."
        ),
    ] = 0,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Name the run, or resume the run of that name."),
    ] = None,
    store: Annotated[StoreName, typer.Option(help="Where the run's files go.")] = "local",
    # Left out of the manifest: how many processes run the SQL changes no score.
    workers: Annotated[
        int | None,
        typer.Option(
            min=1, help="How many processes run the predicted SQL; by default one per CPU, up to 4."
        ),
    ] = None,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty", help="Run on uncommitted changes; the manifest's SHA says -dirty."
        ),
    ] = False,
) -> None:
    """Generate, execute and score a split; resume a run by passing its --run-id.

    With --config, a file in configs/eval/ gives the options it sets. Files go to
    runs/<run_id>/ (FORGE_RUNS_DIR): manifest.json, config.yaml and
    eval/<split>-generations.jsonl and -scores.jsonl. Exits 1 if the run fails.
    """
    from forge.eval.config import EvalConfig, NotAvailable
    from forge.eval.run_files import RunConflict, RunCorrupt, RunLocked
    from forge.eval.runner import WORKERS, run_eval
    from forge.ops.dev_usage import DevBudgetExceeded
    from forge.ops.manifest import DirtyTreeError, GitError
    from forge.ops.paths import RepoRootNotFound
    from forge.sql.sandbox import SandboxError

    config = EvalConfig(
        model=model,
        adapter=adapter,
        backend=backend,
        device=device,
        split=split,
        reason=reason,
        limit=limit,
        prompt_version=prompt_version,
        n=n,
        temperature=temperature,
        retry=retry,
        run_id=run_id,
        store=store,
    )

    def progress(done: int, total: int) -> None:
        typer.echo(f"scored {done} of {total}", err=True)

    try:
        run = run_eval(
            config,
            allow_dirty=allow_dirty,
            workers=WORKERS if workers is None else workers,
            progress=progress,
        )
    # OSError covers missing data (DataNotFound), a model that can't be found and the
    # official evaluator's overall timeout; ValueError a wrong option, a failing gold
    # query (GoldFailed) and RunCorrupt; ImportError the cpu extra not installed.
    except (
        DevBudgetExceeded,
        DirtyTreeError,
        GitError,
        ImportError,
        NotAvailable,
        OSError,
        RepoRootNotFound,
        RunConflict,
        RunCorrupt,
        RunLocked,
        SandboxError,
        ValueError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    scores = run.scores
    samples = f"{n} samples" if n > 1 else "1 sample"
    typer.echo(f"run {run.manifest.run_id}: {run.paths.root}")
    if config_file is not None:
        typer.echo(f"options from {config_file}, and the command line's over them")
    typer.echo(
        f"split {split}: {run.questions} questions, {samples} each, prompt {prompt_version}, "
        f"backend {backend}, model {model}" + (f", adapter {adapter}" if adapter else "")
    )
    typer.echo(f"platform {run.manifest.platform}")
    if run.resumed:
        typer.echo(f"resumed: {run.resumed} scores were stored by an earlier attempt")
    if len(scores) > run.resumed:
        typer.echo(f"generated {len(scores) - run.resumed} samples in {run.generate_s:.1f} s")
    for name, hits in (
        ("valid", sum(s.valid for s in scores)),
        ("fast EX", sum(s.ex_fast for s in scores)),
        ("official EX", sum(s.ex_official is True for s in scores)),
    ):
        typer.echo(f"{name:<11}  {hits:>5}/{len(scores)}  {hits / len(scores):.2%}")
    if retry:
        from forge.eval.retry import without_retry

        asked = sum(s.attempts > 1 for s in scores)
        first = sum(without_retry(s).ex_official is True for s in scores)
        typer.echo(
            f"retried {asked} of {len(scores)} samples whose first query failed to run; "
            f"official EX without retry {first}/{len(scores)}  {first / len(scores):.2%}"
        )


@eval_app.command("gold")
def eval_gold(
    split: Annotated[SplitName, typer.Option(help="Whose gold queries to score.")] = "val",
) -> None:
    """Score each gold query against itself with both checkers; exit 1 unless both give 100%.

    The known failing golds are skipped. It runs no model, so --split dev is not one of
    the six dev evaluations.
    """
    from forge.data.spider import FAILING_GOLD_IDS, DataNotFound, load_raw, resolve_data_dir
    from forge.data.splits import split_examples
    from forge.eval.agreement import GOLD_TIMEOUT_MS, gold_mutations, score
    from forge.eval.testsuite import OfficialEvalTimeout
    from forge.ops.paths import RepoRootNotFound
    from forge.sql.sandbox import SandboxError

    try:
        data_dir = resolve_data_dir()
        examples = split_examples(load_raw(data_dir))[split]
        kept = [example for example in examples if example.id not in FAILING_GOLD_IDS]
        if not kept:
            raise ValueError(f"the {split} split has no questions in {data_dir}")
        scores = score(gold_mutations(kept), data_dir=data_dir, pred_timeout_ms=GOLD_TIMEOUT_MS)
    # ValueError covers GoldQueryFailed, a gold query that fails on its database;
    # SandboxError a sandbox pool that can't start its workers.
    except (DataNotFound, OfficialEvalTimeout, RepoRootNotFound, SandboxError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    databases = len({example.db_id for example in kept})
    skipped = len(examples) - len(kept)
    typer.echo(f"data: {data_dir}")
    if split == "dev":
        typer.echo(NOT_A_DEV_USE)
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
    from forge.sql.sandbox import SandboxError

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
    # ValueError covers GoldQueryFailed; SandboxError a pool that can't start its workers.
    except (OSError, OfficialEvalTimeout, RepoRootNotFound, SandboxError, ValueError) as exc:
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


def _report_errors() -> tuple[type[Exception], ...]:
    """What reading runs for a report can raise: OSError for a missing run, report or data
    (DataNotFound), ValueError for a corrupt run (RunCorrupt), one made on other data or
    runs that can't be compared, and RunLocked while another process writes a run."""
    from forge.eval.run_files import RunLocked, RunUnfinished
    from forge.ops.paths import RepoRootNotFound

    return (OSError, RepoRootNotFound, RunLocked, RunUnfinished, ValueError)


@report_app.command("run")
def report_run(
    run_id: Annotated[str, typer.Argument(help="The run, a folder in runs/ (FORGE_RUNS_DIR).")],
) -> None:
    """Write runs/<run_id>/eval/<split>-report.md and its by-hardness chart.

    Everything in it comes from the run's stored files and the questions of its split, in
    data that must hash to the manifest's data hash (FORGE_DATA_DIR).
    """
    from forge.eval.report import write_run_report

    try:
        written = write_run_report(run_id)
    except _report_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"report: {written.report}")
    typer.echo(f"chart: {written.chart}")


@report_app.command("compare")
def report_compare(
    a: Annotated[str, typer.Argument(help="The run compared against, such as the base model.")],
    b: Annotated[str, typer.Argument(help="The run compared with it, on the same questions.")],
    out: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="Write here, not runs/<b>/eval/compare-<a>.md."),
    ] = None,
) -> None:
    """Run b against run a: the change in official EX with its interval, McNemar's test,
    and the changes by hardness and by error class."""
    from forge.eval.report import write_compare_report

    try:
        written = write_compare_report(a, b, out=out)
    except _report_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"report: {written}")


@report_app.command("rebuild")
def report_rebuild(
    run_id: Annotated[str, typer.Argument(help="The run, a folder in runs/ (FORGE_RUNS_DIR).")],
) -> None:
    """Render the run's report, and each compare report in its eval folder, again from
    the stored files; exit 1 unless each matches the stored one, its Generated line aside.
    """
    from forge.eval.report import rebuild

    try:
        checks = rebuild(run_id)
    except _report_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for check in checks:
        if check.diff:
            typer.echo(f"{check.path}: differs from its rebuild")
            for line in check.diff:
                typer.echo(line)
        else:
            typer.echo(f"{check.path}: matches its rebuild")
    differ = sum(bool(check.diff) for check in checks)
    if differ:
        typer.echo(f"error: {differ} of {len(checks)} reports differ from their rebuild", err=True)
        raise typer.Exit(1)


def _report_git_sha() -> str:
    """The commit a report came from, marked dirty when the tree has changes."""
    from forge.ops.manifest import GitError, git_sha
    from forge.ops.paths import RepoRootNotFound, repo_root

    try:
        return git_sha(repo_root(), allow_dirty=True)
    except (GitError, RepoRootNotFound):
        # A wheel installed outside a checkout (a Kaggle kernel) has no history to name.
        return "unknown (not run from a git checkout)"


@ledger_app.command("add")
def ledger_add(
    platform: str = typer.Option(..., help="kaggle, colab, modal, local-mps or local-cpu"),
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
