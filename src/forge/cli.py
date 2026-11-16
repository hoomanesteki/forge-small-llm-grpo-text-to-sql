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
QuantizationName = Literal["bitsandbytes"]

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


@data_app.command("build-sft")
def data_build_sft(
    prompt_version: Annotated[
        PromptVersionName, typer.Option(help="The prompt version training uses (ADR-0007).")
    ],
    tokenizer: Annotated[
        str,
        typer.Option(
            help="The model to train: a Hub repo, as <repo>@<revision> to pin it, or a local "
            "directory. Its chat template counts prompt tokens for the overflow policy."
        ),
    ],
    out: Annotated[
        Path | None, typer.Option(help="Where the <hash> folders go (default data/sft).")
    ] = None,
) -> None:
    """Build the SFT dataset from the train split and save it as <out>/<hash>/.

    Rows are TRL prompt-completion pairs: the prompt messages, then the gold query in a
    sql code block. Repeated questions, gold that fails on SQLite and prompts the overflow
    policy can't fit or leaves without the columns the gold reads are left out and counted.
    """
    from forge.data.prompts import chat_token_counter, load_tokenizer, render_prompt_stats
    from forge.data.spider import resolve_data_dir
    from forge.ops.paths import RepoRootNotFound
    from forge.train.sft_data import build_sft_dataset, render_counts, save_sft

    try:
        data_dir = resolve_data_dir()
        count = chat_token_counter(load_tokenizer(tokenizer))
        dataset = build_sft_dataset(prompt_version, tokenizer, count, data_dir=data_dir)
        folder = save_sft(dataset, out)
    # OSError covers missing data, a tokenizer that can't be loaded and a failed write;
    # ValueError covers SplitLeak.
    except (ImportError, OSError, RepoRootNotFound, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    meta = dataset.meta
    typer.echo(f"data: {data_dir}")
    typer.echo(f"prompt {prompt_version}, tokenizer {tokenizer}\n")
    typer.echo(render_counts(meta.counts) + "\n")
    typer.echo(render_prompt_stats(meta.prompts) + "\n")
    typer.echo(f"sft dataset: {folder}\nhash: {meta.hash}")


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
        BackendName,
        typer.Option(
            help="hf runs transformers; vllm runs vLLM on a kernel job's GPU (forge kaggle run); "
            "fake answers without a model."
        ),
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
        float,
        typer.Option(
            min=0.0, help="0 decodes greedily; sampling takes 0.01 or more, as --n above 1 does."
        ),
    ] = 0.0,
    retry: Annotated[
        int,
        typer.Option(
            min=0, max=1, help="1 asks once more, with the error, when a query fails to run."
        ),
    ] = 0,
    quantization: Annotated[
        QuantizationName | None,
        typer.Option(help="Load vllm's model in 4-bit with bitsandbytes, for a model past 4B."),
    ] = None,
    gpu_memory_utilization: Annotated[
        float | None,
        typer.Option(help="The share of the GPU's memory vllm's engine takes; 0.85 if unset."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Name the run, or resume the run of that name."),
    ] = None,
    store: Annotated[
        StoreName,
        typer.Option(
            help="local keeps the run's files in runs/ only; hub also puts them in the Hub run "
            "store (FORGE_RUN_STORE) when the run ends, and resumes a run from it."
        ),
    ] = "local",
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
    # Left out of the manifest too: when an attempt stops changes no score.
    deadline_minutes: Annotated[
        float | None,
        typer.Option(
            "--deadline-minutes",
            min=0.0,
            help="Stop between batches so the run ends within this many minutes from now, a "
            "tenth kept back for official EX and storing; it then says stopped_time_guard "
            "and resumes with its --run-id.",
        ),
    ] = None,
) -> None:
    """Generate, execute and score a split; resume a run by passing its --run-id.

    With --config, a file in configs/eval/ gives the options it sets. Files go to
    runs/<run_id>/ (FORGE_RUNS_DIR): manifest.json, config.yaml and
    eval/<split>-generations.jsonl and -scores.jsonl. With --retry 1,
    eval/<split>-first-tries.jsonl keeps the first reply of each retried sample, and the
    generations file holds the reply each score is for. With --store hub the folder also
    goes to the Hub run store when the run ends, finished or failed, and the store's copy
    of a --run-id is fetched to resume when runs/ has none or an older one. Exits 1 unless
    the run finishes: when it fails, or when --deadline-minutes stops it.
    """
    from forge.eval.config import EvalConfig, NotAvailable
    from forge.eval.run_files import RunConflict, RunCorrupt, RunLocked
    from forge.eval.runner import WORKERS, run_eval
    from forge.ops.dev_usage import DevBudgetExceeded
    from forge.ops.manifest import DirtyTreeError, GitError
    from forge.ops.paths import RepoRootNotFound
    from forge.ops.timeguard import eval_deadline
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
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        run_id=run_id,
        store=store,
    )

    def progress(done: int, total: int) -> None:
        typer.echo(f"scored {done} of {total}", err=True)

    try:
        # From the start, so loading the model counts against it.
        deadline = None if deadline_minutes is None else eval_deadline(deadline_minutes)
        run = run_eval(
            config,
            allow_dirty=allow_dirty,
            workers=WORKERS if workers is None else workers,
            progress=progress,
            deadline=deadline,
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
    if run.stored:
        typer.echo(f"run store {run.stored}: runs/{run.manifest.run_id}")
    if run.resumed:
        typer.echo(f"resumed: {run.resumed} scores were stored by an earlier attempt")
    if len(scores) > run.resumed:
        typer.echo(f"generated {len(scores) - run.resumed} samples in {run.generate_s:.1f} s")
    if run.manifest.status == "stopped_time_guard":
        typer.echo(
            f"error: the deadline stopped run {run.manifest.run_id} with {len(scores)} of "
            f"{run.questions * n} samples scored; official EX waits for the attempt that "
            f"finishes it: run again with --run-id {run.manifest.run_id}",
            err=True,
        )
        raise typer.Exit(1)
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
    """Render the run's report and chart, and each compare report in its eval folder,
    again from the stored files; exit 1 unless each matches the stored one, a report's
    Generated line aside. A chart drawn on another platform, or with other releases of
    what draws it, isn't compared, and the output says so.
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
        elif check.skipped:
            typer.echo(f"{check.path}: not compared, as {check.skipped}")
        else:
            typer.echo(f"{check.path}: matches its rebuild")
    differ = sum(bool(check.diff) for check in checks)
    if differ:
        typer.echo(f"error: {differ} of {len(checks)} files differ from their rebuild", err=True)
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
    on: str = typer.Option("", help="session date in UTC, YYYY-MM-DD (default today in UTC)"),
) -> None:
    """Add a row by hand, for sessions the CLI didn't launch (for example Colab)."""
    from datetime import date

    from forge.ops.clock import utc_today
    from forge.ops.ledger import LedgerRow, append

    day = date.fromisoformat(on) if on else utc_today()
    append(LedgerRow(day, platform, gpu, hours, run_id, purpose, cost))
    typer.echo(f"added {hours:.2f} h on {platform} for {run_id}")


@ledger_app.command("record")
def ledger_record(
    run_id: Annotated[
        str, typer.Argument(help="A run in runs/ (FORGE_RUNS_DIR) that has ended, such as a pilot.")
    ],
) -> None:
    """Add the rows of a run made on this machine, one for each session that has ended, once.

    The date, platform, GPU and hours come from the manifest and the run's sessions, as
    forge runs pull takes them for a pulled run, so a local-mps pilot's hours are never
    typed. A local-cpu run holds no GPU and gets no row.
    """
    from forge.ops.hub import LocalRunStore
    from forge.ops.ledger import record_run
    from forge.ops.paths import runs_dir
    from forge.ops.sessions import sessions_of

    try:
        manifest = LocalRunStore(runs_dir()).get_manifest(run_id)
        _, notes = record_run(manifest, sessions_of(manifest, runs_dir() / run_id))
    except _runs_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for note in notes:
        typer.echo(note)


@ledger_app.command("summary")
def ledger_summary(days: int = typer.Option(7, help="quota window in days")) -> None:
    """Hours this week against the Kaggle quota, total GPU hours and cash spent."""
    from forge.ops.ledger import summary

    typer.echo(summary(window_days=days).render())


kaggle_app = typer.Typer(
    help="Kaggle: GPU jobs, and the private code and data datasets they mount.",
    no_args_is_help=True,
)
app.add_typer(kaggle_app, name="kaggle")


def _kaggle_owner() -> str:
    from forge.ops.dotenv import forge_env
    from forge.ops.kaggle import kaggle_owner

    return kaggle_owner(forge_env(), Path.home())


@kaggle_app.command("publish-code")
def kaggle_publish_code() -> None:
    """Publish the wheel built at HEAD as the private dataset <owner>/forge-code.

    build/kaggle-code gets the wheel, kaggle/requirements-gpu.txt and GIT_SHA, and goes up
    as a new version noted with the SHA, or as the first one. The owner is
    FORGE_KAGGLE_USER, else the Kaggle credentials' username. A dirty tree is refused.
    """
    from forge.ops.kaggle import KaggleClient, KaggleError, publish_code
    from forge.ops.manifest import DirtyTreeError, GitError
    from forge.ops.paths import RepoRootNotFound, repo_root

    try:
        bundle, published = publish_code(repo_root(), _kaggle_owner(), KaggleClient())
    # OSError covers a missing requirements file and a bundle that can't be written,
    # ValueError an unpinned requirement, GitError a root that isn't a checkout.
    except (
        DirtyTreeError,
        GitError,
        KaggleError,
        OSError,
        RepoRootNotFound,
        ValueError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"bundle {bundle.path}: {bundle.wheel}, requirements-gpu.txt, GIT_SHA")
    action = "created" if published.created else "uploaded a new version of"
    typer.echo(f"{action} {published.ref} at {bundle.git_sha}")
    typer.echo(published.output)


@kaggle_app.command("publish-data")
def kaggle_publish_data(
    keep_build: Annotated[
        bool,
        typer.Option(
            "--keep-build", help="Keep build/kaggle-data after the upload, a copy of Spider."
        ),
    ] = False,
) -> None:
    """Upload the verified Spider files once, as the private dataset <owner>/forge-spider.

    Every file spider.sha256 lists must be in FORGE_DATA_DIR (default data/spider) and
    match it. build/kaggle-data gets them and spider.sha256, and the description carries
    Spider's CC BY-SA 4.0 attribution and citation; the folder goes once the upload is
    done. A dataset whose status says ready is left as it is: the checksums pin Spider,
    so one upload serves every job. One whose last version failed or was deleted gets a
    new version, and one Kaggle is still processing is an error until it is ready.
    """
    from forge.data.spider import ChecksumMismatch, resolve_data_dir
    from forge.ops.kaggle import DATA_BUNDLE, DATA_SLUG, KaggleClient, KaggleError, publish_data
    from forge.ops.paths import RepoRootNotFound, repo_root

    try:
        root = repo_root()
        owner = _kaggle_owner()
        data_dir = resolve_data_dir()
        published = publish_data(root, owner, KaggleClient(), data_dir, keep_build=keep_build)
    except ChecksumMismatch as exc:
        typer.echo(
            f"error: {len(exc.files)} file(s) in {data_dir} are missing or differ from "
            "spider.sha256; run `forge data fetch` or point FORGE_DATA_DIR at the full download",
            err=True,
        )
        raise typer.Exit(1) from exc
    # OSError covers a bundle that can't be written, ValueError a malformed checksum list.
    except (KaggleError, OSError, RepoRootNotFound, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if published is None:
        typer.echo(f"{owner}/{DATA_SLUG} is ready already; nothing to upload")
        return
    kept = "kept" if keep_build else "removed after the upload"
    typer.echo(
        f"bundle {root / DATA_BUNDLE} ({kept}): the files of {data_dir}, each matching "
        "spider.sha256"
    )
    action = "created" if published.created else "uploaded a new version of"
    typer.echo(f"{action} {published.ref}")
    typer.echo(published.output)


KaggleJobName = Literal["eval", "sft", "grpo", "spike"]


def _kaggle_job_errors() -> tuple[type[Exception], ...]:
    """What a Kaggle job command can raise: a refused check, a failed kaggle call, a bad
    config (ValueError) or one a later story builds (NotAvailable), a file that can't be
    read (OSError), a root that isn't a checkout (GitError), and a dev look logged past
    the budget (DevBudgetExceeded)."""
    from forge.eval.config import NotAvailable
    from forge.ops.dev_usage import DevBudgetExceeded
    from forge.ops.kaggle import KaggleError
    from forge.ops.kaggle_job import JobRefused
    from forge.ops.manifest import DirtyTreeError, GitError
    from forge.ops.paths import RepoRootNotFound

    return (
        DevBudgetExceeded,
        DirtyTreeError,
        GitError,
        JobRefused,
        KaggleError,
        NotAvailable,
        OSError,
        RepoRootNotFound,
        ValueError,
    )


@kaggle_app.command("run")
def kaggle_run(
    job: Annotated[
        KaggleJobName,
        typer.Option(
            help="eval runs forge eval run; sft, forge train sft; spike, FG-301's throughput "
            "script."
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="configs/eval/*.yaml, configs/sft/*.yaml for sft, or configs/spike/*.yaml "
            "for the spike; its ops.est_gpu_hours is the job's estimate.",
        ),
    ],
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Name the run; by default its date, model, job and commit."),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Carry on the run from its last kernel's output.")
    ] = False,
    preflight: Annotated[
        bool | None,
        typer.Option(
            "--preflight/--no-preflight", help="Run the job on 5 examples first; on by default."
        ),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Push without asking.")] = False,
    emit_notebook: Annotated[
        Path | None,
        typer.Option(
            "--emit-notebook",
            dir_okay=False,
            help="Write the job as a Colab notebook here, and push nothing.",
        ),
    ] = None,
    allow_over_budget: Annotated[
        bool,
        typer.Option(
            "--allow-over-budget",
            help="Push past the week's GPU budget; an eval run's config records it.",
        ),
    ] = False,
    reason: Annotated[
        str | None,
        typer.Option(help="Why dev is evaluated: required for a dev config, and logged."),
    ] = None,
    adapter: Annotated[
        str | None,
        typer.Option(
            help="An eval job's LoRA adapter, for a config that names none: a training run's "
            "checkpoint, runs/<run_id>/checkpoints/<name>, which the kernel fetches from the "
            "run store, or a Hub repo@revision."
        ),
    ] = None,
) -> None:
    """Push a GPU job to Kaggle once it passes the checks every GPU job must pass.

    HEAD must be clean with make smoke passed on it; the code dataset must hold HEAD,
    which is published if it doesn't; and the config's ops.est_gpu_hours must fit what the
    ledger and Kaggle's quota leave this week. It prints the estimate, asks unless --yes,
    pushes the kernel on a T4 and prints the run id, which status, logs, wait and output
    take. A dev job's push spends one of the six dev looks, logged in docs/dev-usage.md as
    soon as Kaggle accepts it; a --resume of the run is the same look. A source Kaggle
    could not add is a warning: that version runs without it, recorded like any other.
    """
    import json

    from forge.ops import kaggle_job as jobs
    from forge.ops.dev_usage import MAX_DEV_USES
    from forge.ops.doctor import DEFAULT_RUN_STORE
    from forge.ops.dotenv import forge_env
    from forge.ops.kaggle import KaggleClient
    from forge.ops.ledger import summary
    from forge.ops.paths import repo_root, runs_dir

    client = KaggleClient()
    try:
        root = repo_root()
        sha = jobs.head_commit(root)
        owner = _kaggle_owner()
        plan = jobs.plan_job(
            job,
            config,
            owner=owner,
            sha=sha,
            runs_root=runs_dir(),
            run_id=run_id,
            resume=resume,
            preflight=preflight,
            allow_over_budget=allow_over_budget,
            reason=reason,
            adapter=adapter,
            notebook=emit_notebook is not None,
        )
        code = data = None
        if plan.mounts_datasets:
            code = jobs.code_version(root, owner, client, sha, say=typer.echo)
            data = jobs.data_source(owner, client)
            typer.echo(f"code {code} at {sha[:7]}, data {data}")
        run_store = forge_env().get("FORGE_RUN_STORE", DEFAULT_RUN_STORE)
        if emit_notebook is not None:
            book = jobs.notebook(root, plan, code=code, data=data, run_store=run_store)
            emit_notebook.write_text(json.dumps(book, indent=1) + "\n", encoding="utf-8")
            typer.echo(f"notebook {emit_notebook}: run {plan.run_id}, nothing pushed")
            typer.echo(
                f"estimate {plan.est_gpu_hours:g} GPU h; log what it takes with "
                "forge ledger add --platform colab"
            )
            return
        budget = jobs.check_budget(plan.est_gpu_hours, summary(), client)
        for line in budget.lines():
            typer.echo(line)
        if budget.over and not allow_over_budget:
            raise jobs.JobRefused(
                "the estimate is over what is left this week; pass --allow-over-budget to "
                "push anyway, and the run records that it did"
            )
        typer.echo(
            f"job {plan.job}: run {plan.run_id} as kernel {plan.kernel}, stopped after "
            f"{plan.timeout_s / 3600:.2f} h at most"
            + (f", carrying on {plan.kernel_sources[0]}" if plan.kernel_sources else "")
        )
        if not yes and not typer.confirm("Push it?"):
            typer.echo("nothing pushed")
            raise typer.Exit(1)
        folder = jobs.write_kernel(root, plan, code=code, data=data, run_store=run_store)
        pushed = client.push_kernel(folder, timeout_s=plan.timeout_s, gpu=True)
    except _kaggle_job_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    kernel, version = pushed.kernel or plan.kernel, jobs.version_name(pushed.version)
    # Kaggle runs the kernel now, and it may read dev.
    looks, problems = jobs.settle_push(
        runs_dir(), plan, pushed, kernel=kernel, code=code, budget=budget
    )
    if problems:
        typer.echo(
            f"error: {'; '.join(problems)}; Kaggle runs {kernel} {version} all the same, so "
            f"don't push it again: follow it at https://www.kaggle.com/code/{kernel}",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(pushed.output)
    for line in pushed.dropped:
        typer.echo(f"warning: {line}", err=True)
    if pushed.dropped:
        typer.echo(
            f"warning: Kaggle started {version} without them; forge kaggle wait "
            f"{plan.run_id} says how the job ends",
            err=True,
        )
    if plan.resume and pushed.kernel_source_dropped:
        typer.echo(
            "warning: this attempt can't restore the run, so a --resume mounts "
            f"{plan.kernel_sources[0]} again",
            err=True,
        )
    if looks is not None:
        typer.echo(f"dev look {looks} of {MAX_DEV_USES} logged in docs/dev-usage.md")
    elif plan.reads_dev:
        typer.echo(f"dev look: run {plan.run_id}'s own, logged when it was first pushed")
    typer.echo(f"run {plan.run_id}: kernel {kernel}, {version}")
    typer.echo(f"next: forge kaggle wait {plan.run_id}")


def _run_kernel(run_id: str) -> tuple[str, str]:
    """The kernel of the run's latest push and its job, from runs/<run_id>/kernels.json."""
    from forge.ops.kaggle_job import last_kernel
    from forge.ops.paths import runs_dir

    record = last_kernel(runs_dir(), run_id)
    return record.kernel, record.job


RunIdArgument = Annotated[str, typer.Argument(help="The run id forge kaggle run printed.")]


@kaggle_app.command("status")
def kaggle_status(run_id: RunIdArgument) -> None:
    """Print the state of the run's kernel: QUEUED, RUNNING, COMPLETE, ERROR or CANCEL_*."""
    from forge.ops.kaggle import KaggleClient

    try:
        kernel, _ = _run_kernel(run_id)
        status = KaggleClient().kernel_status(kernel)
    except _kaggle_job_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{kernel}: {status.state}")
    if status.failure:
        typer.echo(f"failure: {status.failure}")


@kaggle_app.command("logs")
def kaggle_logs(run_id: RunIdArgument) -> None:
    """Print the log of the run's kernel."""
    from forge.ops.kaggle import KaggleClient

    try:
        kernel, _ = _run_kernel(run_id)
        typer.echo(KaggleClient().kernel_logs(kernel))
    except _kaggle_job_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@kaggle_app.command("wait")
def kaggle_wait(
    run_id: RunIdArgument,
    poll: Annotated[int, typer.Option(min=5, help="Seconds between checks.")] = 60,
) -> None:
    """Wait until the run's kernel ends: COMPLETE, ERROR or CANCEL_*.

    Once the job has ended, its session gets its row in docs/compute-ledger.md, as forge
    runs pull adds it, from the status.json, manifest and sessions in the kernel's output.
    Exits 0 only when the job itself finished, as status.json says: the entry script ends
    a failed job without an error, so its output is kept.
    """
    import tempfile

    from forge.ops import kaggle_job as jobs
    from forge.ops.kaggle import KaggleClient

    client = KaggleClient()
    with tempfile.TemporaryDirectory(prefix="forge-status-") as folder:
        try:
            kernel, job = _run_kernel(run_id)
            state = jobs.wait_for(client, kernel, poll_s=poll, say=typer.echo)
            if state != "COMPLETE":
                raise jobs.JobRefused(
                    f"the kernel ended {state}; read forge kaggle logs {run_id}, and fetch "
                    f"what it kept with forge kaggle output {run_id}, which logs its hours"
                )
            if job == "spike":
                typer.echo(f"the spike ran; fetch its report with forge kaggle output {run_id}")
                return
            record = jobs.job_record(client, kernel, run_id, Path(folder))
            outcome = jobs.read_status(record)
        except _kaggle_job_errors() as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc
        stages = outcome.get("stages", {})
        upload = stages.get("upload", {}).get("result", "not reached")
        typer.echo(f"job {outcome['status']} at stage {outcome['stage']}; upload: {upload}")
        _log_run(record, "has ended")
    if outcome["status"] == "stopped_time_guard":
        typer.echo(
            "error: the job stopped at its time guard before Kaggle's timeout; carry it on "
            f"with forge kaggle run --resume --run-id {run_id} and the same --job and --config",
            err=True,
        )
        raise typer.Exit(1)
    if outcome["status"] != "finished":
        typer.echo(f"error: {outcome.get('error') or outcome['status']}", err=True)
        raise typer.Exit(1)


@kaggle_app.command("output")
def kaggle_output(run_id: RunIdArgument) -> None:
    """Download the run's kernel output into runs/<run_id>/ (FORGE_RUNS_DIR), and add the
    ledger rows of its sessions.

    The run's files go in that folder and the rest, the kernel's log and the preflight
    among it, in its kernel/ folder. It is how results come back when the kernel had no
    HF_TOKEN to upload them with. Each of the run's sessions that has ended gets its row in
    docs/compute-ledger.md once, as forge runs pull adds them.
    """
    from forge.ops.kaggle import KaggleClient
    from forge.ops.kaggle_job import fetch_output
    from forge.ops.paths import runs_dir

    try:
        kernel, _ = _run_kernel(run_id)
        run_dir = fetch_output(KaggleClient(), kernel, run_id, runs_dir())
    except _kaggle_job_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"run {run_id}: {run_dir}; the kernel's log and the rest in {run_dir / 'kernel'}")
    _log_run(run_dir, "is fetched")


def _log_run(run_dir: Path, done: str) -> None:
    """Say what the run in ``run_dir`` is, and add the ledger rows of its sessions that have
    ended, once each. A folder with no manifest, a spike's output or a kernel's that failed
    before its run began, gets none. ``done`` says where the run is, for an error."""
    from forge.eval.run_files import MANIFEST_FILE
    from forge.ops.hub import LocalRunStore
    from forge.ops.kaggle_job import no_run_note
    from forge.ops.ledger import record_run
    from forge.ops.sessions import sessions_of

    if not (run_dir / MANIFEST_FILE).is_file():
        typer.echo(no_run_note(run_dir))
        return
    try:
        manifest = LocalRunStore(run_dir.parent).get_manifest(run_dir.name)
        typer.echo(f"{manifest.kind} run, {manifest.status}")
        _, notes = record_run(manifest, sessions_of(manifest, run_dir))
    except _runs_errors() as exc:
        typer.echo(
            f"error: run {run_dir.name} {done}, but its ledger rows weren't added: {exc}",
            err=True,
        )
        raise typer.Exit(1) from exc
    for note in notes:
        typer.echo(note)


runs_app = typer.Typer(
    help="The run store: runs' files off the machine that made them, mirrored into runs/.",
    no_args_is_help=True,
)
app.add_typer(runs_app, name="runs")
RunStoreName = Literal["local", "hub"]
PullSource = Literal["hub", "kaggle"]
StoreOption = Annotated[
    RunStoreName,
    typer.Option(
        "--store",
        help="hub, the Hub run store FORGE_RUN_STORE names, or local, runs/ (FORGE_RUNS_DIR).",
    ),
]
RunArgument = Annotated[str, typer.Argument(help="The run's id, such as forge runs list prints.")]


def _runs_errors() -> tuple[type[Exception], ...]:
    """What a run store command can raise: OSError for a missing run or file and for the
    Hub's errors (StoreUnavailable once its retries are spent), ValueError for a bad run
    id or a corrupt manifest."""
    from forge.ops.paths import RepoRootNotFound

    return (OSError, RepoRootNotFound, ValueError)


@runs_app.command("list")
def runs_list(store: StoreOption = "hub") -> None:
    """Print the id of each run the store holds, oldest first: ids start with their date."""
    from forge.ops.hub import open_store

    try:
        where = open_store(store)
        ids = where.runs()
    except _runs_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for run_id in ids:
        typer.echo(run_id)
    typer.echo(f"{len(ids)} run{'' if len(ids) == 1 else 's'} in {where}", err=True)


@runs_app.command("show")
def runs_show(run_id: RunArgument, store: StoreOption = "hub") -> None:
    """Print the run's manifest, what made it and how it ended, and its latest checkpoint."""
    from forge.ops.hub import describe, open_store

    try:
        where = open_store(store)
        lines = describe(where.get_manifest(run_id), where.latest_checkpoint(run_id))
    except _runs_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for line in lines:
        typer.echo(line)


@runs_app.command("pull")
def runs_pull(
    run_id: RunArgument,
    store: Annotated[
        PullSource,
        typer.Option(
            "--store",
            help="hub, the Hub run store FORGE_RUN_STORE names, or kaggle, the output of the "
            "run's last kernel, as forge kaggle output fetches it.",
        ),
    ] = "hub",
) -> None:
    """Mirror the run into runs/<run_id>/ (FORGE_RUNS_DIR), and add its ledger rows.

    The folder then holds the store's copy and no file it lacks, such as an earlier
    attempt's error.txt, but for kernels.json and kernel/, which stay unless the copy's
    kernels.json lists more pushes than this machine's.
    A run that held a GPU gets a row in docs/compute-ledger.md for each of its sessions
    that has ended, once.
    --store kaggle is forge kaggle output.
    """
    from forge.ops.hub import LocalRunStore, hub_store
    from forge.ops.paths import runs_dir

    if store == "kaggle":
        kaggle_output(run_id)
        return
    try:
        local, hub = LocalRunStore(runs_dir()), hub_store()
        hub.pull(run_id, local.root / run_id)
    except _runs_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"run {run_id}: {local.root / run_id}, from {hub}")
    _log_run(local.root / run_id, "is pulled")


@runs_app.command("push")
def runs_push(run_id: RunArgument) -> None:
    """Make the Hub run store's copy of the run what runs/<run_id>/ holds.

    For a run whose upload failed, or one that came back through forge kaggle output. Its
    manifest must be there. Files the store has and the folder lacks are deleted, so push
    a run only from a folder that holds all of it, such as one forge runs pull filled. A
    store whose copy holds a session the folder lacks keeps it, and the push is refused.
    """
    from forge.ops.hub import LocalRunStore, hub_store
    from forge.ops.paths import runs_dir
    from forge.ops.run_copies import put_run

    try:
        hub, local = hub_store(), LocalRunStore(runs_dir())
        manifest = local.get_manifest(run_id)
        path = put_run(hub, local.root / run_id, replace=True)
    except _runs_errors() as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"run {run_id} ({manifest.kind}, {manifest.status}): uploaded to {hub} as {path}")


@runs_app.command("promote")
def runs_promote(
    run_id: RunArgument,
    evals: Annotated[
        list[str],
        typer.Option(
            "--eval",
            help="An eval run in runs/ of one of the run's checkpoints on val; once for each.",
        ),
    ],
    store: StoreOption = "hub",
) -> None:
    """Copy the SFT run's checkpoint that val scores highest to checkpoints/best.

    Each --eval is a finished val run in runs/ (FORGE_RUNS_DIR) whose adapter is one of the
    run's checkpoints, runs/<run_id>/checkpoints/step-NNNNNN or .../final, with the others'
    settings. The one with the highest official EX, the later one on a tie, is copied to
    runs/<run_id>/checkpoints/best/, and best.json beside the manifest records the scores and
    the eval runs they come from. With --store hub, the default, the run store gets them
    too, so pull the run first: its checkpoints must be here. Dev is never used.
    """
    from forge.eval.run_files import RunUnfinished
    from forge.ops.hub import hub_store
    from forge.ops.paths import runs_dir
    from forge.train.sft import checkpoint_scores, promote_best

    # An eval run that hasn't finished is a RuntimeError, which a store's errors are not.
    errors: tuple[type[Exception], ...] = (*_runs_errors(), RunUnfinished)
    try:
        scores, sources = checkpoint_scores(run_id, evals, runs_root=runs_dir())
        where = hub_store() if store == "hub" else None
        promoted = promote_best(run_id, scores, sources=sources, runs_root=runs_dir(), store=where)
    except errors as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for rel, score in promoted.record["scores"].items():
        typer.echo(f"{rel}: official EX {score:.2%} on val, in run {sources[rel]}")
    copies = f"{promoted.best}" + (f" and {where}" if where is not None else "")
    typer.echo(f"best: {promoted.checkpoint}, now checkpoints/best in {copies}")


train_app = typer.Typer(help="Train LoRA adapters on the train split.", no_args_is_help=True)
app.add_typer(train_app, name="train")


@train_app.command("sft")
def train_sft(
    config_file: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            help="The run's training config, such as configs/sft/tiny-smoke.yaml.",
        ),
    ],
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Name the run; by default <date>-<model>-sft-<sha7>."),
    ] = None,
    sft_dir: Annotated[
        Path | None,
        typer.Option(help="Where the SFT dataset's <hash> folders are kept (default data/sft)."),
    ] = None,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty", help="Run on uncommitted changes; the manifest's SHA says -dirty."
        ),
    ] = False,
) -> None:
    """Fine-tune a LoRA adapter on the train split's gold queries, as the config says.

    It trains on the rows forge data build-sft builds for the config's model and prompt
    version, kept in <sft-dir>/<hash>/. Files go to runs/<run_id>/ (FORGE_RUNS_DIR):
    manifest.json, config.yaml, metrics.jsonl, checkpoints/step-NNNNNN/ every
    ops.checkpoint_every steps, and the adapter in checkpoints/final/, which forge eval run
    --adapter loads. Exits 1 if the run can't start.
    """
    from forge.ops.hub import checkpoint_dir
    from forge.ops.manifest import DirtyTreeError, GitError
    from forge.ops.paths import RepoRootNotFound
    from forge.train.backends import BackendUnavailable
    from forge.train.config import load_train_config
    from forge.train.sft import run_sft

    try:
        config = load_train_config(config_file)
        run = run_sft(config, run_id=run_id, allow_dirty=allow_dirty, sft_root=sft_dir)
    # OSError covers missing data, a model that can't be found and a run id already used;
    # ValueError a config no SFT run takes, a device with no platform, and SplitLeak.
    except (
        BackendUnavailable,
        DirtyTreeError,
        GitError,
        ImportError,
        OSError,
        RepoRootNotFound,
        ValueError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    manifest = run.manifest
    trainer = manifest.config["trainer"]
    sft_data = manifest.config["sft_data"]
    typer.echo(f"run {manifest.run_id}: {run.root}")
    typer.echo(f"sft data {run.sft_data}: {sft_data['rows']} rows, hash {sft_data['hash']}")
    typer.echo(
        f"trained on {sft_data['trained']} rows; {sft_data['over_max_length']} were over "
        f"max_length {manifest.config['sft']['max_length']} and left out"
    )
    typer.echo(
        f"platform {manifest.platform}, backend {trainer['backend']} on {trainer['device']} "
        f"in {trainer['dtype']}"
    )
    if run.stopped_at is not None:
        typer.echo(
            f"stopped by the time guard after step {run.stopped_at}, so there is no final "
            f"adapter; its checkpoint is {run.root / checkpoint_dir(run.stopped_at)}"
        )
    else:
        typer.echo(f"adapter {run.adapter}")


if __name__ == "__main__":
    app()
