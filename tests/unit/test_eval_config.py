"""The eval runner's options: what is refused up front, and what the manifest records."""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pytest
import typer
from typer.testing import CliRunner

from forge import cli
from forge.core.types import GenParams
from forge.data.schema import PROMPT_VERSIONS
from forge.data.splits import SPLITS
from forge.eval import runner
from forge.eval.config import (
    BACKENDS,
    DEVICES,
    FILE_OPTIONS,
    STORES,
    EvalConfig,
    NotAvailable,
    Ops,
    check_config,
    read_config_file,
    read_config_sections,
)

BASE = EvalConfig(model="org/model")


@pytest.mark.phase(2)
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"split": "dev"}, ValueError, "--split dev needs --reason"),
        ({"split": "dev", "reason": " \t"}, ValueError, "--split dev needs --reason"),
        ({"reason": "a look"}, ValueError, "--reason goes with --split dev"),
        ({"split": "test"}, ValueError, "split must be one of train, val, dev"),
        ({"backend": "onnx"}, ValueError, "backend must be one of hf, vllm, llamacpp, fake"),
        ({"device": "cuda"}, ValueError, "device must be one of cpu, mps"),
        (
            {"device": "mps", "split": "dev", "reason": "a look"},
            ValueError,
            "pilots never evaluate dev",
        ),
        ({"store": "s3"}, ValueError, "store must be one of local, hub"),
        ({"prompt_version": "p3"}, ValueError, "unknown prompt version 'p3'"),
        ({"limit": 0}, ValueError, "limit must be at least 1"),
        ({"retry": 2}, ValueError, "retry must be 0 or 1"),
        ({"run_id": "Run-1"}, ValueError, "names a folder"),
        ({"run_id": "../up"}, ValueError, "names a folder"),
        ({"run_id": "-x"}, ValueError, "names a folder"),
        ({"backend": "fake"}, ValueError, "--backend fake takes --model gold or select-1"),
        ({"backend": "fake", "model": "gold", "adapter": "a"}, ValueError, "loads no adapter"),
        (
            {"backend": "fake", "model": "gold", "device": "mps"},
            ValueError,
            "--backend fake runs no model, so it takes no --device mps",
        ),
        ({"n": 4}, ValueError, "n=4 needs a temperature above 0"),
        ({"backend": "vllm"}, NotAvailable, "the vllm backend is not available yet"),
        ({"backend": "llamacpp"}, NotAvailable, "comes with FG-702"),
        ({"store": "hub"}, NotAvailable, "--store hub is available after FG-304"),
    ],
)
def test_config_refused_before_anything_runs(
    changes: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        check_config(dataclasses.replace(BASE, **changes))


@pytest.mark.phase(2)
def test_config_accepted() -> None:
    for config in (
        BASE,
        EvalConfig(model="gold", backend="fake", split="train", limit=3, run_id="smoke-abc1234"),
        EvalConfig(model="org/model", adapter="runs/x/best", split="dev", reason="FG-508"),
        EvalConfig(model="org/model", n=8, temperature=0.7, prompt_version="p2"),
        EvalConfig(model="org/model", retry=1),
        EvalConfig(model="org/model", n=4, temperature=0.7, retry=1),
        EvalConfig(model="org/model", device="mps"),
    ):
        check_config(config)


@pytest.mark.phase(2)
def test_manifest_config_records_what_changes_a_score() -> None:
    config = BASE.manifest_config()
    assert config["generation"] == GenParams().model_dump()
    assert config["execution"] == {
        "timeout_ms": 2000,
        "gold_timeout_ms": 10_000,
        "max_rows": 100_000,
        "official_item_timeout_s": 10.0,
    }
    assert config["device"] == "cpu"
    for changes in (
        {"n": 2, "temperature": 0.5},
        {"limit": 5},
        {"prompt_version": "p2"},
        {"device": "mps"},
    ):
        assert dataclasses.replace(BASE, **changes).manifest_config() != config
    # Where the files go and why dev was read don't change a score.
    assert dataclasses.replace(BASE, store="hub").manifest_config() == config
    dev = dataclasses.replace(BASE, split="dev", reason="a look").manifest_config()
    assert "reason" not in dev and dev == {**config, "split": "dev"}


@pytest.mark.phase(3)
def test_platform_is_the_host_of_a_kernel_job() -> None:
    # kaggle/entry.py names the machine it runs on; the manifest's config doesn't change.
    on_kaggle = dataclasses.replace(BASE, host="kaggle")
    assert on_kaggle.platform == "kaggle"
    assert dataclasses.replace(BASE, host="colab").platform == "colab"
    assert on_kaggle.manifest_config() == BASE.manifest_config()
    check_config(dataclasses.replace(on_kaggle, backend="fake", model="gold"))
    with pytest.raises(NotAvailable, match="comes with FG-306"):
        check_config(dataclasses.replace(on_kaggle, backend="vllm"))
    with pytest.raises(ValueError, match="host must be one of kaggle, colab, got 'modal'"):
        check_config(dataclasses.replace(BASE, host="modal"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Apple silicon's GPU, which kaggle has not"):
        check_config(dataclasses.replace(on_kaggle, backend="vllm", device="mps"))
    # The hf backend runs on the CPU or MPS, which would leave a T4 idle.
    with pytest.raises(ValueError, match="backend hf runs on the CPU or MPS; a job on kaggle"):
        check_config(on_kaggle)


@pytest.mark.phase(3)
def test_a_kernel_job_pushed_over_budget_says_so_in_its_config() -> None:
    kernel = EvalConfig(model="gold", backend="fake", host="kaggle")
    over = dataclasses.replace(kernel, allow_over_budget=True)
    check_config(over)
    assert over.manifest_config() == {**kernel.manifest_config(), "allow_over_budget": True}
    # A laptop run spends no GPU budget, so it can't be over it.
    with pytest.raises(ValueError, match="allow_over_budget is for a kernel job"):
        check_config(dataclasses.replace(over, host=None))


@pytest.mark.phase(2)
def test_platform_follows_the_device() -> None:
    # A run on Apple silicon's GPU is a local-mps pilot; every other local run is local-cpu.
    assert BASE.platform == "local-cpu"
    assert dataclasses.replace(BASE, device="mps").platform == "local-mps"
    assert EvalConfig(model="gold", backend="fake").platform == "local-cpu"


@pytest.mark.phase(2)
def test_cli_offers_exactly_the_backends_stores_splits_and_prompts() -> None:
    hints = get_type_hints(cli.eval_run, include_extras=False)
    assert get_args(hints["backend"]) == BACKENDS
    assert get_args(hints["device"]) == DEVICES
    assert get_args(hints["store"]) == STORES
    assert get_args(hints["split"]) == SPLITS
    assert get_args(hints["prompt_version"]) == PROMPT_VERSIONS


@pytest.mark.phase(2)
def test_eval_run_help_names_its_files_and_how_limit_picks() -> None:
    # The command typer builds, whose help is what --help prints before wrapping it.
    run = typer.main.get_command(cli.app).commands["eval"].commands["run"]
    text = " ".join(run.help.split())
    assert "eval/<split>-first-tries.jsonl keeps the first reply of each retried sample" in text
    assert "the generations file holds the reply each score is for" in text
    (limit,) = [param for param in run.params if param.name == "limit"]
    assert limit.help.startswith("Answer only the first N questions: by id on val and dev")
    assert "in file order on train (train_spider before the rowless train_others)" in limit.help


@pytest.mark.phase(2)
def test_cli_sizes_the_sql_worker_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    pools: list[int] = []

    def run_eval(config: EvalConfig, *, workers: int, **kwargs: Any) -> None:
        pools.append(workers)
        raise ValueError("stopped before any work")

    monkeypatch.setattr(runner, "run_eval", run_eval)
    cli_runner = CliRunner()
    args = ["eval", "run", "--backend", "fake", "--model", "gold"]
    for options in (["--workers", "2"], []):
        result = cli_runner.invoke(cli.app, [*args, *options])
        assert (result.exit_code, result.stderr) == (1, "error: stopped before any work\n")
    # Unset, the pool has the runner's default: one worker per CPU, at most 4.
    assert pools == [2, runner.WORKERS]
    refused = cli_runner.invoke(cli.app, [*args, "--workers", "0"])
    assert refused.exit_code == 2 and "--workers" in refused.stderr
    assert len(pools) == 2


def _captured(monkeypatch: pytest.MonkeyPatch) -> list[EvalConfig]:
    """The configs ``forge eval run`` hands the runner, which stops before any work."""
    configs: list[EvalConfig] = []

    def run_eval(config: EvalConfig, **kwargs: Any) -> None:
        configs.append(config)
        raise ValueError("stopped before any work")

    monkeypatch.setattr(runner, "run_eval", run_eval)
    return configs


@pytest.mark.phase(2)
def test_config_file_gives_defaults_the_command_line_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = _captured(monkeypatch)
    path = tmp_path / "eval.yaml"
    path.write_text(
        "model: gold\nbackend: fake\nsplit: train\nlimit: 3\nprompt_version: p2\n"
        "n: 2\ntemperature: 0.7\nretry: 1\nadapter: null\n",
        encoding="utf-8",
    )
    cli_runner = CliRunner()
    for args in ([], ["--limit", "5", "--model", "select-1"]):
        result = cli_runner.invoke(cli.app, ["eval", "run", "--config", str(path), *args])
        assert (result.exit_code, result.stderr) == (1, "error: stopped before any work\n")
    from_file = EvalConfig(
        model="gold",
        backend="fake",
        split="train",
        limit=3,
        prompt_version="p2",
        n=2,
        temperature=0.7,
        retry=1,
    )
    assert configs == [from_file, dataclasses.replace(from_file, limit=5, model="select-1")]

    # Without a file nothing changes, and the model is still required.
    missing = cli_runner.invoke(cli.app, ["eval", "run", "--backend", "fake"])
    assert missing.exit_code == 2 and "Missing option '--model'" in missing.stderr


@pytest.mark.phase(2)
@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("model: gold\nrun_id: x\n", "sets run_id, which a config file can't"),
        ("model: gold\nreason: a look\nstore: hub\n", "sets reason, store, which"),
        ("model: gold\nn: '2'\n", "n must be a whole number, got '2'"),
        ("model: gold\nretry: true\n", "retry must be a whole number, got True"),
        ("model: gold\ntemperature: warm\n", "temperature must be a number"),
        ("model: null\n", "model must be text, got None"),
        ("- model\n- gold\n", "must be a mapping of options"),
        ("", "must be a mapping of options"),
        ("model: [gold\n", "is not valid YAML"),
    ],
)
def test_config_file_refused_unless_it_holds_options(
    tmp_path: Path, text: str, message: str
) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        read_config_file(path)


@pytest.mark.phase(2)
def test_cli_refuses_a_bad_config_file_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = _captured(monkeypatch)
    path = tmp_path / "eval.yaml"
    cli_runner = CliRunner()
    for text, message in (
        ("model: gold\nrun_id: x\n", "sets run_id, which a config file can't"),
        # A value of the right kind is still checked by its option's type.
        ("model: gold\nbackend: onnx\n", "Invalid value for '--backend'"),
    ):
        path.write_text(text, encoding="utf-8")
        result = cli_runner.invoke(cli.app, ["eval", "run", "--config", str(path)])
        assert result.exit_code == 2 and message in _unboxed(result.stderr), result.stderr
    gone = cli_runner.invoke(cli.app, ["eval", "run", "--config", str(tmp_path / "none.yaml")])
    assert gone.exit_code == 2 and "does not exist" in _unboxed(gone.stderr)
    assert configs == []


@pytest.mark.phase(3)
def test_ops_section_is_read_apart_from_the_options(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("model: gold\nbackend: fake\nops:\n  est_gpu_hours: 0.25\n", encoding="utf-8")
    options, ops = read_config_sections(path)
    assert (options, ops) == ({"model": "gold", "backend": "fake"}, Ops(est_gpu_hours=0.25))
    # forge eval run's reader checks the section and leaves it out.
    assert read_config_file(path) == options
    # A whole number of hours is a number too, and a file without the section has no ops.
    path.write_text("model: gold\nops: {est_gpu_hours: 2}\n", encoding="utf-8")
    assert read_config_sections(path)[1] == Ops(est_gpu_hours=2.0)
    path.write_text("model: gold\n", encoding="utf-8")
    assert read_config_sections(path)[1] == Ops()


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("ops", "message"),
    [
        ("ops: null", "ops must be a mapping"),
        ("ops: [est_gpu_hours]", "ops must be a mapping"),
        ("ops: {est_gpu_hours: 1, checkpoint_every: 50}", "sets ops.checkpoint_every, which"),
        ("ops: {est_gpu_hours: 0}", "ops.est_gpu_hours must be a positive number, got 0"),
        ("ops: {est_gpu_hours: -1.5}", "must be a positive number, got -1.5"),
        ("ops: {est_gpu_hours: .inf}", "must be a positive number, got inf"),
        ("ops: {est_gpu_hours: .nan}", "must be a positive number, got nan"),
        ("ops: {est_gpu_hours: true}", "must be a positive number, got True"),
        ("ops: {est_gpu_hours: '1'}", "must be a positive number, got '1'"),
        ("ops: {est_gpu_hours: null}", "must be a positive number, got None"),
    ],
)
def test_ops_section_refused_unless_it_holds_known_ops(
    tmp_path: Path, ops: str, message: str
) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(f"model: gold\n{ops}\n", encoding="utf-8")
    for read in (read_config_sections, read_config_file):
        with pytest.raises(ValueError, match=re.escape(message)):
            read(path)


@pytest.mark.phase(3)
def test_eval_run_takes_a_config_with_ops_and_ignores_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = _captured(monkeypatch)
    path = tmp_path / "eval.yaml"
    path.write_text(
        "model: gold\nbackend: fake\nlimit: 3\nops:\n  est_gpu_hours: 0.5\n", encoding="utf-8"
    )
    result = CliRunner().invoke(cli.app, ["eval", "run", "--config", str(path)])
    assert (result.exit_code, result.stderr) == (1, "error: stopped before any work\n")
    assert configs == [EvalConfig(model="gold", backend="fake", limit=3)]


def _unboxed(text: str) -> str:
    """A usage error as one line: click's message comes in a box, wrapped to its width."""
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", text).split())


@pytest.mark.phase(2)
def test_config_file_options_are_eval_run_options() -> None:
    # The file's keys feed click's default map, which knows options by parameter name.
    parameters = inspect.signature(cli.eval_run).parameters
    fields = {field.name for field in dataclasses.fields(EvalConfig)}
    assert set(FILE_OPTIONS) <= set(parameters) & fields
