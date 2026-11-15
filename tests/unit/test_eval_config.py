"""The eval runner's options: what is refused up front, and what the manifest records."""

from __future__ import annotations

import dataclasses
import re
from typing import Any, get_args, get_type_hints

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import GenParams
from forge.data.schema import PROMPT_VERSIONS
from forge.data.splits import SPLITS
from forge.eval import runner
from forge.eval.config import BACKENDS, STORES, EvalConfig, NotAvailable, check_config

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
        ({"store": "s3"}, ValueError, "store must be one of local, hub"),
        ({"prompt_version": "p3"}, ValueError, "unknown prompt version 'p3'"),
        ({"limit": 0}, ValueError, "limit must be at least 1"),
        ({"retry": 2}, ValueError, "retry must be 0 or 1"),
        ({"run_id": "Run-1"}, ValueError, "names a folder"),
        ({"run_id": "../up"}, ValueError, "names a folder"),
        ({"run_id": "-x"}, ValueError, "names a folder"),
        ({"backend": "fake"}, ValueError, "--backend fake takes --model gold or select-1"),
        ({"backend": "fake", "model": "gold", "adapter": "a"}, ValueError, "loads no adapter"),
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
    for changes in ({"n": 2, "temperature": 0.5}, {"limit": 5}, {"prompt_version": "p2"}):
        assert dataclasses.replace(BASE, **changes).manifest_config() != config
    # Where the files go and why dev was read don't change a score.
    assert dataclasses.replace(BASE, store="hub").manifest_config() == config
    dev = dataclasses.replace(BASE, split="dev", reason="a look").manifest_config()
    assert "reason" not in dev and dev == {**config, "split": "dev"}


@pytest.mark.phase(2)
def test_cli_offers_exactly_the_backends_stores_splits_and_prompts() -> None:
    hints = get_type_hints(cli.eval_run, include_extras=False)
    assert get_args(hints["backend"]) == BACKENDS
    assert get_args(hints["store"]) == STORES
    assert get_args(hints["split"]) == SPLITS
    assert get_args(hints["prompt_version"]) == PROMPT_VERSIONS


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
