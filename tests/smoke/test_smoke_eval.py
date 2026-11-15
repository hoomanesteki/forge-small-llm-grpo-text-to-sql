"""The eval pipeline end to end on the tiny random-weight model, through the commands that
``make smoke`` runs (FG-207).

The model writes noise, so its accuracy is about 0%. What this checks is every join
between the stages: the fixture's val questions in, the model's replies to them, their
queries run in the sandbox and asked again with the error when they fail, both checkers'
verdicts, and a report whose every number traces to the stored files (I4) and that
rebuilds to the same text.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import GenParams
from forge.data.splits import load_val
from tests.report_trace import read_stored, untraced

SECTIONS = ("Execution accuracy", "Retry", "By hardness", "Error classes", "Example failures")


@pytest.mark.phase(2)
def test_eval_pipeline_on_tiny_model(
    spider_mini: Path, tiny_model_ref: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("FORGE_RUNS_DIR", str(runs))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    cli_runner = CliRunner()
    evaluate = [
        *("eval", "run", "--backend", "hf", "--model", tiny_model_ref, "--split", "val"),
        *("--limit", "20", "--retry", "1", "--run-id", "smoke", "--workers", "2"),
        "--allow-dirty",
    ]

    evaluated = cli_runner.invoke(cli.app, evaluate)
    assert evaluated.exit_code == 0, evaluated.output
    reported = cli_runner.invoke(cli.app, ["report", "run", "smoke"])
    assert reported.exit_code == 0, reported.output

    run_dir = runs / "smoke"
    stored = read_stored(run_dir, spider_mini)
    manifest, scores = stored.manifest, stored.scores
    assert (manifest.status, manifest.model_ref) == ("finished", tiny_model_ref)
    assert manifest.config["backend"] == "hf"
    # Each val question was answered by the model, and both checkers judged the reply.
    assert [score.example_id for score in scores] == sorted(e.id for e in load_val())
    assert all(score.ex_official is not None for score in scores)
    longest = GenParams().max_new_tokens
    for reply in [*stored.generations.values(), *stored.first_tries.values()]:
        assert reply.model_ref == tiny_model_ref and reply.text
        assert reply.tokens_in > 0 and 1 <= reply.tokens_out <= longest
    # Random weights write no query that runs, so retries happen: once for each sample
    # whose first query failed, and never for one that ran.
    retried = {(s.example_id, s.sample_idx) for s in scores if s.attempts == 2}
    assert retried and set(stored.first_tries) == retried
    assert all(score.valid for score in scores if score.attempts == 1)

    report = (run_dir / "eval" / "val-report.md").read_text(encoding="utf-8")
    for section in SECTIONS:
        assert f"\n## {section}\n" in report, section
    assert (run_dir / "eval" / "val-hardness.png").read_bytes().startswith(b"\x89PNG")
    assert untraced(report, run_dir, data_dir=spider_mini) == []
    rebuilt = cli_runner.invoke(cli.app, ["report", "rebuild", "smoke"])
    assert rebuilt.exit_code == 0, rebuilt.output
    assert rebuilt.stdout == f"{run_dir / 'eval' / 'val-report.md'}: matches its rebuild\n"
