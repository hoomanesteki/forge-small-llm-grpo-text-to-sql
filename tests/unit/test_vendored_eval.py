from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VENDORED = ROOT / "third_party" / "test_suite_eval"
UPSTREAM_COMMIT = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"
MODULES = ["evaluation.py", "exec_eval.py", "parse.py", "process_sql.py"]


@pytest.mark.phase(1)
def test_vendored_modules_compile_without_warnings() -> None:
    # Upstream's non-raw regex literals raise SyntaxWarning on 3.12, an error under -W error.
    for name in MODULES:
        path = VENDORED / name
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.phase(1)
def test_changed_files_carry_a_change_notice() -> None:
    # Apache-2.0 section 4b: modified files must say so; CHANGES.md says how.
    assert "Apache License" in (VENDORED / "LICENSE").read_text(encoding="utf-8")
    changes = (VENDORED / "CHANGES.md").read_text(encoding="utf-8")
    assert UPSTREAM_COMMIT in changes
    for name in MODULES:
        head = (VENDORED / name).read_text(encoding="utf-8").splitlines()[:5]
        notice = "\n".join(head)
        assert UPSTREAM_COMMIT in notice, f"{name} lacks the source notice"
        assert "CHANGES.md" in notice, f"{name} does not point to CHANGES.md"


@pytest.mark.phase(1)
def test_evaluator_connection_matches_the_executor() -> None:
    # The vendored package keeps its own copy so it imports without Forge; they must agree.
    from test_suite_eval import exec_eval

    from forge.sql import executor

    assert exec_eval.ALLOWED_ACTIONS == executor.ALLOWED_ACTIONS
    assert dict(executor.LIMITS) == exec_eval.LIMITS
    # Results are bounded the same way, and their rows counted alike.
    assert exec_eval.MAX_RESULT_BYTES == executor.MAX_RESULT_BYTES
    assert exec_eval.MAX_VALUE_SIZE == executor.MAX_VALUE_SIZE
    for row in [(), (None,), (1, 2.5, "text", b"\x00blob"), ("\U0001f600" * 3, 2**70)]:
        assert exec_eval.row_size(row) == executor.row_size(row)


@pytest.mark.phase(1)
def test_evaluator_imports_without_nltk() -> None:
    # Execution matching and hardness from Spider's parsed `sql` field must not need nltk.
    from test_suite_eval import evaluation, exec_eval

    assert callable(exec_eval.eval_exec_match)
    assert callable(evaluation.Evaluator().eval_hardness)
    assert "nltk" not in sys.modules
