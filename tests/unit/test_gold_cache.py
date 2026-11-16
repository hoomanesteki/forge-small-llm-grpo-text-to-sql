"""The gold cache file: values come back exactly, and a file that doesn't fit is rebuilt."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from forge.core.types import Example, ExecResult
from forge.data.manifest import data_manifest_hash
from forge.data.spider import FAILING_GOLD_IDS
from forge.data.splits import SplitLeak, load_train, load_val
from forge.eval import gold_cache
from forge.eval.gold_cache import (
    FORMAT,
    CacheHeader,
    GoldFailed,
    SplitGold,
    cache_path,
    gold_results,
    read_cache,
    split_gold,
    write_cache,
)


def _example(index: int, sql: str = "SELECT 1") -> Example:
    return Example(
        id=f"spider-train-{index:05d}",
        db_id="concert_singer",
        question="How many singers are there?",
        gold_sql=sql,
        hardness="easy",
        split="train",
    )


@pytest.mark.phase(2)
def test_cache_round_trips_every_value(tmp_path: Path) -> None:
    examples = [_example(0), _example(1), _example(2)]
    results = {
        examples[0].id: ExecResult(
            ok=True,
            columns=["blob", "pos", "neg", "null", "big", "text"],
            rows=[
                (b"\x00\xff\xfe", float("inf"), float("-inf"), None, 2**62, "naïve ✓"),
                (b"", 0.1, -0.0, 1, -(2**63), ""),
            ],
            ms=1.25,
        ),
        examples[1].id: ExecResult(ok=True, columns=["n"], rows=[], ms=0.5),
        examples[2].id: ExecResult(
            ok=True, columns=["x"], rows=[(1.0000000000000002,)], truncated=True
        ),
    }
    header = CacheHeader(split="train", data_manifest_hash="a" * 64, examples=3)
    path = cache_path(tmp_path, "train", "a" * 64)
    assert path == tmp_path / "gold" / f"train-{'a' * 12}.jsonl"

    write_cache(path, header, examples, results)
    back = read_cache(path, header, examples)
    assert back == results
    assert back is not None and back[examples[0].id].rows is not None
    assert [type(value) for value in back[examples[0].id].rows[0]] == [
        *(bytes, float, float, type(None), int, str)
    ]
    # Plain JSON, one object a line, and no temporary file left beside it.
    assert all(isinstance(json.loads(line), dict) for line in path.read_text().splitlines())
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.phase(2)
def test_cache_that_does_not_fit_is_rebuilt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    examples = [_example(0), _example(1)]
    results = {
        example.id: ExecResult(ok=True, columns=["one"], rows=[(1,)]) for example in examples
    }
    header = CacheHeader(split="val", data_manifest_hash="b" * 64, examples=2)
    path = cache_path(tmp_path, "val", "b" * 64)
    assert read_cache(path, header, examples) is None
    write_cache(path, header, examples, results)
    assert read_cache(path, header, examples) == results

    # Built for other settings or other questions: rebuilt, never misread.
    assert read_cache(path, header.model_copy(update={"max_rows": 10_000}), examples) is None
    assert read_cache(path, header.model_copy(update={"format": FORMAT + 1}), examples) is None
    edited = [examples[0], examples[1].model_copy(update={"gold_sql": "SELECT 2"})]
    assert read_cache(path, header, edited) is None
    assert read_cache(path, header, [examples[0], _example(7)]) is None

    text = path.read_text()
    with caplog.at_level(logging.WARNING, logger="forge.eval.gold_cache"):
        path.write_text(text.replace('"rows":[[1]]', '"rows":[[{"pickle":"x"}]]', 1))
        assert read_cache(path, header, examples) is None
        path.write_text(text[:-20])
        assert read_cache(path, header, examples) is None
        path.write_text("")
        assert read_cache(path, header, examples) is None
    assert sum("can't be read" in message for message in caplog.messages) == 3


@pytest.mark.phase(2)
def test_gold_runs_once_and_a_failing_gold_is_refused(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val = load_val()
    built = gold_results(val, "val", "c" * 64, cache_root=tmp_path)
    assert built.keys() == {example.id for example in val}
    assert all(result.ok and not result.truncated for result in built.values())

    def no_second_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("the gold queries ran again")

    with monkeypatch.context() as patch:
        patch.setattr(gold_cache, "run_gold", no_second_build)
        assert gold_results(val, "val", "c" * 64, cache_root=tmp_path) == built

    broken = val[0].model_copy(update={"gold_sql": "SELECT wingspan FROM aircraft"})
    with pytest.raises(GoldFailed, match=rf"{broken.id} fails on flight_1 \(no_such_column"):
        gold_results([broken], "val", "d" * 64, cache_root=tmp_path)
    assert not cache_path(tmp_path, "val", "d" * 64).exists()


@pytest.mark.phase(5)
def test_split_gold_serves_train_and_val_only(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = data_manifest_hash(spider_mini)
    train = split_gold("train", cache_root=tmp_path)
    assert train.built
    assert train.path == cache_path(tmp_path, "train", digest)
    assert train.results.keys() == {e.id for e in load_train() if e.id not in FAILING_GOLD_IDS}
    # The file is the one gold_results reads, so the eval runner and the rewards share it.
    with monkeypatch.context() as patched:
        patched.setattr(gold_cache, "run_gold", _no_second_build)
        again = split_gold("train", cache_root=tmp_path)
        assert again == SplitGold(train.path, train.results, built=False)
        assert gold_results(load_train(), "train", digest, cache_root=tmp_path) == train.results

    val = split_gold("val", cache_root=tmp_path)
    assert val.results.keys() == {example.id for example in load_val()}
    assert sorted(path.name for path in (tmp_path / "gold").iterdir()) == [
        f"train-{digest[:12]}.jsonl",
        f"val-{digest[:12]}.jsonl",
    ]
    with pytest.raises(ValueError, match="for train and val only"):
        split_gold("dev", cache_root=tmp_path)  # type: ignore[arg-type]
    # Train passes the train-only guard as well as load_train.
    monkeypatch.setattr(gold_cache, "load_train", lambda root: load_val(root))
    with pytest.raises(SplitLeak):
        split_gold("train", cache_root=tmp_path)


def _no_second_build(*args: object, **kwargs: object) -> None:
    raise AssertionError("the gold queries ran again")
