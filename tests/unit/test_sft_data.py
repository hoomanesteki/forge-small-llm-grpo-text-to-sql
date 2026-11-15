"""The SFT dataset builder: train ids only, the loss on the reply only, a stable hash."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import load_from_disk
from trl.data_utils import is_conversational

from forge.core.hashing import canonical_json, sha256_hex
from forge.core.types import Example
from forge.data.prompts import chat_token_counter, fit_prompt
from forge.data.spider import load_raw
from forge.data.splits import (
    SplitLeak,
    load_train,
    load_val,
    read_split_ids,
    split_examples,
    write_manifests,
)
from forge.infer.hf_backend import split_ref
from forge.train import sft_data
from forge.train.sft_data import (
    SftCounts,
    _hidden_tables,
    build_sft_dataset,
    read_meta,
    save_sft,
)
from tests.fakes import WordTokenizer

ROOT = Path(__file__).resolve().parents[2]
WORDS = chat_token_counter(WordTokenizer())


def _build(version: str = "p1", tokenizer: str = "org/tok", **kwargs: int) -> sft_data.SftDataset:
    return build_sft_dataset(version, tokenizer, WORDS, **kwargs)


@pytest.mark.phase(4)
def test_train_ids_only(spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _build()

    # Checked against the fixture's own split manifests, written as `forge data splits
    # --write` writes the committed ones.
    write_manifests(split_examples(load_raw()), tmp_path / "splits")
    train = set(read_split_ids("train", tmp_path / "splits"))
    held_out = {
        pair for name in ("val", "dev") for pair in read_split_ids(name, tmp_path / "splits")
    }
    pairs = [(row["example_id"], row["db_id"]) for row in dataset.rows]
    assert set(pairs) <= train
    assert not set(pairs) & held_out
    # Nothing in the fixture's train split is left out, and the rows keep its order.
    assert pairs == [(example.id, example.db_id) for example in load_train()]
    assert dataset.meta.counts.rows == len(pairs) == 40
    # No row is from a database the committed manifests hold out of training either.
    committed = {db for name in ("val", "dev") for _, db in read_split_ids(name)}
    assert not {db_id for _, db_id in pairs} & committed

    # The guard checks what is built, whatever the loader hands over: val questions,
    # and val questions labeled train, as load_raw() labels them.
    val = load_val()
    relabeled = [example.model_copy(update={"split": "train"}) for example in val]
    for leaked in (val, relabeled):
        monkeypatch.setattr(sft_data, "load_train", lambda data_dir=None, rows=leaked: rows)
        with pytest.raises(SplitLeak, match="20 val or dev example"):
            _build()


@pytest.mark.phase(4)
@pytest.mark.requires_full_data
def test_train_ids_only_on_full_data() -> None:
    dataset = _build()

    train = dict(read_split_ids("train"))  # the committed manifest: id -> db_id
    assert all(train.get(row["example_id"]) == row["db_id"] for row in dataset.rows)
    counts = dataset.meta.counts
    assert counts.train == len(train) == 7787
    left_out = counts.duplicates + counts.failing_gold + counts.over_limit
    assert counts.rows == len(dataset.rows) == counts.train - left_out - counts.hidden_gold_tables
    # Exact repeats of a question on its database, case and spacing aside, and the three
    # gold queries that fail on SQLite.
    assert (counts.duplicates, counts.failing_gold) == (25, 3)


@pytest.mark.phase(4)
def test_assistant_only_labels(spider_mini: Path, tiny_model_ref: str, tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    repo, revision = split_ref(tiny_model_ref)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    dataset = build_sft_dataset("p1", tiny_model_ref, chat_token_counter(tokenizer))
    saved = load_from_disk(str(save_sft(dataset, tmp_path / "sft")))

    # TRL's own preparation, as SFT training runs it. The sequence length is ADR-0010's:
    # TRL's default of 1,024 would cut the reply off long prompts.
    args = SFTConfig(
        output_dir=str(tmp_path / "out"),
        use_cpu=True,
        bf16=False,
        report_to="none",
        max_length=2048,
    )
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision)
    trainer = SFTTrainer(model=model, args=args, train_dataset=saved, processing_class=tokenizer)
    assert trainer.completion_only_loss
    prepared = trainer.train_dataset
    assert prepared["example_id"] == [row["example_id"] for row in dataset.rows]

    gold = {example.id: example.gold_sql for example in load_train()}
    for row in prepared:
        prompt = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"]
        n = len(prompt)
        ids, labels = row["input_ids"], row["labels"]
        assert ids[:n] == prompt
        # Every prompt token is masked, every reply token is learned: the query in its
        # code block, then the end of the turn, so the model learns to stop.
        assert labels[:n] == [-100] * n
        assert labels[n:] == ids[n:]
        reply = f"```sql\n{gold[row['example_id']]}\n```{tokenizer.eos_token}\n"
        assert tokenizer.decode(labels[n:]) == reply


@pytest.mark.phase(4)
def test_dataset_hash_stable(spider_mini: Path, tmp_path: Path) -> None:
    first = _build()
    assert _build() == first
    payload = {"rows": first.rows, "prompt_version": "p1", "tokenizer": "org/tok"}
    assert first.meta.hash == sha256_hex(canonical_json(payload))

    # A fresh interpreter with other hash seeds gets the same hash: nothing depends on
    # set order or PYTHONHASHSEED.
    code = (
        "from forge.data.prompts import chat_token_counter\n"
        "from forge.train.sft_data import build_sft_dataset\n"
        "from tests.fakes import WordTokenizer\n"
        "count = chat_token_counter(WordTokenizer())\n"
        "print(build_sft_dataset('p1', 'org/tok', count).meta.hash)\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("1", "2")
    }
    assert outputs == {first.meta.hash + "\n"}

    # Each input to the hash moves it: the prompt version, the tokenizer and the rows.
    fewer = _build(max_tokens=120)
    assert fewer.rows != first.rows
    others = {_build("p2").meta.hash, _build(tokenizer="org/other").meta.hash, fewer.meta.hash}
    assert len(others) == 3 and first.meta.hash not in others

    # Saved under its hash with its meta.json, once: a second save keeps the first.
    folder = save_sft(first, tmp_path / "sft")
    assert folder == tmp_path / "sft" / first.meta.hash
    assert read_meta(folder) == first.meta
    assert load_from_disk(str(folder)).to_list() == first.rows
    written = (folder / "meta.json").stat().st_mtime_ns
    assert save_sft(_build(), tmp_path / "sft") == folder
    assert (folder / "meta.json").stat().st_mtime_ns == written
    assert [path.name for path in (tmp_path / "sft").iterdir()] == [first.meta.hash]

    # A folder whose meta.json names another dataset is refused, not reused.
    stale = tmp_path / "stale"
    shutil.copytree(save_sft(fewer, tmp_path / "other"), stale / first.meta.hash)
    with pytest.raises(ValueError, match=f"holds the dataset with hash {fewer.meta.hash}"):
        save_sft(first, stale)


@pytest.mark.phase(4)
def test_rows_and_what_is_left_out(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fixture with its first question asked again, in other case and spacing, and its
    # second question's gold counted as one that fails on SQLite.
    root = Path(shutil.copytree(spider_mini, tmp_path / "data"))
    items = json.loads((root / "train_spider.json").read_text(encoding="utf-8"))
    repeat = {**items[0], "question": "how many heads of the  departments are older than 56 ?"}
    (root / "train_spider.json").write_text(json.dumps([*items, repeat]), encoding="utf-8")
    monkeypatch.setenv("FORGE_DATA_DIR", str(root))
    monkeypatch.setattr(sft_data, "FAILING_GOLD_IDS", frozenset({"spider-train-00001"}))

    # At 120 words every college_1 and geo prompt is over the limit and loses the
    # columns of the tables its question doesn't name.
    dataset = _build(max_tokens=120)

    over = {"spider-others-00002"}
    hidden = {
        *(f"spider-train-{n:05d}" for n in (31, 34, 35, 36, 38, 39)),
        *(f"spider-others-{n:05d}" for n in (3, 4, 5, 8, 9)),
    }
    assert dataset.meta.counts == SftCounts(
        train=41, duplicates=1, failing_gold=1, over_limit=1, hidden_gold_tables=11, rows=27
    )
    left_out = {"spider-train-00050", "spider-train-00001", *over, *hidden}
    train = load_train()
    ids = [row["example_id"] for row in dataset.rows]
    assert ids == [example.id for example in train if example.id not in left_out]
    stats = dataset.meta.prompts
    assert (stats.prompts, stats.over_limit, stats.dropped_columns, stats.excluded) == (
        39,
        20,
        20,
        1,
    )

    # spider-train-00031 asks about the "accounting dept": DEPARTMENT is shown without
    # columns, and its gold query counts professors by department.
    example = next(e for e in train if e.id == "spider-train-00031")
    assert _hidden_tables(example, ("CLASS", "DEPARTMENT")) == ["DEPARTMENT"]

    # spider-train-00030 is cut down too, but its gold reads only tables shown in full.
    kept = next(e for e in train if e.id == "spider-train-00030")
    fitted = fit_prompt(kept, "p1", WORDS, max_tokens=120)
    assert fitted.dropped_columns
    row = dataset.rows[ids.index(kept.id)]
    assert row == {
        "example_id": "spider-train-00030",
        "db_id": "college_1",
        "prompt": [dict(message) for message in fitted.prompt.messages],
        "completion": [{"role": "assistant", "content": f"```sql\n{kept.gold_sql}\n```"}],
    }
    assert is_conversational(dict(row))
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]


@pytest.mark.phase(4)
def test_hidden_tables_reads_the_gold_query() -> None:
    def example(sql: str) -> Example:
        return Example(
            id="x-1", db_id="db", question="q", gold_sql=sql, hardness="easy", split="train"
        )

    sql = (
        "WITH busy AS (SELECT dept_id FROM Staff GROUP BY dept_id HAVING count(*) > 3) "
        'SELECT name FROM "Dept" JOIN busy USING (dept_id)'
    )
    # Case aside; a common table expression named like a stub is no table.
    assert _hidden_tables(example(sql), ("dept", "STAFF", "busy", "room")) == ["dept", "STAFF"]
    assert _hidden_tables(example("SELECT * FROM ("), ()) == []
    with pytest.raises(ValueError, match="x-1: sqlglot can't read the gold query"):
        _hidden_tables(example("SELECT * FROM ("), ("dept",))
