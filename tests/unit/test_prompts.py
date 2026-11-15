"""Prompt messages in the section 3.7 format, token counting, and the overflow policy."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from forge.core.types import Example, Prompt
from forge.data.prompts import (
    FittedPrompt,
    PromptStats,
    build_prompt,
    chat_token_counter,
    fit_prompt,
    load_tokenizer,
    names_table,
    prompt_stats,
    render_prompt_stats,
)
from forge.data.schema import schema_tables, schema_text
from forge.data.spider import load_raw
from tests.fakes import WordTokenizer

SYSTEM = (
    "You write SQLite queries. Reply with exactly one SQL query in a ```sql code block. "
    "No explanation."
)


def _fixture_example(example_id: str) -> Example:
    return next(example for example in load_raw() if example.id == example_id)


def _schema_lines(fitted: FittedPrompt) -> list[str]:
    user = fitted.prompt.messages[1]["content"]
    schema, _, _ = user.removeprefix("Database schema:\n").partition("\n\nQuestion: ")
    return schema.splitlines()


@pytest.mark.phase(1)
def test_prompt_messages_shape(spider_mini: Path) -> None:
    example = _fixture_example("spider-train-00010")
    assert example.question == "How many aircrafts do we have?"

    prompt = build_prompt(example, "p1")

    assert prompt.example_id == "spider-train-00010"
    assert prompt.prompt_version == "p1"
    assert prompt.messages == (
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "Database schema:\n"
            + schema_text("flight_1", "p1")
            + "\n\nQuestion: How many aircrafts do we have?",
        },
    )

    p2 = build_prompt(example, "p2")
    assert p2.prompt_version == "p2"
    assert p2.messages[0] == prompt.messages[0]
    assert p2.messages[1]["content"] == (
        "Database schema:\n"
        + schema_text("flight_1", "p2")
        + "\n\nQuestion: How many aircrafts do we have?"
    )

    # The question goes on one line, like the schema.
    spaced = example.model_copy(update={"question": "How  many\naircrafts? "})
    assert (
        build_prompt(spaced, "p1")
        .messages[1]["content"]
        .endswith("\n\nQuestion: How many aircrafts?")
    )
    with pytest.raises(ValueError, match="unknown prompt version"):
        build_prompt(example, "p9")


@pytest.mark.phase(1)
def test_chat_token_counter_encodes_like_grpo(spider_mini: Path) -> None:
    tokenizer = WordTokenizer()
    prompt = build_prompt(_fixture_example("spider-train-00010"), "p1")
    words = sum(len(message["content"].split()) for message in prompt.messages)

    # Two role tags and the opened assistant turn, on top of the words.
    assert chat_token_counter(tokenizer)(prompt) == words + 3
    [(conversation, kwargs)] = tokenizer.calls
    assert conversation == list(prompt.messages)
    assert kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "enable_thinking": False,
    }


@pytest.mark.phase(1)
def test_overflow_policy(spider_mini: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="forge.data.prompts")
    example = _fixture_example("spider-train-00025")
    # It names employee and certificate ("employees", "certificates"), but not flight or
    # aircraft, whose columns are the ones step 2 drops.
    assert example.question.startswith("Show names for all employees who have certificates")
    count = chat_token_counter(WordTokenizer())
    full = count(build_prompt(example, "p2"))
    without_samples = count(build_prompt(example, "p1"))
    smallest = fit_prompt(example, "p2", count, max_tokens=0).tokens
    assert full > without_samples > smallest
    flight, aircraft, employee, certificate = schema_tables("flight_1", "p1")
    caplog.clear()

    # It fits: nothing changes and nothing is logged.
    fits = fit_prompt(example, "p2", count, max_tokens=full)
    assert fits == FittedPrompt(prompt=build_prompt(example, "p2"), tokens=full, full_tokens=full)
    assert caplog.messages == []

    # Step 1: the sample rows go first, leaving the p1 schema.
    step1 = fit_prompt(example, "p2", count, max_tokens=full - 1)
    assert step1.prompt.messages == build_prompt(example, "p1").messages
    assert step1.prompt.prompt_version == "p2"
    assert (step1.tokens, step1.full_tokens) == (without_samples, full)
    assert step1.dropped_samples
    assert step1.dropped_columns == ()
    assert not step1.exclude_from_training
    assert caplog.messages == [
        f"spider-train-00025: {full} tokens is over {full - 1}; dropped the sample rows: "
        f"{without_samples} tokens"
    ]
    caplog.clear()

    # Step 2: then the columns of the tables the question doesn't name.
    step2 = fit_prompt(example, "p2", count, max_tokens=without_samples - 1)
    assert step2.dropped_samples
    assert step2.dropped_columns == ("flight", "aircraft")
    assert _schema_lines(step2) == [
        "CREATE TABLE flight (...)",
        "CREATE TABLE aircraft (...)",
        employee.create,
        certificate.create,
    ]
    assert step2.tokens == smallest
    assert not step2.exclude_from_training
    assert len(caplog.messages) == 2
    assert caplog.messages[1] == (
        f"spider-train-00025: {without_samples} tokens is over {without_samples - 1}; "
        "dropped the columns of 2 table(s) the question doesn't name (flight, aircraft): "
        f"{smallest} tokens"
    )
    caplog.clear()

    # Step 3: still too long, so training leaves it out; the prompt is the shortest one.
    step3 = fit_prompt(example, "p2", count, max_tokens=smallest - 1)
    assert step3.exclude_from_training
    assert step3.prompt == step2.prompt
    assert (step3.tokens, step3.full_tokens) == (smallest, full)
    assert len(caplog.messages) == 3
    assert caplog.messages[2] == (
        f"spider-train-00025: still {smallest} tokens, over {smallest - 1}; "
        "excluded from training, kept in evaluation"
    )
    caplog.clear()

    # p1 has no sample rows, so it starts at step 2.
    p1 = fit_prompt(example, "p1", count, max_tokens=without_samples - 1)
    assert not p1.dropped_samples
    assert p1.dropped_columns == ("flight", "aircraft")
    assert len(caplog.messages) == 1
    assert "dropped the columns of 2 table(s)" in caplog.messages[0]
    assert {flight.name, aircraft.name} == set(p1.dropped_columns)


@pytest.mark.phase(1)
def test_overflow_policy_excludes_when_every_table_is_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="forge.data.prompts")
    path = tmp_path / "database" / "solo" / "solo.sqlite"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE aircraft (aid int PRIMARY KEY, name text)")
    conn.close()
    example = Example(
        id="x-1",
        db_id="solo",
        question="How many aircraft are there?",
        gold_sql="SELECT count(*) FROM aircraft",
        hardness="easy",
        split="train",
    )

    fitted = fit_prompt(
        example, "p1", chat_token_counter(WordTokenizer()), max_tokens=5, data_dir=tmp_path
    )

    assert fitted.exclude_from_training
    assert (fitted.dropped_samples, fitted.dropped_columns) == (False, ())
    assert fitted.prompt == build_prompt(example, "p1", data_dir=tmp_path)
    assert caplog.messages == [
        f"x-1: still {fitted.tokens} tokens, over 5; excluded from training, kept in evaluation"
    ]


@pytest.mark.phase(1)
def test_names_table_is_a_case_insensitive_substring_match() -> None:
    assert names_table("How many Aircrafts do we have?", "aircraft")
    assert names_table("List every FLIGHT", "flight")
    assert names_table("which states have border  info", "border_info")
    assert names_table("count the border_info rows", "Border_Info")
    assert not names_table("what is the longest river", "lake")
    assert not names_table("departments managed by a secretary", "management")


@pytest.mark.phase(1)
def test_prompt_stats_counts_lengths_and_overflow() -> None:
    prompt = Prompt(example_id="x", messages=(), prompt_version="p2")
    # Lengths 100 to 2000; the five from 1600 up are over the limit. Of those, every one
    # lost its sample rows, three lost columns too, and one is still too long.
    fitted = [
        FittedPrompt(
            prompt=prompt,
            tokens=min(length, 1500),
            full_tokens=length,
            dropped_samples=length > 1536,
            dropped_columns=("t",) if length >= 1800 else (),
            exclude_from_training=length == 2000,
        )
        for length in range(2000, 0, -100)
    ]

    stats = prompt_stats(fitted, max_tokens=1536)

    assert stats == PromptStats(
        prompts=20,
        p50=1000,
        p95=1900,
        max=2000,
        max_tokens=1536,
        over_limit=5,
        dropped_samples=5,
        dropped_columns=3,
        excluded=1,
    )
    assert [line.rsplit(maxsplit=1) for line in render_prompt_stats(stats).splitlines()] == [
        ["prompts", "20"],
        ["tokens p50", "1000"],
        ["tokens p95", "1900"],
        ["tokens max", "2000"],
        ["over 1536 tokens", "5"],
        ["  sample rows dropped", "5"],
        ["  columns dropped", "3"],
        ["  excluded from training", "1"],
    ]
    # Nearest rank picks a length that occurs, even for a single prompt.
    one = prompt_stats(fitted[:1])
    assert (one.p50, one.p95, one.max) == (2000, 2000, 2000)
    with pytest.raises(ValueError, match="no prompts"):
        prompt_stats([])


@pytest.mark.phase(4)
def test_load_tokenizer_takes_a_pinned_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoTokenizer

    calls: list[tuple[str, str | None]] = []

    def fake_from_pretrained(repo: str, *, revision: str | None) -> WordTokenizer:
        calls.append((repo, revision))
        return WordTokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_from_pretrained)
    for name in ("org/model@0123abc", "org/model", str(tmp_path)):
        assert isinstance(load_tokenizer(name), WordTokenizer)
    # A pinned commit, the default branch, and a local directory as it is.
    assert calls == [("org/model", "0123abc"), ("org/model", None), (str(tmp_path), None)]
    with pytest.raises(FileNotFoundError, match="neither a directory nor a Hub repo id"):
        load_tokenizer(str(tmp_path / "no" / "such" / "model"))
