"""The in-training eval (FG-504): a fixed subset of val, scored by fast EX on the fixture
databases at the steps it is due, with a stand-in for the training model's replies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from forge.core.types import GenParams, Prompt
from forge.data.splits import load_train, load_val
from forge.eval.callbacks import EX_KEY, ValEvalCallback, val_eval_callback, val_subset
from forge.infer.base import FakeBackend, InferenceBackend
from forge.sql.sandbox import SandboxPool
from tests.fakes import WordTokenizer


@pytest.mark.phase(5)
def test_val_subset_is_fixed_and_val_only(spider_mini: Path) -> None:
    val = load_val()
    subset = val_subset(5)
    # The same questions every time, in id order, from val's databases alone.
    assert [example.id for example in val_subset(5)] == [example.id for example in subset]
    assert [example.id for example in subset] == sorted(example.id for example in subset)
    assert {example.db_id for example in subset} <= {example.db_id for example in val}
    assert not {example.db_id for example in subset} & {e.db_id for e in load_train()}
    # A hash draw, not the first questions by id.
    assert [example.id for example in subset] != sorted(example.id for example in val)[:5]
    # The fixture's val is smaller than the plan's 200, so all of it is asked.
    assert sorted(example.id for example in val_subset(200)) == sorted(e.id for e in val)
    with pytest.raises(ValueError, match="asks 1 question or more"):
        val_subset(0)


def _state(step: int, max_steps: int) -> TrainerState:
    return TrainerState(global_step=step, max_steps=max_steps)


@pytest.mark.phase(5)
def test_val_eval_scores_and_records_at_its_steps(spider_mini: Path, tmp_path: Path) -> None:
    examples = val_subset(6)
    right = {example.id for example in examples[::2]}
    gold_sql = {example.id: example.gold_sql for example in examples}

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        # Greedy, as the eval asks, and the length the run's completions are capped at.
        assert (params.temperature, params.max_new_tokens, params.n) == (0.0, 16, 1)
        if prompt.example_id in right:
            return f"```sql\n{gold_sql[prompt.example_id]}\n```"
        return "```sql\nSELECT 1\n```" if prompt.example_id != examples[1].id else "no SQL"

    given: list[tuple[Any, Any, str]] = []

    def backend_for(model: Any, tokenizer: Any, model_ref: str) -> InferenceBackend:
        given.append((model, tokenizer, model_ref))
        return FakeBackend(reply, model_ref=model_ref)

    lines: list[tuple[int, dict[str, Any]]] = []
    with SandboxPool(1, data_dir=spider_mini) as pool:
        callback = val_eval_callback(
            pool,
            WordTokenizer(),
            every=2,
            examples=6,
            prompt_version="p1",
            max_new_tokens=16,
            model_ref="tiny",
            record=lambda step, line: lines.append((step, dict(line))),
            cache_root=tmp_path,
            backend_for=backend_for,
        )
        assert [example.id for example in callback.examples] == [e.id for e in examples]
        args = TrainingArguments(output_dir=str(tmp_path / "out"), report_to="none")
        model, tokenizer = object(), object()
        for step in range(1, 5):
            control = TrainerControl(should_training_stop=step == 4)
            callback.on_step_end(
                args, _state(step, 4), control, model=model, processing_class=tokenizer
            )
        # Step 2 of 6, where the time guard stops training: the deadline kept no time.
        guarded = TrainerControl(should_save=True, should_training_stop=True)
        callback.on_step_end(args, _state(2, 6), guarded, model=model, processing_class=tokenizer)

    # Steps 2 and 4, the run's last included; the guard's stop has no line.
    assert [step for step, _ in lines] == [2, 4]
    assert given == [(model, tokenizer, "tiny")] * 2
    for _, line in lines:
        assert line[EX_KEY] == pytest.approx(3 / 6)
        # Every reply but the one with no SQL ran: SELECT 1 is valid, and wrong.
        assert line["val/valid"] == pytest.approx(5 / 6)
        assert line["val/questions"] == 6.0 and line["val/seconds"] >= 0


@pytest.mark.phase(5)
def test_val_eval_refuses_what_it_cannot_score(spider_mini: Path, tmp_path: Path) -> None:
    examples = val_subset(2)
    prompts = [Prompt(example_id=e.id, messages=(), prompt_version="p1") for e in examples]
    with SandboxPool(1, data_dir=spider_mini) as pool:
        for every, use, gold, message in (
            (0, prompts, {}, "every 1 step or more"),
            (1, prompts[::-1], {}, "one each, in their order"),
            (1, prompts, {}, f"no gold result for {examples[0].id}"),
        ):
            with pytest.raises(ValueError, match=message):
                ValEvalCallback(
                    examples,
                    use,
                    gold,
                    pool,
                    every=every,
                    max_new_tokens=8,
                    model_ref="tiny",
                    record=lambda step, line: None,
                )
