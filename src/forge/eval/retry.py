"""Execution-guided retry (FG-206): one more turn for a query that fails to run.

With ``--retry 1`` the eval runner asks again for every sample whose query failed to run:
its result isn't ``ok``, which takes in a reply with no SQL, a denied statement, an error
and a timeout. A query that ran is never retried, even when its rows are wrong, since
there is no error to show. The second turn holds the prompt's messages, the first reply
as the assistant's turn, and ``RETRY_MESSAGE`` with the error as the user's. The sample
is scored on the second reply, whose query runs in the run's ``SandboxPool`` like the
first: model SQL is untrusted (ADR-0014).

A query that timed out is told the timeout, not the time its kill measured, which varies,
so a resumed run asks the same second question. Each retry is drawn with a seed of its
own (``retry_params``): the first reply's seed would steer it to the choices that made
the first reply.

``retried`` marks a second reply's score as its sample's second attempt (ADR-0015), and
``without_retry`` turns a score back into what its sample scored on the first reply
alone, which reports show beside the scores with retry.
"""

from __future__ import annotations

from collections.abc import Sequence

from forge.core.types import ExecResult, Generation, GenParams, Prompt, Score
from forge.eval.config import PRED_TIMEOUT_MS
from forge.infer.base import InferenceBackend, sample_seed

RETRY_MESSAGE = "That query failed with: {error}. Reply with a corrected query."
# Seeds a retry's draws where an example id would: no Spider question has this id.
_RETRY = "retry"
NEVER_RETRIED = "a query that ran is never retried, even when its rows are wrong"

# What a retry is asked from: the prompt, its first reply, and that reply's result.
Ask = tuple[Prompt, Generation, ExecResult]


def retry_message(result: ExecResult) -> str:
    """The user's turn after a query whose result is ``result``.

    Raises ``ValueError`` for a query that ran, or a failure with no error to show.
    """
    if result.ok:
        raise ValueError(NEVER_RETRIED)
    if result.error_kind == "timeout":
        error = f"timed out after {PRED_TIMEOUT_MS} ms"
    elif result.error:
        error = result.error
    else:
        raise ValueError("the failed query's result has no error to show")
    return RETRY_MESSAGE.format(error=error)


def retry_prompt(prompt: Prompt, first: Generation, result: ExecResult) -> Prompt:
    """The second turn: ``prompt``'s messages, ``first`` as the assistant's reply, then
    ``retry_message(result)`` as the user's."""
    if first.example_id != prompt.example_id:
        raise ValueError(
            f"a reply to {first.example_id} can't be retried with the prompt of {prompt.example_id}"
        )
    return Prompt(
        example_id=prompt.example_id,
        messages=(
            *prompt.messages,
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": retry_message(result)},
        ),
        prompt_version=prompt.prompt_version,
    )


def retry_params(params: GenParams, sample_idx: int) -> GenParams:
    """How the retries of sample ``sample_idx`` are drawn: one reply a prompt, with a seed
    unlike the first replies' and the other samples' retries."""
    return params.model_copy(update={"n": 1, "seed": sample_seed(params.seed, _RETRY, sample_idx)})


def ask_again(
    backend: InferenceBackend, asks: Sequence[Ask], params: GenParams
) -> list[Generation]:
    """The second reply to each ask, in order, with its first reply's sample index.

    A backend seeds a reply by its example id and its index among the prompt's replies,
    so retries of one question's samples, asked together, would share their draws. Each
    sample index gets a call of its own, with its own seed.
    """
    positions: dict[int, list[int]] = {}
    for position, (_, first, _) in enumerate(asks):
        positions.setdefault(first.sample_idx, []).append(position)
    seconds: dict[int, Generation] = {}
    for sample_idx, group in sorted(positions.items()):
        prompts = [retry_prompt(*asks[position]) for position in group]
        replies = backend.generate(prompts, retry_params(params, sample_idx))
        if [(r.example_id, r.sample_idx) for r in replies] != [(p.example_id, 0) for p in prompts]:
            raise RuntimeError(
                f"the backend returned {len(replies)} replies to {len(prompts)} retry prompts, "
                "or not one each in prompt order"
            )
        for position, reply in zip(group, replies, strict=True):
            seconds[position] = reply.model_copy(update={"sample_idx": sample_idx})
    return [seconds[position] for position in range(len(asks))]


def retried(score: Score, first: ExecResult) -> Score:
    """``score``, of a second reply, as its sample's second attempt.

    The first reply's official EX comes from the run's official batch, unless its query
    timed out: that is no match without the evaluator, as in scoring (ADR-0008).
    """
    if first.ok:
        raise ValueError(NEVER_RETRIED)
    first_try_ex = False if first.error_kind == "timeout" else None
    return score.model_copy(update={"attempts": 2, "first_try_ex": first_try_ex})


def without_retry(score: Score) -> Score:
    """What ``score``'s sample scored on its first reply alone.

    A sample with one reply is its score. A retried one's first query failed to run, so
    it was neither valid nor a fast match, and its official EX is ``first_try_ex``. It
    has no error class: only the reply a score is for gets one.
    """
    if score.attempts == 1:
        return score
    return Score(
        example_id=score.example_id,
        sample_idx=score.sample_idx,
        hardness=score.hardness,
        valid=False,
        ex_fast=False,
        ex_official=score.first_try_ex,
    )
