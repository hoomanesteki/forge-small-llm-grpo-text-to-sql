"""``forge.ops.timeguard.Deadline`` on a clock the tests move by hand."""

from __future__ import annotations

import pytest

from forge.ops.timeguard import Deadline


class Clock:
    """A monotonic clock that only moves when told to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.phase(3)
def test_deadline_stops_before_a_step_that_would_pass_it() -> None:
    clock = Clock()
    deadline = Deadline(1, clock=clock)
    assert deadline.remaining_s() == 60
    assert not deadline.near()  # no step timed yet, and a minute left

    clock.advance(20)
    deadline.lap()
    assert (deadline.longest_step_s, deadline.remaining_s()) == (20, 40)
    assert not deadline.near()  # 40 s left fits another 20 s step

    clock.advance(20)
    deadline.lap()
    assert deadline.near()  # 20 s left: the next step could end on the deadline or after
    assert deadline.describe() == (
        "20 s left of a 1-minute deadline, the longest step 20 s and the margin 0 s"
    )

    clock.advance(30)
    assert deadline.remaining_s() == -10
    assert deadline.describe().startswith("0 s left")


@pytest.mark.phase(3)
def test_longest_step_and_margin_are_kept_back() -> None:
    clock = Clock()
    deadline = Deadline(10, margin_s=120, clock=clock)
    # Loading before the first mark is no step's time.
    clock.advance(240)
    deadline.mark()
    for seconds in (90, 30):
        clock.advance(seconds)
        deadline.lap()
    assert deadline.longest_step_s == 90
    # 600 - 360 = 240 s left: more than 90 + 120, then not.
    assert deadline.remaining_s() == 240 and not deadline.near()
    clock.advance(30)
    assert deadline.remaining_s() == 210 and deadline.near()


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    "minutes, margin_s, message",
    [
        (0, 0.0, "positive number of minutes"),
        (-5, 0.0, "positive number of minutes"),
        (float("nan"), 0.0, "positive number of minutes"),
        (float("inf"), 0.0, "positive number of minutes"),
        (1, -1.0, "non-negative number of seconds"),
        (1, float("nan"), "non-negative number of seconds"),
    ],
)
def test_bad_deadlines_rejected(minutes: float, margin_s: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Deadline(minutes, margin_s=margin_s, clock=Clock())
