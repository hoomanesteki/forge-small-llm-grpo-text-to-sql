"""The time guard: a job stops cleanly before its machine's session ends.

A Kaggle session ends at 12 hours, and what is still running then is lost, so a job gets a
deadline short of it (plan section 3.8 sets 11 h 15 min for a full session, which leaves
time to save and upload). ``Deadline`` counts it on a monotonic clock from when it is
made, and times the job's steps: a training step, or an eval batch. Once the time left is
no more than the longest step so far plus ``margin_s``, the next step might not end in
time, so the job stops before starting it. ``forge.eval.runner`` asks between batches, and
``forge.train.callbacks.TimeGuardCallback`` after each training step.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

Clock = Callable[[], float]  # seconds on a clock that never goes back


class Deadline:
    """A deadline ``minutes`` from now on ``clock``, and the steps timed against it.

    ``margin_s`` is time kept back beyond one step, for what the job does once it stops.
    """

    def __init__(
        self, minutes: float, *, margin_s: float = 0.0, clock: Clock = time.monotonic
    ) -> None:
        if not math.isfinite(minutes) or minutes <= 0:
            raise ValueError(f"a deadline is a positive number of minutes, got {minutes!r}")
        if not math.isfinite(margin_s) or margin_s < 0:
            raise ValueError(f"margin_s must be a non-negative number of seconds, got {margin_s!r}")
        self.minutes = minutes
        self.margin_s = margin_s
        self.clock = clock
        self.at = clock() + minutes * 60
        self.longest_step_s = 0.0
        self._mark = clock()

    def remaining_s(self) -> float:
        """Seconds left before the deadline; below zero once it has passed."""
        return self.at - self.clock()

    def mark(self) -> None:
        """Time the next step from now, rather than from the last mark."""
        self._mark = self.clock()

    def lap(self) -> None:
        """End a step: it lasted from the last mark, which is now."""
        now = self.clock()
        self.longest_step_s = max(self.longest_step_s, now - self._mark)
        self._mark = now

    def near(self) -> bool:
        """Whether a step as long as the longest so far, and the margin, would pass it."""
        return self.remaining_s() <= self.longest_step_s + self.margin_s

    def describe(self) -> str:
        """The deadline's state, for a log line when a job stops at it."""
        return (
            f"{max(0.0, self.remaining_s()):.0f} s left of a {self.minutes:g}-minute deadline, "
            f"the longest step {self.longest_step_s:.0f} s and the margin {self.margin_s:g} s"
        )
