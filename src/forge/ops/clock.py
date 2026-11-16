"""Today's date in UTC, the one date Forge names and windows things by.

Run ids, ledger rows, dev-usage rows and the ledger's weekly window all take their day
from here, so a job pushed in the evening west of UTC carries one date in all of them.
The laptop's own zone never decides a date.
"""

from __future__ import annotations

from datetime import UTC, date, datetime


def utc_today() -> date:
    return datetime.now(UTC).date()
