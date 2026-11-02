"""Running totals read from a bot's output, one line at a time.

Both runners log a keyword after the timestamp (`PLAY #3 ...`, `QUEUE #1`,
`MATCH_END #1 ...`, `RESULT exit ...`), so the panel counts those rather than
asking the bot anything: it works the same for a bot it did not start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DEBUG = re.compile(r"\|\s*DEBUG\s*\|")
_PLAY = re.compile(r"\bPLAY #\d+")
_QUEUE = re.compile(r"\bQUEUE (#\d+|match=\d+)")
_FINISHED = re.compile(r"\b(MATCH_END #\d+|RESULT exit)")
_ERROR = re.compile(r"\b(ERROR|Traceback)\b")


@dataclass(frozen=True)
class LogStats:
    plays: int = 0
    queued: int = 0
    finished: int = 0
    errors: int = 0
    last_event: str = ""
    last_error: str = ""


def clean_line(line: str) -> str:
    """`line` without terminal colour codes (loguru colours its output)."""
    return _ANSI.sub("", line).rstrip()


def is_debug(line: str) -> bool:
    """BuildABot's per-frame loguru chatter, hidden unless asked for."""
    return bool(_DEBUG.search(line))


def update_stats(stats: LogStats, line: str) -> LogStats:
    """A new LogStats with `line` accounted for; `stats` is left untouched."""
    text = line.strip()
    if not text:
        return stats
    changes: dict = {"last_event": text}
    if _PLAY.search(text):
        changes["plays"] = stats.plays + 1
    if _QUEUE.search(text):
        changes["queued"] = stats.queued + 1
    if _FINISHED.search(text):
        changes["finished"] = stats.finished + 1
    if _ERROR.search(text):
        changes["errors"] = stats.errors + 1
        changes["last_error"] = text
    return replace(stats, **changes)
