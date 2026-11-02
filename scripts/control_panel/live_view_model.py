"""Pure helpers for the live view: docking, reading the bot's files, moving boxes."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VIEW_JSON = "native_view.json"
STALE_SECONDS = 6.0
GAP = 4

Rect = tuple[int, int, int, int]       # left, top, right, bottom


@dataclass(frozen=True)
class View:
    data: dict
    stale: bool
    age: float


@dataclass(frozen=True)
class Poll:
    view: Optional[View]
    changed: bool                      # the file was rewritten since the last poll


class ViewReader:
    """read_view for a 60 Hz timer: the JSON is parsed only when the bot rewrote it."""

    def __init__(self, folder: Path):
        self.path = Path(folder) / VIEW_JSON
        self._stamp: Optional[int] = None
        self._data: Optional[dict] = None

    def poll(self, now: Optional[float] = None) -> Poll:
        now = time.time() if now is None else now
        try:
            stat = self.path.stat()
        except OSError:
            changed = self._data is not None
            self._stamp, self._data = None, None
            return Poll(None, changed)
        changed = stat.st_mtime_ns != self._stamp
        if changed:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return Poll(None, False)         # caught mid-write: try again next tick
            self._stamp, self._data = stat.st_mtime_ns, data
        if self._data is None:
            return Poll(None, changed)
        age = now - stat.st_mtime
        return Poll(View(data=self._data, stale=age > STALE_SECONDS, age=age), changed)


@dataclass(frozen=True)
class Box:
    side: str                          # "ally" / "enemy"
    name: str
    conf: float
    rect: tuple                        # x0, y0, x1, y1 in the bot's 1080x1920 frame
    velocity: tuple = (0.0, 0.0)       # px/s, from the last two detections


def _centre(rect: tuple) -> tuple[float, float]:
    return (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2


def _shift(rect: tuple, dx: float, dy: float) -> tuple:
    return (rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy)


class BoxFollower:
    """Turns ~10 detections a second into boxes that move at the HUD's 60 Hz.

    Each unit is matched to the box it was in the previous detection (same
    side and name, nearest centre). Between detections its box keeps going at
    the speed it was last seen moving, for at most MAX_PREDICT_SECONDS, and
    every drawn box eases toward that prediction rather than jumping to it.
    """

    MATCH_PX = 220              # farthest a unit moves between two published frames
    MAX_SPEED = 200.0           # px/s: faster than any troop, so faster is noise
    MAX_PREDICT_SECONDS = 0.25  # coast this long at most on a stale detection
    EASE_SECONDS = 0.04         # time constant of the glide toward the prediction
    UNIT_SQUARE = 48            # a unit read without a box: a square on its tile

    def __init__(self):
        self.targets: list[Box] = []
        self.drawn: list[tuple] = []
        self.data_time: Optional[float] = None
        self.received = 0.0
        self.last_frame: Optional[float] = None

    @classmethod
    def _rect(cls, unit: dict) -> tuple:
        box = unit.get("box")
        if box:
            return tuple(float(v) for v in box)
        half = cls.UNIT_SQUARE / 2
        return (unit["px"] - half, unit["py"] - half, unit["px"] + half, unit["py"] + half)

    def update(self, data: dict, received: float) -> None:
        """New detections from the bot (`received`: when the HUD read them)."""
        stamp = data.get("time")
        if stamp is not None and stamp == self.data_time:
            return                      # the same frame read again: nothing moved
        units = [Box(side, u["name"], float(u.get("conf", 0)), self._rect(u))
                 for side, key in (("ally", "allies"), ("enemy", "enemies"))
                 for u in data.get(key, [])]
        gap = (stamp - self.data_time) if stamp is not None and self.data_time is not None else 0
        pairs = sorted(
            (math.dist(_centre(old.rect), _centre(new.rect)), ni, oi)
            for ni, new in enumerate(units) for oi, old in enumerate(self.targets)
            if old.side == new.side and old.name == new.name)
        matched: dict[int, int] = {}
        for distance, ni, oi in pairs:
            if distance <= self.MATCH_PX and ni not in matched and oi not in matched.values():
                matched[ni] = oi
        targets, drawn = [], []
        for ni, new in enumerate(units):
            oi = matched.get(ni)
            if oi is None:
                targets.append(new)
                drawn.append(new.rect)
                continue
            old = self.targets[oi]
            targets.append(Box(new.side, new.name, new.conf, new.rect,
                               self._velocity(old, new, gap)))
            drawn.append(self.drawn[oi])
        self.targets, self.drawn = targets, drawn
        self.data_time, self.received = stamp, received

    def _velocity(self, old: Box, new: Box, gap: float) -> tuple:
        if not 0 < gap <= 1.0:
            return (0.0, 0.0)
        (ox, oy), (nx, ny) = _centre(old.rect), _centre(new.rect)
        clamp = lambda v: max(-self.MAX_SPEED, min(self.MAX_SPEED, v))   # noqa: E731
        seen = (clamp((nx - ox) / gap), clamp((ny - oy) / gap))
        # Half the last estimate, half this one: a detection's jitter is not a sprint.
        return tuple((a + b) / 2 for a, b in zip(old.velocity, seen))

    def frame(self, now: float) -> list[Box]:
        """The boxes to draw at `now`."""
        ahead = min(max(0.0, now - self.received), self.MAX_PREDICT_SECONDS)
        step = None if self.last_frame is None else max(0.0, now - self.last_frame)
        alpha = 1.0 if step is None else 1 - math.exp(-step / self.EASE_SECONDS)
        self.last_frame = now
        out, drawn = [], []
        for target, current in zip(self.targets, self.drawn):
            goal = _shift(target.rect, target.velocity[0] * ahead, target.velocity[1] * ahead)
            rect = tuple(c + (g - c) * alpha for c, g in zip(current, goal))
            drawn.append(rect)
            out.append(Box(target.side, target.name, target.conf, rect, target.velocity))
        self.drawn = drawn
        return out


def dock_geometry(mumu: Optional[Rect], screen: Rect, width: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) for the view: beside MuMu, same height, right side first."""
    s_left, s_top, s_right, s_bottom = screen
    if mumu is None:
        return (s_right - width, s_top, width, s_bottom - s_top)
    left, top, right, bottom = mumu
    height = max(400, bottom - top)
    if right + GAP + width <= s_right:
        return (right + GAP, top, width, height)
    if left - GAP - width >= s_left:
        return (left - GAP - width, top, width, height)
    return (s_right - width, top, width, height)


def read_view(folder: Path, now: Optional[float] = None,
              file_age: Optional[float] = None) -> Optional[View]:
    """The bot's last published decision, or None if there is none to read.

    `file_age` is the seconds since the JSON was written; read from the file
    when not given. A partly written file (it is replaced atomically, but a
    copy could still be caught) reads as None rather than raising.
    """
    path = Path(folder) / VIEW_JSON
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if file_age is None:
        file_age = (now if now is not None else time.time()) - path.stat().st_mtime
    return View(data=data, stale=file_age > STALE_SECONDS, age=file_age)
