"""Battle frames kept by the runner, with what it saw and what it played.

The detector's dataset grows from these (scripts/autolabel.py): one frame a
second during a match, each with a JSON sidecar -

* `tracks`: the units seen in that very frame (held tracks are left out,
  their box is from an earlier frame), with name, side, box in 1080x1920
  pixels, confidence and how many frames the track has lasted;
* `plays`: our cards played in the last PLAY_MEMORY seconds, with the tile -
  what our own units are, for certain.

One folder per match, named by its start time. Saving is cheap (a JPEG a
second) and a failure is reported once and never stops the bot.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

PLAY_MEMORY = 12.0          # seconds a play still explains the units on the board
JPEG_QUALITY = 90


class DatasetRecorder:
    def __init__(self, folder: Path, every: float = 1.0):
        self.folder = Path(folder)
        self.every = every
        self.last = float("-inf")
        self.plays: list[tuple[str, tuple[int, int], float]] = []
        self.match_started: Optional[float] = None
        self.match_folder: Optional[Path] = None
        self.saved = 0

    def played(self, card: str, tile, now: float) -> None:
        self.plays = [p for p in self.plays if now - p[2] <= PLAY_MEMORY]
        self.plays.append((card, (int(tile[0]), int(tile[1])), now))

    def _folder_for(self, match_started: float) -> Path:
        if self.match_started != match_started or self.match_folder is None:
            self.match_started = match_started
            base = self.folder / f"match_{datetime.now():%Y%m%d_%H%M%S}"
            folder, n = base, 1
            while folder.exists():
                n += 1
                folder = base.with_name(f"{base.name}_{n}")
            folder.mkdir(parents=True)
            self.match_folder = folder
        return self.match_folder

    def record(self, frame, perception, now: float, match_started: float) -> bool:
        """Save `frame` and its sidecar if `every` seconds have passed."""
        if now - self.last < self.every:
            return False
        self.last = now
        folder = self._folder_for(match_started)
        stem = f"{now - match_started:07.2f}".replace(".", "_")
        tracks = [{"id": t.id, "name": t.name, "side": t.side,
                   "x": round(float(t.x), 2), "y": round(float(t.y), 2),
                   "box": [int(v) for v in t.box] if t.box else None,
                   "conf": round(float(t.conf), 3), "hits": int(t.hits)}
                  for t in perception.tracker.tracks if not t.missing]
        plays = [{"card": card, "tile": list(tile), "age": round(now - when, 2)}
                 for card, tile, when in self.plays if now - when <= PLAY_MEMORY]
        frame.convert("RGB").save(folder / f"{stem}.jpg", quality=JPEG_QUALITY)
        (folder / f"{stem}.json").write_text(json.dumps(
            {"match_time": round(now - match_started, 2), "tracks": tracks, "plays": plays}),
            encoding="utf-8")
        self.saved += 1
        return True
