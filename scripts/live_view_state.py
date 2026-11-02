"""What the native runner publishes for the HUD over MuMu and the decision panel.

After every frame it decided on, the runner writes one JSON: the units it
read (box, tile, confidence, side), its hand, both elixir bars - its own as
read, the opponent's as inferred - and why it is doing what it does.
The HUD draws straight over MuMu's window, so no image is written at all:
the viewer already sees the game, and watching costs the bot nothing.

The file is written to a temporary name and moved into place, so a reader
never sees half of it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

VIEW_JSON = "native_view.json"
REPLACE_ATTEMPTS = 4
REPLACE_RETRY_SECONDS = 0.002
TOWERS = {
    "enemy_left": "left_enemy_princess_hp", "enemy_right": "right_enemy_princess_hp",
    "ally_left": "left_ally_princess_hp", "ally_right": "right_ally_princess_hp",
}


def _units(group, tile_to_px, box_scale: float) -> list[dict]:
    out = []
    for det in group:
        x, y = float(det.position.tile_x), float(det.position.tile_y)
        px, py = tile_to_px(x, y)
        bbox = tuple(getattr(det.position, "bbox", ()) or ())
        box = ([round(value * box_scale) for value in bbox]
               if len(bbox) == 4 and any(bbox) else None)
        out.append({"name": det.unit.name, "x": round(x, 1), "y": round(y, 1),
                    "conf": round(float(det.position.conf), 2),
                    "px": int(px), "py": int(py), "box": box})
    return out


def snapshot(state, *, phase: str, reason: str, recent: Sequence[str],
             target: Optional[tuple], strategy: str, finished: int, actions: int,
             enemy_elixir: Optional[float], tile_to_px: Callable[[float, float], tuple],
             box_scale: float = 1.0, record: Optional[dict] = None) -> dict:
    """A JSON-ready picture of one decision, in 1080x1920 frame coordinates.

    `box_scale` brings unit boxes to that frame: 1.0 for YOLO, which ran on
    the 1080x1920 frame, 1080/368 for BuildABot's own model.
    """
    numbers = state.numbers
    names = [card.name for card in state.cards]
    return {
        "screen": state.screen.name,
        "elixir": int(numbers.elixir.number),
        "enemy_elixir": enemy_elixir,
        "hand": [{"name": name, "ready": slot in state.ready}
                 for slot, name in enumerate(names[1:5])],
        "next_card": names[0],
        "allies": _units(state.allies, tile_to_px, box_scale),
        "enemies": _units(state.enemies, tile_to_px, box_scale),
        "towers": {key: round(float(getattr(numbers, attr).number), 2)
                   for key, attr in TOWERS.items()},
        "phase": phase, "reason": reason, "recent": list(recent),
        "target": list(target) if target else None,
        "strategy": strategy, "finished": finished, "actions": actions,
        "record": dict(record or {}),
    }


class ViewWriter:
    """Writes the view file at most every `every` seconds."""

    def __init__(self, folder: Path, every: float = 0.05):
        self.folder = Path(folder)
        self.every = every
        self.last = float("-inf")
        self.folder.mkdir(parents=True, exist_ok=True)

    def write(self, data: dict, now: float) -> bool:
        if now - self.last < self.every:
            return False
        self.last = now
        part = self.folder / f"{VIEW_JSON}.part"
        part.write_text(json.dumps({**data, "time": now}, ensure_ascii=False),
                        encoding="utf-8")
        # Windows will not replace a file another process has open, and the
        # HUD reads this one 60 times a second: retry briefly, then let the
        # next frame try again rather than raise into the bot's loop.
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(part, self.folder / VIEW_JSON)
                return True
            except PermissionError:
                if attempt + 1 < REPLACE_ATTEMPTS:
                    time.sleep(REPLACE_RETRY_SECONDS)
        return False
