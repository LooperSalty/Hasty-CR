"""The main menu's Battle tab, found by its icon in the bottom bar.

Left on another tab (Cards after upgrading by hand, Shop, Clan), the game is
on a screen BuildABot does not know: the runner found no popup to close and
relaunched the app, which reopens on the same tab. The tab buttons move
with the selected one, so the Battle tab is located by template matching its
lightning icon along the bottom bar (scripts/assets/battle_tab.png, cut from
the Cards tab).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

TEMPLATE = Path(__file__).resolve().parent / "assets" / "battle_tab.png"
WIDTH, HEIGHT = 1080, 1920
BAR_TOP, BAR_BOTTOM = 1730, 1890      # where the tab icons sit
STEP_X = 4
MAX_MEAN_DIFF = 28.0                  # per-channel mean absolute difference


@lru_cache(maxsize=1)
def _template() -> np.ndarray:
    return np.asarray(Image.open(TEMPLATE).convert("RGB"), dtype=np.int16)


def find_battle_tab(frame) -> Optional[tuple[int, int]]:
    """Centre of the Battle tab button, or None when it is not on screen."""
    image = frame.convert("RGB")
    if image.size != (WIDTH, HEIGHT):
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    template = _template()
    th, tw = template.shape[:2]
    strip = np.asarray(image.crop((0, BAR_TOP, WIDTH, BAR_BOTTOM)), dtype=np.int16)
    best = (float("inf"), 0, 0)
    for y in range(0, strip.shape[0] - th + 1, STEP_X):
        for x in range(0, WIDTH - tw + 1, STEP_X):
            diff = float(np.abs(strip[y:y + th, x:x + tw] - template).mean())
            if diff < best[0]:
                best = (diff, x, y)
    # The coarse grid can sit 2 px off the icon: refine around the best.
    _, bx, by = best
    for y in range(max(0, by - STEP_X + 1), min(strip.shape[0] - th, by + STEP_X - 1) + 1):
        for x in range(max(0, bx - STEP_X + 1), min(WIDTH - tw, bx + STEP_X - 1) + 1):
            diff = float(np.abs(strip[y:y + th, x:x + tw] - template).mean())
            if diff < best[0]:
                best = (diff, x, y)
    diff, x, y = best
    if diff > MAX_MEAN_DIFF:
        return None
    return x + tw // 2, BAR_TOP + y + th // 2
