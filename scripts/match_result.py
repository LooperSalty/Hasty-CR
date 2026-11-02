"""Victory or defeat, from the end-of-game screen, by counting crowns.

The layout never moves: the opponent's red banner is always on top, ours
(blue) always below, each with three crown slots above it. A crown won is
gold; an empty slot is a bare red or blue cushion. So each side's crowns are
counted slot by slot, and the side with more crowns won.

(Two earlier versions got this wrong: one assumed the winner's banner moved
to the top - so did scripts/results.py - and read a 3-crown win as a defeat;
one compared the total gold area per side, which an emote bubble hiding one
crown turned into a tie. Calibrated on 35 live result frames: a crown fills
~0.20 of its slot, a half-hidden one ~0.08, an empty slot 0.)

Equal crowns (a tie decided on tower HP, or a draw) read as 'inconnu'.
Frames are kept in tmp/live/results so any result can be checked by eye.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 1080, 1920
SLOTS_X = ((170, 400), (410, 660), (670, 910))      # left, middle, right crown
THEIR_ROWS = (0.12, 0.25)                             # fractions of the height
OUR_ROWS = (0.43, 0.54)
CROWN_SHARE = 0.04          # gold in a slot above this: a crown is there


def _gold(px: np.ndarray) -> np.ndarray:
    r, g, b = (px[:, :, i].astype(int) for i in range(3))
    return (r > 190) & (g > 120) & (g < 225) & (b < 110) & (r - b > 100)


def _count(gold: np.ndarray, rows: tuple[float, float]) -> int:
    """Crowns fill the slots from the left: the rightmost gold slot is the count.

    An emote picture drawn over a crown (the King emote over the left slot,
    2026-10-29) hides it; the gold slot to its right still says it is won.
    """
    top, bottom = round(rows[0] * HEIGHT), round(rows[1] * HEIGHT)
    won = [float(gold[top:bottom, x0:x1].mean()) > CROWN_SHARE for x0, x1 in SLOTS_X]
    return max((index + 1 for index, crown in enumerate(won) if crown), default=0)


def crowns(frame) -> Optional[tuple[int, int]]:
    """(ours, theirs) crowns on an end-of-game frame, or None if unreadable."""
    if frame is None:
        return None
    try:
        image = frame.convert("RGB")
        if image.size != (WIDTH, HEIGHT):
            image = image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
        gold = _gold(np.asarray(image))
        return _count(gold, OUR_ROWS), _count(gold, THEIR_ROWS)
    except Exception:
        # Called on the result screen: an error here must not keep the
        # runner from tapping OK and moving on.
        return None


def banner_outcome(frame) -> str:
    """'victoire', 'défaite' or 'inconnu' for an end-of-game frame."""
    counted = crowns(frame)
    if counted is None:
        return "inconnu"
    ours, theirs = counted
    if ours > theirs:
        return "victoire"
    if theirs > ours:
        return "défaite"
    return "inconnu"
