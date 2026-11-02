"""Find the blue confirm button (OK / Continue) on a between-match screen.

Arena unlocks and King Tower level-ups end on one saturated blue button and
have no close cross, so `popup_guard` cannot dismiss them. The button's blue
is fully saturated (B=255, G 150-200, R low) where the screen's blue backdrop
stops near B=200, which is what separates the two. Coordinates are returned
in the frame's own pixels, so it works on 540x960 and 1080x1920 alike.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Confirm buttons sit in the lower half; the trophy road's OK is at 93-99%.
SEARCH_TOP, SEARCH_BOTTOM = 0.45, 1.0
MIN_WIDTH, MIN_HEIGHT = 0.12, 0.03         # fractions of the frame
MAX_WIDTH = 0.60                           # wider than this is a banner, not a button
MIN_FILL = 0.55                            # button pixels / bounding box


def _button_pixels(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (b >= 245) & (g >= 145) & (g <= 205) & (r <= 120)


def find_ok_button(image: Image.Image) -> Optional[Tuple[int, int]]:
    from scipy import ndimage

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    height, width = rgb.shape[:2]
    top, bottom = int(height * SEARCH_TOP), int(height * SEARCH_BOTTOM)
    mask = _button_pixels(rgb[top:bottom])
    labels, count = ndimage.label(mask)
    best = None
    for index, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        rows, cols = region
        box_h, box_w = rows.stop - rows.start, cols.stop - cols.start
        if not (MIN_WIDTH * width <= box_w <= MAX_WIDTH * width
                and box_h >= MIN_HEIGHT * height):
            continue
        filled = int((labels[region] == index).sum())
        if filled / (box_w * box_h) < MIN_FILL:
            continue
        if best is None or filled > best[0]:
            centre = ((cols.start + cols.stop) // 2, top + (rows.start + rows.stop) // 2)
            best = (filled, centre)
    return best[1] if best else None
