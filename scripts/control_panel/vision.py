"""What the bot sees: BuildABot's detector run on a screenshot, drawn over it.

This is the same `Detector` the native runner plays with, given the same
deck, so the boxes, confidences, hand and elixir shown are the inputs its
decisions are made from - not a separate model that merely looks similar.

The pure half (`seen_from_state`, `summary`, `scale_bbox`) is tested; the
detector and the drawing are loaded lazily so the panel starts without
onnxruntime unless the overlay is switched on.
"""

from __future__ import annotations

import io
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DETECTOR_W, DETECTOR_H = 368, 652
ROOT = Path(__file__).resolve().parents[2]
TOWER_KEYS = {
    "enemy_left": "left_enemy_princess_hp", "enemy_right": "right_enemy_princess_hp",
    "ally_left": "left_ally_princess_hp", "ally_right": "right_ally_princess_hp",
}


@dataclass(frozen=True)
class Box:
    label: str
    conf: float
    ally: bool
    bbox: tuple[float, float, float, float]       # screenshot pixels


@dataclass(frozen=True)
class Seen:
    boxes: tuple[Box, ...] = ()
    hand: tuple[str, ...] = ()
    next_card: str = ""
    ready: tuple[int, ...] = ()
    elixir: Optional[int] = None
    towers: dict = field(default_factory=dict)
    tower_boxes: dict = field(default_factory=dict)
    screen: str = ""


def scale_bbox(bbox, frame_size) -> tuple[float, float, float, float]:
    """A box in the detector's 368x652 frame, in screenshot pixels."""
    sx, sy = frame_size[0] / DETECTOR_W, frame_size[1] / DETECTOR_H
    left, top, right, bottom = bbox
    return (left * sx, top * sy, right * sx, bottom * sy)


def seen_from_state(state, frame_size) -> Seen:
    if state is None:
        return Seen()
    boxes = tuple(
        Box(det.unit.name, float(det.position.conf), ally,
            scale_bbox(det.position.bbox, frame_size))
        for ally, group in ((True, state.allies), (False, state.enemies))
        for det in group)
    names = [card.name for card in state.cards]
    numbers = state.numbers
    towers = {key: getattr(numbers, attr).number for key, attr in TOWER_KEYS.items()}
    tower_boxes = {key: scale_bbox(getattr(numbers, attr).bbox, frame_size)
                   for key, attr in TOWER_KEYS.items()}
    return Seen(boxes=boxes, hand=tuple(names[1:5]), next_card=names[0],
                ready=tuple(state.ready), elixir=int(numbers.elixir.number),
                towers=towers, tower_boxes=tower_boxes, screen=state.screen.name)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' if count > 1 else ''}"


IN_GAME = "in_game"


def summary(seen: Seen) -> str:
    if not seen.boxes and not seen.hand:
        return "Aucune détection (pas en match, ou image illisible)."
    if seen.screen != IN_GAME:
        # The card reader runs on every frame, but outside a battle the hand
        # strip holds menus, and what it reports there is noise.
        return f"Écran : {seen.screen} (pas en match)"
    hand = ", ".join(f"{name} {'✓' if slot in seen.ready else '✗'}"
                     for slot, name in enumerate(seen.hand))
    allies = sum(box.ally for box in seen.boxes)
    enemies = len(seen.boxes) - allies
    strongest = sorted(seen.boxes, key=lambda box: -box.conf)[:6]
    units = ", ".join(f"{box.label} {box.conf * 100:.0f} %" for box in strongest)
    towers = seen.towers
    return (f"Écran : {seen.screen} · Élixir {seen.elixir} · Main : {hand} "
            f"(suivante : {seen.next_card})\n"
            f"Tours ennemies {percent(towers['enemy_left'])} / {percent(towers['enemy_right'])}, "
            f"alliées {percent(towers['ally_left'])} / {percent(towers['ally_right'])}\n"
            f"{_plural(allies, 'allié')}, {_plural(enemies, 'ennemi')} · {units or '—'}")


def percent(fraction: float) -> str:
    """BuildABot reports princess bars as a fraction of full health."""
    return f"{fraction * 100:.0f} %"


class BotVision:
    """BuildABot's Detector for one deck, built on first use, one frame at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._deck_key: Optional[str] = None
        self._detector = None

    def _detector_for(self, deck_key: str):
        if self._detector is None or self._deck_key != deck_key:
            for path in (ROOT / "scripts", ROOT / "vendor" / "ClashRoyaleBuildABot"):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            from clashroyalebuildabot.detectors.detector import Detector
            if deck_key == "hog26":
                import mumu_overnight_bot
                cards = list(mumu_overnight_bot.DECK)
            else:
                import mumu_buildabot_native
                cards = [action.CARD for action in mumu_buildabot_native.DECK_PRESETS[deck_key]]
            self._detector = Detector(cards)
            self._deck_key = deck_key
        return self._detector

    def analyse(self, png: bytes, deck_key: str):
        """(annotated QImage, Seen) for one screenshot."""
        from PIL import Image
        frame = Image.open(io.BytesIO(png)).convert("RGB")
        with self._lock:
            state = self._detector_for(deck_key).run(frame.resize((DETECTOR_W, DETECTOR_H)))
        seen = seen_from_state(state, frame.size)
        return draw(png, seen), seen


def draw(png: bytes, seen: Seen):
    """The screenshot with the detections painted on (QImage, safe off the UI thread)."""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen

    image = QImage.fromData(png, "PNG").convertToFormat(QImage.Format.Format_RGB32)
    scale = image.width() / 1080
    painter = QPainter(image)
    font = QFont("Segoe UI", max(7, round(20 * scale)))
    font.setBold(True)
    painter.setFont(font)
    metrics = painter.fontMetrics()

    def chip(x: float, y: float, text: str, colour: QColor) -> None:
        width, height = metrics.horizontalAdvance(text) + 10 * scale, metrics.height()
        rect = QRectF(x, max(0.0, y - height), width, height)
        painter.fillRect(rect, colour)
        painter.setPen(QColor("white"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    for box in seen.boxes:
        colour = QColor("#58a6ff") if box.ally else QColor("#f85149")
        painter.setPen(QPen(colour, max(2, round(4 * scale))))
        left, top, right, bottom = box.bbox
        painter.drawRect(QRectF(left, top, right - left, bottom - top))
        chip(left, top, f"{box.label} {box.conf * 100:.0f}%", colour)

    if seen.screen != IN_GAME:
        painter.end()
        return image

    for key, hp in seen.towers.items():
        left, top, _, _ = seen.tower_boxes[key]
        chip(left, top, f"PV {percent(hp)}", QColor(0, 0, 0, 170))

    if seen.hand:
        from clashroyalebuildabot.constants import CARD_CONFIG
        for slot, name in enumerate(seen.hand):
            left, top, right, bottom = scale_bbox(CARD_CONFIG[slot + 1], (image.width(), image.height()))
            ready = slot in seen.ready
            colour = QColor("#3fb950") if ready else QColor("#8b949e")
            painter.setPen(QPen(colour, max(2, round(4 * scale))))
            painter.drawRect(QRectF(left, top, right - left, bottom - top))
            chip(left, top, name, colour)
        chip(20 * scale, image.height() - 140 * scale,
             f"Élixir {seen.elixir}", QColor("#a371f7"))
    painter.end()
    return image
