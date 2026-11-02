"""Side-by-side on real battle frames: our YOLO detector vs BuildABot's model.

No labels exist for live frames, so this is a visual check: each frame is
drawn twice, boxes and names from each detector, and the counts printed. The
point is the cases that cost matches - a troop missed, a Giant named a Hog.

    python scripts/compare_detectors.py tmp/frames/live out.jpg --frames 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

from PIL import Image, ImageDraw  # noqa: E402

TILE = (360, 640)


def draw(frame: Image.Image, units, label: str) -> Image.Image:
    """`units`: (name, side, (x0, y0, x1, y1) in 1080x1920 pixels)."""
    canvas = frame.copy()
    pen = ImageDraw.Draw(canvas)
    for name, side, (x0, y0, x1, y1) in units:
        colour = "#3399ff" if side == "ally" else "#ff3333"
        pen.rectangle((x0, y0, x1, y1), outline=colour, width=5)
        pen.text((x0, max(0, y0 - 26)), name, fill=colour)
    pen.text((20, 1860), label, fill="white")
    return canvas.resize(TILE)


def yolo_units(vision, frame):
    ours, theirs = vision.detect(frame)
    return ([(s.name, "ally", s.box) for s in ours]
            + [(s.name, "enemy", s.box) for s in theirs])


def buildabot_units(detector, frame):
    state = detector.run(frame.resize((368, 652)))
    scale = 1080 / 368
    out = []
    for side, group in (("ally", state.allies), ("enemy", state.enemies)):
        for det in group:
            if not 0 <= det.position.tile_x <= 17:
                continue                                  # scenery read as a unit
            box = tuple(round(v * scale) for v in det.position.bbox)
            out.append((det.unit.name, side, box))
    return out, state.screen.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", dest="count", type=int, default=8)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO confidence floor")
    args = parser.parse_args()

    import mumu_buildabot_native as native
    from brain.vision import YoloVision
    native.use_deck("starter")
    detector = native.Detector(native.DECK)
    vision = YoloVision(weights=args.weights, confidence=args.conf)
    print(f"YOLO weights: {vision.weights} framing: {vision.framing}")

    battle = []
    for path in sorted(args.frames.glob("*.jpg")):
        frame = Image.open(path).convert("RGB").resize((1080, 1920))
        units, screen = buildabot_units(detector, frame)
        if screen == "in_game":
            battle.append((path, frame, units))
    step = max(1, len(battle) // args.count)
    rows = []
    for path, frame, upstream in battle[::step][:args.count]:
        mine = yolo_units(vision, frame)
        print(f"{path.stem}: YOLO {len(mine)} {sorted(n for n, _, _ in mine)} | "
              f"BuildABot {len(upstream)} {sorted(n for n, _, _ in upstream)}")
        rows.append((draw(frame, mine, "YOLO"), draw(frame, upstream, "BuildABot")))
    sheet = Image.new("RGB", (TILE[0] * 2 * min(4, len(rows)),
                              TILE[1] * ((len(rows) + 3) // 4)))
    for i, (left, right) in enumerate(rows):
        x, y = (i % 4) * TILE[0] * 2, (i // 4) * TILE[1]
        sheet.paste(left, (x, y))
        sheet.paste(right, (x + TILE[0], y))
    sheet.save(args.out, quality=85)
    print(f"sheet: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
