"""A checked set of live frames from our own arena, to measure unit vision on.

Every score the detectors had was measured on synthetic frames or on frames
from another phone; the only real check was 11 frames with names and no boxes.
This builds the missing reference from frames the runner recorded
(tmp/dataset/live, 1080x1920, one a second):

    python scripts/truth_set.py select OUT --day 20261106 --frames 48

writes, for each chosen frame, OUT/<id>.jpg (the arena, cropped), OUT/<id>_pre.jpg
(the same with every unit our YOLO and BuildABot's tracker saw, numbered) and
OUT/<id>_pre.json (those detections). A checker - a person or an agent -
writes OUT/<id>_truth.json:

    {"units": [{"name": "knight", "side": "ally", "box": [x1, y1, x2, y2]}, ...]}

names in BuildABot's spelling, sides "ally" (ours, blue, bottom) or "enemy",
boxes in the 1080x1920 frame. Scoring is scripts/eval_truth.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "vendor" / "ClashRoyaleBuildABot"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

LIVE = ROOT / "tmp" / "dataset" / "live"
ARENA_CROP = (84, 161, 996, 1487)      # the playing field in a 1080x1920 frame
SKIP_START, SKIP_END = 8.0, 6.0        # seconds: deploy banners, end screens
MIN_TRACKS = 3
PER_MATCH = 3
YOLO_CONF = 0.30
COLOURS = {"yolo": (255, 220, 0), "buildabot": (0, 220, 255)}


def candidates(day: str, rng: random.Random) -> list[tuple[Path, Path]]:
    """(jpg, sidecar) of busy mid-match frames, a few per match, shuffled."""
    chosen = []
    for match in sorted(LIVE.glob(f"match_{day}_*")):
        frames = sorted(match.glob("*.json"))
        if len(frames) < 60:
            continue
        end = float(frames[-1].stem.replace("_", "."))
        busy = []
        for sidecar in frames:
            t = float(sidecar.stem.replace("_", "."))
            if not SKIP_START <= t <= end - SKIP_END:
                continue
            tracks = json.loads(sidecar.read_text(encoding="utf-8")).get("tracks", [])
            if len(tracks) >= MIN_TRACKS and sidecar.with_suffix(".jpg").exists():
                busy.append((sidecar.with_suffix(".jpg"), sidecar))
        chosen += rng.sample(busy, min(PER_MATCH, len(busy)))
    rng.shuffle(chosen)
    return chosen


def _draw(image: Image.Image, detections: list, offset: tuple, source: str) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = (det["box"][0] - offset[0], det["box"][1] - offset[1],
                          det["box"][2] - offset[0], det["box"][3] - offset[1])
        colour = COLOURS[source]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=3 if source == "yolo" else 1)
        tag = f"{det['n']}{'Y' if source == 'yolo' else 'B'} {det['name']} {det['side'][0].upper()}"
        draw.text((x1 + 2, y1 - 12 if source == "yolo" else y2 + 2), tag, fill=colour, font=font)


def select(out: Path, day: str, count: int, seed: int) -> None:
    from brain.vision import RUNS, YoloVision

    out.mkdir(parents=True, exist_ok=True)
    vision = YoloVision(weights=RUNS / "cr_detector_y26" / "weights" / "best.pt",
                        confidence=YOLO_CONF)
    if not vision.ready:
        raise SystemExit(f"YOLO not ready: {vision.error}")
    rng = random.Random(seed)
    picked = candidates(day, rng)[:count]
    for index, (jpg, sidecar) in enumerate(picked):
        frame = Image.open(jpg).convert("RGB")
        ours, theirs = vision.detect(frame)
        yolo = [{"n": i, "name": s.name, "side": "ally" if s.side == 0 else "enemy",
                 "box": [round(v) for v in s.box], "conf": round(s.confidence, 2)}
                for i, s in enumerate(ours + theirs, 1)]
        tracks = json.loads(sidecar.read_text(encoding="utf-8")).get("tracks", [])
        buildabot = [{"n": i, "name": t["name"], "side": t["side"], "box": t["box"]}
                     for i, t in enumerate(tracks, 1) if t.get("box")]
        stem = f"f{index:02d}"
        crop = frame.crop(ARENA_CROP)
        crop.save(out / f"{stem}.jpg", quality=92)
        annotated = crop.copy()
        _draw(annotated, buildabot, ARENA_CROP, "buildabot")
        _draw(annotated, yolo, ARENA_CROP, "yolo")
        annotated.save(out / f"{stem}_pre.jpg", quality=92)
        (out / f"{stem}_pre.json").write_text(json.dumps(
            {"frame": str(jpg.relative_to(ROOT)), "crop": list(ARENA_CROP),
             "yolo": yolo, "buildabot": buildabot}, indent=1), encoding="utf-8")
    print(f"{len(picked)} frames -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select")
    pick.add_argument("out", type=Path)
    pick.add_argument("--day", default="2026110")
    pick.add_argument("--frames", type=int, default=48)
    pick.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    select(args.out, args.day, args.frames, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
