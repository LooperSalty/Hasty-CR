"""Clean arena backgrounds from recorded battle frames, by temporal median.

Units move and towers do not: the per-pixel median of a match's first minute
is the arena with its towers and no troops - the canvas the synthetic
detector data is painted on, in this account's own arena and framing.

    python scripts/arena_backgrounds.py tmp/frames/live tmp/yolo_synth/backgrounds

Frames are grouped into matches by time (a gap over GAP_SECONDS starts a new
one); only in-game frames (BuildABot's screen detector) from the first
FIRST_SECONDS of each match are used, before a tower can fall - and, with
--late, a second background from its last LATE_SECONDS, when towers often
have fallen: ruins the detector must learn are not a unit (it read them as
a Goblin Hut).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

WIDTH, HEIGHT = 1080, 1920
GAP_SECONDS = 20
FIRST_SECONDS = 70
LATE_SECONDS = 40
MIN_FRAMES = 20


def frame_time(path: Path) -> datetime:
    return datetime.strptime(path.stem[:15], "%Y%m%d-%H%M%S")


def in_game_frames(paths):
    from clashroyalebuildabot.detectors.screen_detector import ScreenDetector
    detector = ScreenDetector()
    for path in paths:
        image = Image.open(path).convert("RGB")
        if detector.run(image.resize((720, 1280))).name == "in_game":
            yield path, image


def matches(frames):
    """Lists of (time, image), one per match, in order."""
    current, last = [], None
    for path, image in frames:
        when = frame_time(path)
        if last is not None and (when - last).total_seconds() > GAP_SECONDS and current:
            yield current
            current = []
        current.append((when, image))
        last = when
    if current:
        yield current


def median_background(match, late: bool = False) -> Image.Image:
    if late:
        end = match[-1][0]
        chosen = [image for when, image in match if (end - when).total_seconds() <= LATE_SECONDS]
    else:
        start = match[0][0]
        chosen = [image for when, image in match
                  if (when - start).total_seconds() <= FIRST_SECONDS]
    stack = np.stack([np.asarray(image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR))
                      for image in chosen])
    return Image.fromarray(np.median(stack, axis=0).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--late", action="store_true",
                        help="also a background from each match's last seconds")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.frames.glob("*.jpg"))
    made = 0
    for match in matches(in_game_frames(paths)):
        if len(match) < MIN_FRAMES:
            continue
        name = f"arena_{match[0][0]:%Y%m%d_%H%M%S}.png"
        median_background(match).save(args.out / name)
        made += 1
        if args.late:
            median_background(match, late=True).save(args.out / name.replace(".png", "_late.png"))
            made += 1
        print(f"{name}: {len(match)} in-game frames")
    print(f"{made} backgrounds in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
