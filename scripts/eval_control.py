"""Score detectors on the control set: real frames, units listed by eye.

The control match (scripts/autolabel.py --holdout) is never trained on. For
some of its frames the units on the board are listed by hand in
`truth.json` ({frame stem: [unit names]}); each detector is then scored on
names, frame by frame:

* precision: of the names it reported, how many are really there;
* recall: of the units really there, how many it named.

    python scripts/eval_control.py sheet tmp/yolo_real/control/match_x   # frames to list
    python scripts/eval_control.py score tmp/yolo_real/control/match_x   # needs truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

ARENA = (40, 120, 1040, 1540)            # the board, for the sheets


def name_scores(predicted: list, truth: list) -> tuple[int, int, int]:
    """(matched, predicted, true) counts for one frame, by unit name."""
    hits = sum((Counter(predicted) & Counter(truth)).values())
    return hits, len(predicted), len(truth)


def chosen_frames(match: Path, count: int) -> list:
    frames = sorted(match.glob("*.jpg"))
    step = max(1, len(frames) // count)
    return frames[step // 2::step][:count]


def sheet(match: Path, count: int) -> None:
    from PIL import Image, ImageDraw
    frames = chosen_frames(match, count)
    for page in range(0, len(frames), 4):
        group = frames[page:page + 4]
        canvas = Image.new("RGB", (500 * len(group), 710), "white")
        for i, path in enumerate(group):
            board = Image.open(path).convert("RGB").crop(ARENA).resize((500, 710))
            ImageDraw.Draw(board).text((8, 8), path.stem, fill="yellow")
            canvas.paste(board, (i * 500, 0))
        out = match / f"control_sheet_{page // 4 + 1}.jpg"
        canvas.save(out, quality=92)
        print(out, [p.stem for p in group])


def score(match: Path, weights) -> None:
    from PIL import Image
    from brain.vision import YoloVision
    truth = json.loads((match / "truth.json").read_text(encoding="utf-8"))
    vision = YoloVision(weights=weights)
    totals = {"yolo": [0, 0, 0], "buildabot": [0, 0, 0]}
    for stem, units in truth.items():
        frame = Image.open(match / f"{stem}.jpg").convert("RGB")
        sidecar = json.loads((match / f"{stem}.json").read_text(encoding="utf-8"))
        ours, theirs = vision.detect(frame)
        predictions = {"yolo": [s.name for s in ours + theirs],
                       "buildabot": [t["name"] for t in sidecar["tracks"]]}
        for name, predicted in predictions.items():
            for i, value in enumerate(name_scores(predicted, units)):
                totals[name][i] += value
    for name, (hits, predicted, true) in totals.items():
        print(f"{name:9s} precision {hits / max(1, predicted):.0%} ({hits}/{predicted})  "
              f"recall {hits / max(1, true):.0%} ({hits}/{true})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("sheet", "score"))
    parser.add_argument("match", type=Path)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "sheet":
        sheet(args.match, args.frames)
    else:
        score(args.match, args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
