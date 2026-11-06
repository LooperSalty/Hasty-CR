"""Score unit detectors against the checked frames of scripts/truth_set.py.

    python scripts/eval_truth.py tmp/truth/set1                    # the recorded guesses
    python scripts/eval_truth.py tmp/truth/set1 --weights RUN/weights/best.pt --conf 0.4

Each true unit is paired with the nearest guess (box centres within
max(MIN_MATCH_PX, MATCH_SHARE x the unit's size)), one guess per unit. Then:

* found      true units with a guess at all, and guesses with a true unit (ghosts are the rest);
* name, side among the pairs, how often each is right;
* exact      pairs right on both, over true units (recall) and over guesses (precision).

Confusions (true name -> guessed name), ghosts and misses are listed by name,
which says what to train on.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "vendor" / "ClashRoyaleBuildABot"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MIN_MATCH_PX = 35
MATCH_SHARE = 0.6


def _centre(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def pair(truth: list, guesses: list) -> list[tuple[int, int]]:
    """(truth index, guess index) pairs, nearest first, each used once."""
    options = []
    for ti, unit in enumerate(truth):
        size = max(unit["box"][2] - unit["box"][0], unit["box"][3] - unit["box"][1])
        reach = max(MIN_MATCH_PX, MATCH_SHARE * size)
        tx, ty = _centre(unit["box"])
        for gi, guess in enumerate(guesses):
            gx, gy = _centre(guess["box"])
            distance = math.hypot(tx - gx, ty - gy)
            if distance <= reach:
                options.append((distance, ti, gi))
    used_t, used_g, pairs = set(), set(), []
    for _d, ti, gi in sorted(options):
        if ti not in used_t and gi not in used_g:
            used_t.add(ti)
            used_g.add(gi)
            pairs.append((ti, gi))
    return pairs


def score(frames: list[tuple[list, list]]) -> dict:
    totals = Counter()
    confusions, ghosts, misses, side_errors = Counter(), Counter(), Counter(), Counter()
    for truth, guesses in frames:
        pairs = pair(truth, guesses)
        totals["truth"] += len(truth)
        totals["guesses"] += len(guesses)
        totals["paired"] += len(pairs)
        for ti, gi in pairs:
            t, g = truth[ti], guesses[gi]
            name_ok, side_ok = t["name"] == g["name"], t["side"] == g["side"]
            totals["name_ok"] += name_ok
            totals["side_ok"] += side_ok
            totals["exact"] += name_ok and side_ok
            if not name_ok:
                confusions[(t["name"], g["name"])] += 1
            if not side_ok:
                side_errors[f"{t['name']} {t['side']}->{g['side']}"] += 1
        paired_t = {ti for ti, _ in pairs}
        paired_g = {gi for _, gi in pairs}
        misses.update(t["name"] for i, t in enumerate(truth) if i not in paired_t)
        ghosts.update(g["name"] for i, g in enumerate(guesses) if i not in paired_g)

    def share(a, b):
        return round(100.0 * totals[a] / totals[b], 1) if totals[b] else 0.0

    return {"truth units": totals["truth"], "guesses": totals["guesses"],
            "found %": share("paired", "truth"), "not ghosts %": share("paired", "guesses"),
            "name right %": share("name_ok", "paired"), "side right %": share("side_ok", "paired"),
            "exact recall %": share("exact", "truth"), "exact precision %": share("exact", "guesses"),
            "confusions": confusions.most_common(12), "side errors": side_errors.most_common(8),
            "ghosts": ghosts.most_common(10), "misses": misses.most_common(10)}


def _recorded(folder: Path, source: str, conf: float) -> list[tuple[list, list]]:
    frames = []
    for truth_file in sorted(folder.glob("f*_truth.json")):
        truth = json.loads(truth_file.read_text(encoding="utf-8"))["units"]
        pre = json.loads((folder / truth_file.name.replace("_truth", "_pre")).read_text(encoding="utf-8"))
        guesses = [g for g in pre[source] if g.get("conf", 1.0) >= conf]
        frames.append((truth, guesses))
    return frames


def _fresh(folder: Path, weights: Path, conf: float, side_model=None) -> list[tuple[list, list]]:
    from PIL import Image
    from brain.vision import YoloVision

    vision = YoloVision(weights=weights, confidence=conf, side_model=side_model)
    if not vision.ready:
        raise SystemExit(f"YOLO not ready: {vision.error}")
    frames = []
    for truth_file in sorted(folder.glob("f*_truth.json")):
        truth = json.loads(truth_file.read_text(encoding="utf-8"))["units"]
        pre = json.loads((folder / truth_file.name.replace("_truth", "_pre")).read_text(encoding="utf-8"))
        ours, theirs = vision.detect(Image.open(ROOT / pre["frame"]).convert("RGB"))
        guesses = [{"name": s.name, "side": "ally" if s.side == 0 else "enemy", "box": list(s.box)}
                   for s in ours + theirs]
        frames.append((truth, guesses))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--weights", type=Path, help="score this YOLO checkpoint, run fresh")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--side-model", type=Path, help="side classifier checkpoint for --weights")
    args = parser.parse_args()
    if args.weights:
        side = f" side {args.side_model.stem}" if args.side_model else ""
        results = {f"yolo {args.weights.parent.parent.name} @{args.conf}{side}":
                   score(_fresh(args.folder, args.weights, args.conf, args.side_model))}
    else:
        results = {"buildabot (tracker, live)": score(_recorded(args.folder, "buildabot", 0.0)),
                   f"yolo y26 @{args.conf}": score(_recorded(args.folder, "yolo", args.conf))}
    for name, result in results.items():
        print(f"== {name}")
        for key, value in result.items():
            print(f"  {key:18s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
