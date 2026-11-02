"""Grow the detector's dataset from recorded battles, without a human.

scripts/dataset_recorder.py keeps a frame a second with BuildABot's tracked
units and our recent plays. This writes YOLO labels for the frames where
every unit is explained safely:

* **own**: a unit of ours that appeared where we played its card moments
  before, under that card's name - certain, whatever either model thinks;
* **agree**: BuildABot's tracked unit (seen at least MIN_HITS frames) and the
  YOLO detector name the same unit at the same place.

Anything else - a disagreement, a unit one model sees and the other does not,
a track too new to trust - is erased: painted over with the same match's
empty arena (its per-pixel median), since an unlabelled troop would teach the
detector that troops are background. When erasing would cut into a labelled
unit, or a card was just dropped and nothing shows it yet (the deploy
animation), the frame is not used. Faint detections (below UNRESOLVED_CONF)
do not count either way.

Frames are written in the training framing (ARENA_CROP of the 1080x1920 frame
at 568x896, as scripts/synth_dataset.py). The last `--holdout` matches are
kept apart as the control set, never trained on.

    python scripts/autolabel.py tmp/dataset/live --holdout 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

from unit_tracker import CARD_UNITS  # noqa: E402

ARENA_CROP = (84, 161, 996, 1487)       # as scripts/synth_dataset.py
CROP_W, CROP_H = 568, 896
OWN_RADIUS = 4.0                         # tiles from where the card was dropped...
OWN_WALK = 1.2                           # ...plus how far it can have walked since
OWN_MAX_AGE = 10.0
MIN_HITS = 3
MATCH_IOU = 0.3
# Anything a detector reports at this or above must be explained (labelled or
# erased): a faint troop left as it is would be taught as background.
UNRESOLVED_CONF = 0.3
EMPTY_KEEP = 0.15                        # share of empty frames kept as negatives
DEPLOYING_SECONDS = 2.5                  # a play this fresh must already be labelled
ERASE_MARGIN = 0.15                      # erase a little around an ambiguous box
OUT = ROOT / "tmp" / "yolo_real"


@dataclass(frozen=True)
class Det:
    name: str
    box: tuple
    conf: float


@dataclass(frozen=True)
class Label:
    name: str
    box: tuple
    source: str                          # "own" or "agree"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _own(track: dict, plays: list) -> bool:
    if track.get("side") != "ally":
        return False
    for play in plays:
        units = CARD_UNITS.get(play["card"], (play["card"],))
        if track["name"] not in units or play["age"] > OWN_MAX_AGE:
            continue
        reach = OWN_RADIUS + OWN_WALK * play["age"]
        if math.hypot(track["x"] - play["tile"][0], track["y"] - play["tile"][1]) <= reach:
            return True
    return False


def label_frame(tracks: list, yolo: list, plays: list) -> tuple[list, list]:
    """(labels, unresolved Dets) for one frame. An unresolved Det with no box
    (a card being dropped) cannot be erased: the frame is then unusable."""
    labels, unresolved, used = [], [], set()
    for track in tracks:
        if not track.get("box"):
            continue
        box = tuple(track["box"])
        overlaps = sorted(((iou(det.box, box), i) for i, det in enumerate(yolo)
                           if i not in used), reverse=True)
        best = overlaps[0] if overlaps and overlaps[0][0] >= MATCH_IOU else None
        if _own(track, plays):
            labels.append(Label(track["name"], box, "own"))
            used.update(i for score, i in overlaps if score >= MATCH_IOU)
            continue
        if best is None:
            unresolved.append(Det(track["name"], box, track.get("conf", 0.0)))
            continue
        used.add(best[1])
        det = yolo[best[1]]
        if det.name == track["name"] and track.get("hits", 0) >= MIN_HITS:
            labels.append(Label(track["name"], det.box, "agree"))
        else:
            unresolved.append(Det(track["name"], box, track.get("conf", 0.0)))
    unresolved += [det for i, det in enumerate(yolo)
                   if i not in used and det.conf >= UNRESOLVED_CONF]
    own = {label.name for label in labels if label.source == "own"}
    for play in plays:
        units = CARD_UNITS.get(play["card"], (play["card"],))
        if play["age"] <= DEPLOYING_SECONDS and not own.intersection(units):
            unresolved.append(Det(play["card"], None, 1.0))
    return labels, unresolved


def _grow(box, margin: float) -> tuple:
    dx, dy = (box[2] - box[0]) * margin, (box[3] - box[1]) * margin
    return (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)


def erase(frame, background, unresolved: list, labels: list):
    """`frame` with each unresolved unit painted over by the empty arena, or
    None when that is impossible (no box, or it would cut a labelled unit)."""
    out = frame.copy()
    for det in unresolved:
        if det.box is None:
            return None
        box = _grow(det.box, ERASE_MARGIN)
        if any(iou(box, label.box) > 0 for label in labels):
            return None
        region = tuple(int(round(v)) for v in (max(0, box[0]), max(0, box[1]),
                                               min(frame.width, box[2]),
                                               min(frame.height, box[3])))
        if region[2] > region[0] and region[3] > region[1]:
            out.paste(background.crop(region), region[:2])
    return out


def to_crop(box) -> tuple:
    """A 1080x1920 frame box in the 568x896 training framing."""
    x0, y0, x1, y1 = ARENA_CROP
    sx, sy = CROP_W / (x1 - x0), CROP_H / (y1 - y0)
    return ((box[0] - x0) * sx, (box[1] - y0) * sy, (box[2] - x0) * sx, (box[3] - y0) * sy)


def class_index() -> dict:
    """BuildABot unit name -> class index of the detector's label list."""
    import yaml
    from brain.vision import CLASS_ALIASES, _extra_units, _key
    from clashroyalebuildabot.namespaces.units import NAME2UNIT
    names = yaml.safe_load((ROOT / "tmp" / "yolo" / "cr.yaml").read_text(encoding="utf-8"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    units = dict(NAME2UNIT)
    units.update(_extra_units())
    by_key = {_key(unit.name): unit for unit in units.values()}
    index = {}
    for i, raw in enumerate(names):
        key = _key(raw)
        unit = by_key.get(CLASS_ALIASES.get(key, key))
        if unit is not None:
            index.setdefault(unit.name, i)
    return index


def yolo_dets(vision, frame) -> list:
    ours, theirs = vision.detect(frame)
    return [Det(s.name, tuple(float(v) for v in s.box), float(s.confidence))
            for s in ours + theirs]


def _write(frame, labels, index, images: Path, labels_dir: Path, stem: str) -> int:
    from PIL import Image
    crop = frame.crop(ARENA_CROP).resize((CROP_W, CROP_H), Image.Resampling.BILINEAR)
    lines = []
    for label in labels:
        cls = index.get(label.name)
        if cls is None:
            continue
        x0, y0, x1, y1 = (max(0.0, v) for v in to_crop(label.box))
        x1, y1 = min(CROP_W, x1), min(CROP_H, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        lines.append(f"{cls} {(x0 + x1) / 2 / CROP_W:.6f} {(y0 + y1) / 2 / CROP_H:.6f} "
                     f"{(x1 - x0) / CROP_W:.6f} {(y1 - y0) / CROP_H:.6f}\n")
    crop.save(images / f"{stem}.jpg", quality=92)
    (labels_dir / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")
    return len(lines)


def match_background(match: Path):
    """The match's empty arena: per-pixel median of its first frames."""
    import numpy as np
    from PIL import Image
    frames = sorted(match.glob("*.jpg"))[:70]
    if len(frames) < 15:
        return None
    stack = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in frames])
    return Image.fromarray(np.median(stack, axis=0).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recorded", type=Path)
    parser.add_argument("--holdout", type=int, default=1,
                        help="last matches kept apart as the control set")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()

    from PIL import Image
    from brain.vision import YoloVision
    vision = YoloVision(weights=args.weights, confidence=0.3)
    index = class_index()
    matches = sorted(p for p in args.recorded.iterdir() if p.is_dir())
    train, control = matches[:-args.holdout or None], matches[len(matches) - args.holdout:]
    images, labels_dir = OUT / "images" / "train", OUT / "labels" / "train"
    for folder in (images, labels_dir):
        folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    stats = {"frames": 0, "kept": 0, "empty": 0, "own": 0, "agree": 0, "rejected": 0,
             "erased": 0}
    for match in train:
        background = match_background(match)
        for sidecar in sorted(match.glob("*.json")):
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            frame = Image.open(sidecar.with_suffix(".jpg")).convert("RGB")
            stats["frames"] += 1
            labels, unresolved = label_frame(data["tracks"], yolo_dets(vision, frame),
                                             data["plays"])
            if unresolved:
                frame = erase(frame, background, unresolved, labels) if background else None
                if frame is None:
                    stats["rejected"] += 1
                    continue
                stats["erased"] += len(unresolved)
            if not labels and rng.random() > EMPTY_KEEP:
                continue
            stem = f"{match.name}_{sidecar.stem}"
            written = _write(frame, labels, index, images, labels_dir, stem)
            stats["kept"] += 1
            stats["empty"] += written == 0
            for label in labels:
                stats[label.source] += 1
    control_dir = OUT / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    for match in control:
        shutil.copytree(match, control_dir / match.name, dirs_exist_ok=True)
    print(json.dumps(stats), f"train matches {len(train)}, control matches {len(control)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
