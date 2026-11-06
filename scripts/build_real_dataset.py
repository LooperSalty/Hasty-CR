"""Detector training frames from our own matches, labelled for recall.

On the checked frames (scripts/eval_truth.py) the YOLO detector names what it
sees almost perfectly (96%) and invents nearly nothing (2% ghosts), but misses
one unit in five - small units in groups above all (Archers, Minions,
Skeletons). scripts/autolabel.py kept only what both detectors agreed on,
which keeps those misses. This takes the union, with each source where it is
reliable:

* YOLO boxes at YOLO_SURE or above, as they are;
* BuildABot tracks the recording followed for TRACK_HITS frames where YOLO
  has no box at all - its misses - away from the towers, where BuildABot's
  own ghosts live (tower parts read as Bandit, Sparky, Mini P.E.K.K.A);
* where YOLO saw the unit only faintly, BuildABot's track confirms it when
  both say the same name;
* a faint YOLO box of a card we played lately, within walking reach of where
  it landed, is that unit: half of YOLO's misses on the checked frames were
  our own troops (Mini P.E.K.K.A, Archers, Minions, P.E.K.K.A).

Frames are cut and scaled to the detector's framing (568x896). The checked
frames become the validation split, so training is watched on real labels.

    python scripts/build_real_dataset.py --per-match 15
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "vendor" / "ClashRoyaleBuildABot"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

LIVE = ROOT / "tmp" / "dataset" / "live"
TRUTH = ROOT / "tmp" / "truth" / "set1"
OUT = ROOT / "tmp" / "yolo_ours"
WEIGHTS = ROOT / "tmp" / "yolo" / "runs" / "cr_detector_y26" / "weights" / "best.pt"
ARENA_CROP = (84, 161, 996, 1487)
FRAMING = (568, 896)
YOLO_SURE = 0.35
YOLO_FAINT = 0.10
TRACK_HITS = 5
OVERLAP_IOU = 0.2
# Towers in BuildABot tiles; BuildABot's ghosts sit on them.
TOWERS = ((3.5, 6.5), (14.5, 6.5), (9.5, 2.5), (3.5, 24.5), (14.5, 24.5), (9.5, 28.5))
TOWER_TILES = 2.2
MIN_TRACKS = 2
PLAY_SECONDS = 10.0             # a unit of ours is still near its drop this long after...
PLAY_TILES = 3.0                # ...within this of the tile, plus a walk
WALK_TILES_PER_SECOND = 1.2
SKIP_START, SKIP_END = 6.0, 5.0


def class_ids() -> tuple[dict, dict]:
    import yaml
    names = yaml.safe_load((ROOT / "tmp" / "yolo" / "cr.yaml").read_text(encoding="utf-8"))["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    from brain.vision import _key
    return {int(i): n for i, n in names.items()}, {_key(n): int(i) for i, n in names.items()}


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def to_crop(box) -> list:
    """A 1080x1920 frame box in the 568x896 framing."""
    sx = FRAMING[0] / (ARENA_CROP[2] - ARENA_CROP[0])
    sy = FRAMING[1] / (ARENA_CROP[3] - ARENA_CROP[1])
    return [(box[0] - ARENA_CROP[0]) * sx, (box[1] - ARENA_CROP[1]) * sy,
            (box[2] - ARENA_CROP[0]) * sx, (box[3] - ARENA_CROP[1]) * sy]


def frame_crop(path: Path) -> np.ndarray:
    import cv2
    pixels = np.asarray(Image.open(path).convert("RGB"))
    x0, y0, x1, y1 = ARENA_CROP
    return cv2.resize(pixels[y0:y1, x0:x1], FRAMING, interpolation=cv2.INTER_AREA)


def yolo_line(cls: int, box) -> str:
    x1, y1, x2, y2 = (max(0.0, box[0]), max(0.0, box[1]), min(FRAMING[0], box[2]), min(FRAMING[1], box[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return ""
    return (f"{cls} {(x1 + x2) / 2 / FRAMING[0]:.6f} {(y1 + y2) / 2 / FRAMING[1]:.6f} "
            f"{(x2 - x1) / FRAMING[0]:.6f} {(y2 - y1) / FRAMING[1]:.6f}\n")


def near_tower(x: float, y: float) -> bool:
    return any(math.hypot(x - tx, y - ty) < TOWER_TILES for tx, ty in TOWERS)


def _tile_of(box) -> tuple[float, float]:
    """BuildABot tile of a 568x896 box's ground point (bottom centre)."""
    from brain.vision import YoloVision
    sx = (ARENA_CROP[2] - ARENA_CROP[0]) / FRAMING[0]
    sy = (ARENA_CROP[3] - ARENA_CROP[1]) / FRAMING[1]
    px = ARENA_CROP[0] + (box[0] + box[2]) / 2 * sx
    py = ARENA_CROP[1] + box[3] * sy
    return YoloVision._to_tile(px, py, 1080, 1920)


def _played_here(cls: int, box, plays: list, by_key: dict) -> bool:
    """A card we played lately brings this class and could have walked here."""
    from brain.vision import _key
    from unit_tracker import CARD_UNITS
    x, y = _tile_of(box)
    for play in plays:
        if not 0.0 <= play["age"] <= PLAY_SECONDS:
            continue
        units = CARD_UNITS.get(play["card"], (play["card"],))
        if cls not in {by_key.get(_key(u)) for u in units}:
            continue
        reach = PLAY_TILES + WALK_TILES_PER_SECOND * play["age"]
        if math.hypot(play["tile"][0] - x, play["tile"][1] - y) <= reach:
            return True
    return False


def labels_for(model, pixels: np.ndarray, tracks: list, plays: list, by_key: dict,
               keep_class) -> tuple[str, dict]:
    from brain.vision import _key
    result = model.predict(pixels[:, :, ::-1], imgsz=896, conf=YOLO_FAINT, device="0", verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    classes = [int(c) for c in result.boxes.cls.cpu().numpy()]
    confs = result.boxes.conf.cpu().numpy().tolist()
    lines, counts = [], {"yolo": 0, "confirmed": 0, "buildabot": 0, "played": 0}
    faint = []
    for box, cls, conf in zip(boxes, classes, confs):
        if not keep_class(cls):
            continue
        if conf >= YOLO_SURE:
            lines.append(yolo_line(cls, box))
            counts["yolo"] += 1
        elif _played_here(cls, box, plays, by_key):
            lines.append(yolo_line(cls, box))
            counts["played"] += 1
        else:
            faint.append((box, cls))
    for track in tracks:
        if track.get("hits", 0) < TRACK_HITS or not track.get("box") or near_tower(track["x"], track["y"]):
            continue
        cls = by_key.get(_key(track["name"]))
        if cls is None or not keep_class(cls):
            continue
        box = to_crop(track["box"])
        sure_overlap = any(_iou(box, b) >= OVERLAP_IOU for b, c, s in zip(boxes, classes, confs)
                           if s >= YOLO_SURE)
        if sure_overlap:
            continue
        faint_hits = [(b, c) for b, c in faint if _iou(box, b) >= OVERLAP_IOU]
        if faint_hits:
            if any(c == cls for _b, c in faint_hits):
                lines.append(yolo_line(cls, max(faint_hits, key=lambda h: _iou(box, h[0]))[0]))
                counts["confirmed"] += 1
            continue
        lines.append(yolo_line(cls, box))
        counts["buildabot"] += 1
    return "".join(lines), counts


def held_matches() -> set:
    return {Path(json.loads(p.read_text(encoding="utf-8"))["frame"]).parent.name
            for p in TRUTH.glob("f*_pre.json")}


def write_val(by_key: dict) -> int:
    from brain.vision import _key
    images, labels = OUT / "images" / "val", OUT / "labels" / "val"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    written = 0
    for truth_file in sorted(TRUTH.glob("f*_truth.json")):
        pre = json.loads((TRUTH / truth_file.name.replace("_truth", "_pre")).read_text(encoding="utf-8"))
        units = json.loads(truth_file.read_text(encoding="utf-8"))["units"]
        lines = "".join(yolo_line(by_key[_key(u["name"])], to_crop(u["box"]))
                        for u in units if _key(u["name"]) in by_key)
        stem = truth_file.stem.replace("_truth", "")
        Image.fromarray(frame_crop(ROOT / pre["frame"])).save(images / f"truth_{stem}.jpg", quality=92)
        (labels / f"truth_{stem}.txt").write_text(lines, encoding="utf-8")
        written += 1
    return written


def main() -> int:
    from ultralytics import YOLO
    from brain.vision import NOT_A_UNIT, NOT_A_UNIT_EITHER, SPELL_CLASSES
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-match", type=int, default=15)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    names, by_key = class_ids()
    skip = (set(NOT_A_UNIT) | set(NOT_A_UNIT_EITHER)) - set(SPELL_CLASSES)

    def keep_class(cls: int) -> bool:
        return names.get(cls, "") not in skip

    model = YOLO(str(WEIGHTS))
    rng = random.Random(args.seed)
    held = held_matches()
    images, labels = OUT / "images" / "train", OUT / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    totals = {"frames": 0, "yolo": 0, "confirmed": 0, "buildabot": 0, "played": 0}
    folders = [f for f in sorted(LIVE.glob("match_*")) if f.name not in held]
    for index, folder in enumerate(folders):
        sidecars = sorted(folder.glob("*.json"))
        if len(sidecars) < 60:
            continue
        end = float(sidecars[-1].stem.replace("_", "."))
        busy = []
        for sidecar in sidecars:
            t = float(sidecar.stem.replace("_", "."))
            if SKIP_START <= t <= end - SKIP_END and sidecar.with_suffix(".jpg").exists():
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if len(data.get("tracks", [])) >= MIN_TRACKS:
                    busy.append((sidecar, data.get("tracks", []), data.get("plays", [])))
        for sidecar, tracks, plays in rng.sample(busy, min(args.per_match, len(busy))):
            pixels = frame_crop(sidecar.with_suffix(".jpg"))
            text, counts = labels_for(model, pixels, tracks, plays, by_key, keep_class)
            stem = f"{folder.name}_{sidecar.stem}"
            Image.fromarray(pixels).save(images / f"{stem}.jpg", quality=92)
            (labels / f"{stem}.txt").write_text(text, encoding="utf-8")
            totals["frames"] += 1
            for key, value in counts.items():
                totals[key] += value
        if index % 40 == 0:
            print(f"{index}/{len(folders)} matches {totals}", flush=True)
    val = write_val(by_key)
    synth = ROOT / "tmp" / "yolo_synth" / "images" / "train"
    (OUT / "data.yaml").write_text(
        f"path: {ROOT.as_posix()}\n"
        f"train:\n  - {images.as_posix()}\n  - {synth.as_posix()}\n"
        f"val:\n  - {(OUT / 'images' / 'val').as_posix()}\n"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(names.items())),
        encoding="utf-8")
    print(f"done {totals}, {val} checked val frames -> {OUT / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
