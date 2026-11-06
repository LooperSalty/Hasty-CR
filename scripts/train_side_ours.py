"""Whose unit is it, learned from our own recorded matches.

On the checked frames of scripts/truth_set.py the side classifier trained on
the published dataset (another phone, other arenas) got 83-85% of sides
right, and almost every miss was one of our troops called the enemy's (our
Archers 8 times out of 125 units). Our recordings (tmp/dataset/live, a frame
and its tracks every second) carry labels it never had:

* ALLY  - a track that appears where and just after we dropped a card that
          brings that unit (the sidecar's `plays`);
* ENEMY - a track born beyond the river that moves (a static box there is
          most often a tower misread), with no drop of ours to explain it.

Every frame of a labelled track gives a crop with that label. Matches holding
a checked frame are left out, and the result is scored on the checked units:

    python scripts/train_side_ours.py --matches 300 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "vendor" / "ClashRoyaleBuildABot"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_side_classifier import CROP, build_model  # noqa: E402
from unit_tracker import CARD_UNITS  # noqa: E402

LIVE = ROOT / "tmp" / "dataset" / "live"
TRUTH = ROOT / "tmp" / "truth" / "set1"
OUT = ROOT / "tmp" / "yolo" / "side_classifier_ours.pt"
ALLY, ENEMY = 0, 1                      # Seen.side convention: 0 = ours
PLAY_SECONDS = 4.0                      # our unit shows up this soon after the drop...
PLAY_TILES = 4.0                        # ...this close to it, plus a walk
WALK_TILES_PER_SECOND = 1.2
THEIR_HALF_Y = 17.5
MIN_MOVE_TILES = 1.0
CROPS_PER_TRACK = 6
# Crop around the box, as shares of its size: (left, top, right, bottom). The
# top share reaches the HP bar and level badge above the unit - blue for ours,
# red for theirs - which a tight crop cuts off. Saved in the checkpoint so
# brain/vision.py crops the same way.
PAD = (0.15, 0.15, 0.15, 0.15)
OUR_HALF_MIN_FRAME_Y = 824              # frame pixel row of the river: below it is our half


def _units(card: str) -> tuple:
    return CARD_UNITS.get(card, (card,))


def label_tracks(sidecars: list) -> dict:
    """track id -> (side, [(jpg, box), ...]) for the tracks a rule labels."""
    first, frames, path_of = {}, {}, {}
    plays = set()
    for sidecar in sidecars:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        t = float(data.get("match_time", 0.0))
        for play in data.get("plays", []):
            plays.add((play["card"], tuple(play["tile"]), round(t - play["age"], 1)))
        for track in data.get("tracks", []):
            if not track.get("box"):
                continue
            tid = track["id"]
            first.setdefault(tid, (t, track["name"], track["x"], track["y"]))
            frames.setdefault(tid, []).append((sidecar.with_suffix(".jpg"), track["box"]))
            path_of.setdefault(tid, []).append((track["x"], track["y"]))
    labelled = {}
    for tid, (t, name, x, y) in first.items():
        ours = any(name in _units(card) and 0.0 <= t - when <= PLAY_SECONDS
                   and math.hypot(tile[0] - x, tile[1] - y)
                   <= PLAY_TILES + WALK_TILES_PER_SECOND * (t - when)
                   for card, tile, when in plays)
        steps = path_of[tid]
        moved = max(math.hypot(px - x, py - y) for px, py in steps)
        if ours:
            labelled[tid] = (ALLY, frames[tid])
        elif y >= THEIR_HALF_Y and moved >= MIN_MOVE_TILES and len(steps) >= 2:
            labelled[tid] = (ENEMY, frames[tid])
    return labelled


def crop(image: Image.Image, box, size: int = CROP, pad=PAD) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    region = (max(0, x1 - w * pad[0]), max(0, y1 - h * pad[1]),
              min(image.width, x2 + w * pad[2]), min(image.height, y2 + h * pad[3]))
    return np.asarray(image.crop(region).resize((size, size), Image.BILINEAR))


def truth_matches() -> set:
    return {Path(json.loads(pre.read_text(encoding="utf-8"))["frame"]).parent.name
            for pre in TRUTH.glob("f*_pre.json")}


def build(matches: int, rng: random.Random, size: int, pad):
    held = truth_matches()
    folders = [f for f in sorted(LIVE.glob("match_*")) if f.name not in held]
    rng.shuffle(folders)
    data = {"train": ([], []), "val": ([], [])}
    for index, folder in enumerate(folders[:matches]):
        split = "val" if index % 10 == 0 else "train"
        by_frame = {}
        for side, frames in label_tracks(sorted(folder.glob("*.json"))).values():
            step = max(1, len(frames) // CROPS_PER_TRACK)
            for jpg, box in frames[::step][:CROPS_PER_TRACK]:
                by_frame.setdefault(jpg, []).append((side, box))
        for jpg, items in by_frame.items():
            if not jpg.exists():
                continue
            image = Image.open(jpg).convert("RGB")
            for side, box in items:
                data[split][0].append(crop(image, box, size, pad))
                data[split][1].append(side)
        if index % 25 == 0:
            print(f"{index}/{matches} matches, {len(data['train'][0])} train crops", flush=True)
    return {k: (np.stack(x), np.array(y, dtype=np.int64)) for k, (x, y) in data.items()}


def truth_crops(size: int, pad):
    xs, ys, ours_half = [], [], []
    for truth_file in sorted(TRUTH.glob("f*_truth.json")):
        pre = json.loads((TRUTH / truth_file.name.replace("_truth", "_pre")).read_text(encoding="utf-8"))
        image = Image.open(ROOT / pre["frame"]).convert("RGB")
        for unit in json.loads(truth_file.read_text(encoding="utf-8"))["units"]:
            xs.append(crop(image, unit["box"], size, pad))
            ys.append(ALLY if unit["side"] == "ally" else ENEMY)
            ours_half.append((unit["box"][1] + unit["box"][3]) / 2 > OUR_HALF_MIN_FRAME_Y)
    return np.stack(xs), np.array(ys), np.array(ours_half)


def evaluate(model_path: Path, device: str) -> str:
    import torch
    blob = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(blob["state_dict"])
    model = model.to(device).eval()
    xs, ys, ours = truth_crops(int(blob.get("crop", CROP)), tuple(blob.get("pad", PAD)))
    with torch.no_grad():
        tensor = torch.from_numpy(xs).permute(0, 3, 1, 2).float().div(255).to(device)
        predicted = model(tensor).argmax(1).cpu().numpy()
    right = predicted == ys
    return (f"{right.mean() * 100:.1f}% of {len(ys)} checked units "
            f"(our half {right[ours].mean() * 100:.1f}%, their half {right[~ours].mean() * 100:.1f}%, "
            f"ally {right[ys == ALLY].mean() * 100:.1f}%, enemy {right[ys == ENEMY].mean() * 100:.1f}%)")


def train(splits, epochs: int, batch: int, lr: float, device: str, size: int, pad, out: Path) -> float:
    import torch
    import torch.nn as nn
    (tx, ty), (vx, vy) = splits["train"], splits["val"]
    model = build_model().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, total_steps=epochs * (len(tx) // batch + 1))
    weight = torch.tensor([len(ty) / max(1, int((ty == c).sum())) for c in (0, 1)],
                          dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight / weight.sum() * 2)
    txt = torch.from_numpy(tx).permute(0, 3, 1, 2).float().div_(255)
    tyt = torch.from_numpy(ty)
    vxt = torch.from_numpy(vx).permute(0, 3, 1, 2).float().div_(255).to(device)
    vyt = torch.from_numpy(vy).to(device)
    best = 0.0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(txt))
        for start in range(0, len(txt), batch):
            index = order[start:start + batch]
            xb, yb = txt[index].to(device), tyt[index].to(device)
            if random.random() < 0.5:                  # horizontal only: facing is a cue
                xb = torch.flip(xb, dims=[3])
            optimiser.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            optimiser.step()
            schedule.step()
        model.eval()
        with torch.no_grad():
            predicted = torch.cat([model(vxt[i:i + 2048]).argmax(1) for i in range(0, len(vxt), 2048)])
            accuracy = float((predicted == vyt).float().mean())
        print(f"epoch {epoch + 1:2d}/{epochs}  val {accuracy * 100:5.2f}%", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({"state_dict": model.state_dict(), "crop": size, "pad": list(pad),
                        "accuracy": accuracy}, out)
    return best


def main() -> int:
    import torch
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--crop", type=int, default=CROP)
    parser.add_argument("--pad", type=float, nargs=4, default=PAD, metavar=("L", "T", "R", "B"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    splits = build(args.matches, random.Random(args.seed), args.crop, tuple(args.pad))
    for name, (_x, y) in splits.items():
        print(f"{name}: {len(y)} crops, ally {int((y == ALLY).sum())}, enemy {int((y == ENEMY).sum())}")
    print(f"crops built in {time.time() - started:.0f} s")
    best = train(splits, args.epochs, args.batch, args.lr, device, args.crop, tuple(args.pad), args.out)
    print(f"best val {best * 100:.2f}% -> {args.out}")
    print("checked units, old classifier:", evaluate(ROOT / "tmp" / "yolo" / "side_classifier.pt", device))
    print("checked units, new classifier:", evaluate(args.out, device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
