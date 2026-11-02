"""Synthetic detector training images: dataset sprites on real arena backgrounds.

The detector trained on the published dataset alone was overfit (its
validation frames came from the same matches as its training frames) and
missed or misnamed most troops live. The dataset also ships 4654 transparent
sprites in 150 classes - the material KataCR built its generative dataset
from. This paints them onto arena backgrounds in the dataset's own framing:

* backgrounds: this account's arenas (scripts/arena_backgrounds.py medians,
  cut to ARENA_CROP - the window of our 1080x1920 frame that matches the
  dataset's 568x896 framing, from the tower positions) and the dataset's 28
  arena backgrounds, so the next arena is not new either;
* 4-18 units each, any class, sprite scale jittered, mirrored half the time,
  painted back to front, with level badges and HP bars as unlabelled clutter;
* a label only for what stays at least MIN_VISIBLE visible.

Only units and spells are labelled - towers, bars and UI are background, here
and in the filtered copy of the original labels (`units_only_labels`).

    python scripts/synth_dataset.py --train 8000 --val 800
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DATASET = ROOT / "vendor" / "CR-Detection-Dataset" / "images"
SEGMENTS = DATASET / "segment"
OUT = ROOT / "tmp" / "yolo_synth"
W, H = 568, 896                              # the dataset's arena framing
# Our 1080x1920 frame -> the dataset framing: princess towers at grid x 3/14
# and rows 7/24 land where the dataset's labels put them (x 112/456, y 203/679).
ARENA_CROP = (84, 161, 996, 1487)
NOT_LABELLED = frozenset({
    "backgrounds", "background-items", "big-text", "small-text", "king-tower",
    "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "dagger-duchess-tower-bar",
    "tower-bar", "king-tower-bar", "bar", "bar-level", "clock", "emote", "text", "elixir",
    "selected", "skeleton-king-bar", "evolution-symbol", "ice-spirit-evolution-symbol",
    "skeleton-king-skill"})
CLUTTER = ("bar-level", "bar", "emote")
UNITS_PER_IMAGE = (4, 18)
SCALE = (0.85, 1.15)
MIN_VISIBLE = 0.4
OURS_SHARE = 0.6                              # backgrounds from our own arenas


def class_names() -> list[str]:
    import yaml
    names = yaml.safe_load((ROOT / "tmp" / "yolo" / "cr.yaml").read_text(encoding="utf-8"))["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def to_dataset_frame(image: Image.Image) -> Image.Image:
    """One of our 1080x1920 frames cut and scaled to the dataset framing."""
    full = image.convert("RGB").resize((1080, 1920), Image.Resampling.BILINEAR)
    return full.crop(ARENA_CROP).resize((W, H), Image.Resampling.BILINEAR)


def load_sprites(names: list[str]) -> dict[str, list[Image.Image]]:
    sprites = {}
    for folder in sorted(p for p in SEGMENTS.iterdir() if p.is_dir()):
        if folder.name in NOT_LABELLED or folder.name not in names:
            continue
        images = [Image.open(png).convert("RGBA") for png in sorted(folder.glob("*.png"))]
        if images:
            sprites[folder.name] = images
    return sprites


def load_clutter() -> list[Image.Image]:
    return [Image.open(png).convert("RGBA") for name in CLUTTER
            for png in sorted((SEGMENTS / name).glob("*.png"))][:400]


def load_backgrounds(ours: Path) -> tuple[list[Image.Image], list[Image.Image]]:
    own = [to_dataset_frame(Image.open(p)) for p in sorted(ours.glob("*.png"))]
    theirs = [Image.open(p).convert("RGB").resize((W, H))
              for p in sorted((SEGMENTS / "backgrounds").glob("*.jpg"))]
    return own, theirs


def _jitter(sprite: Image.Image, rng: random.Random) -> Image.Image:
    scale = rng.uniform(*SCALE)
    aspect = rng.uniform(0.93, 1.07)
    w = max(4, round(sprite.width * scale * aspect))
    h = max(4, round(sprite.height * scale / aspect))
    out = sprite.resize((w, h), Image.Resampling.BILINEAR)
    return out.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else out


def compose(background: Image.Image, sprites, clutter, class_id, rng: random.Random):
    """(image, [(class, cx, cy, w, h) normalised])."""
    canvas = background.copy()
    owner = np.full((H, W), -1, dtype=np.int32)          # which placed unit owns each pixel
    placed = []
    count = rng.randint(*UNITS_PER_IMAGE)
    picks = [rng.choice(list(sprites)) for _ in range(count)]
    items = []
    for name in picks:
        sprite = _jitter(rng.choice(sprites[name]), rng)
        x = rng.randint(-sprite.width // 4, W - 3 * sprite.width // 4)
        y = rng.randint(30, H - sprite.height)
        items.append((y + sprite.height, name, sprite, x, y))
    for _, name, sprite, x, y in sorted(items, key=lambda item: item[0]):
        alpha = np.asarray(sprite)[:, :, 3] > 40
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + sprite.width), min(H, y + sprite.height)
        if x1 <= x0 or y1 <= y0:
            continue
        canvas.paste(sprite, (x, y), sprite)
        index = len(placed)
        region = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
        owner[y0:y1, x0:x1][region] = index
        placed.append((name, x0, y0, x1, y1, int(alpha.sum())))
        if clutter and rng.random() < 0.5:                 # level badge / HP bar above it
            extra = rng.choice(clutter)
            canvas.paste(extra, (x + sprite.width // 2 - extra.width // 2,
                                 max(0, y - extra.height)), extra)
    labels = []
    for index, (name, x0, y0, x1, y1, area) in enumerate(placed):
        mine = owner[y0:y1, x0:x1] == index          # only this unit's own window
        visible = int(mine.sum())
        if area and visible / area >= MIN_VISIBLE:
            ys, xs = np.nonzero(mine)
            bx0, bx1 = x0 + xs.min(), x0 + xs.max() + 1
            by0, by1 = y0 + ys.min(), y0 + ys.max() + 1
            labels.append((class_id[name], (bx0 + bx1) / 2 / W, (by0 + by1) / 2 / H,
                           (bx1 - bx0) / W, (by1 - by0) / H))
    return canvas, labels


def write_split(split: str, count: int, own, theirs, sprites, clutter, class_id, seed: int):
    images, labels = OUT / "images" / split, OUT / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(count):
        pool = own if own and (not theirs or rng.random() < OURS_SHARE) else theirs
        image, boxes = compose(rng.choice(pool), sprites, clutter, class_id, rng)
        image.save(images / f"synth_{split}_{i:05d}.jpg", quality=92)
        (labels / f"synth_{split}_{i:05d}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes),
            encoding="utf-8")
        if i % 500 == 0:
            print(f"{split}: {i}/{count}", flush=True)


def units_only_labels(names: list[str]) -> Path:
    """The original dataset's labels with towers, bars and UI removed."""
    keep = {i for i, name in enumerate(names) if name not in NOT_LABELLED}
    src, dst = ROOT / "tmp" / "yolo", ROOT / "tmp" / "yolo_units"
    for split in ("train", "val"):
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        for label in (src / "labels" / split).glob("*.txt"):
            lines = [line for line in label.read_text(encoding="utf-8").splitlines()
                     if line.strip() and int(line.split()[0]) in keep]
            (dst / "labels" / split / label.name).write_text(
                "".join(line + "\n" for line in lines), encoding="utf-8")
            image = src / "images" / split / (label.stem + ".jpg")
            target = dst / "images" / split / image.name
            if image.exists() and not target.exists():
                try:
                    target.hardlink_to(image)
                except OSError:
                    target.write_bytes(image.read_bytes())
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--val", type=int, default=800)
    parser.add_argument("--backgrounds", type=Path, default=OUT / "backgrounds")
    args = parser.parse_args()
    names = class_names()
    class_id = {name: i for i, name in enumerate(names)}
    sprites = load_sprites(names)
    clutter = load_clutter()
    own, theirs = load_backgrounds(args.backgrounds)
    print(f"{len(sprites)} unit classes, {sum(map(len, sprites.values()))} sprites, "
          f"{len(own)} own + {len(theirs)} dataset backgrounds")
    write_split("train", args.train, own, theirs, sprites, clutter, class_id, seed=1)
    write_split("val", args.val, own, theirs, sprites, clutter, class_id, seed=2)
    units = units_only_labels(names)
    data = OUT / "data.yaml"
    data.write_text(
        f"path: {ROOT.as_posix()}\n"
        f"train:\n  - {(OUT / 'images' / 'train').as_posix()}\n"
        f"  - {(units / 'images' / 'train').as_posix()}\n"
        f"val:\n  - {(OUT / 'images' / 'val').as_posix()}\n"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8")
    print(f"dataset ready: {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
