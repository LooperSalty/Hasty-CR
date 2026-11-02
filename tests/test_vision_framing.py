"""A detector trained on arena crops sees arena crops live too.

The synthetic-data detector is trained in the published dataset's framing:
the window ARENA_CROP of our 1080x1920 frame, scaled to 568x896. The crop
travels with the weights (framing.json beside them), so whichever checkpoint
YoloVision picks, it is fed what it was trained on, and its boxes come back
in frame pixels for the tile mapping and the side classifier.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

from brain.vision import crop_to_frame, load_framing  # noqa: E402

CROP = (84, 161, 996, 1487)


def test_crop_corners_map_back_to_the_crop_window():
    boxes = np.array([[0.0, 0.0, 568.0, 896.0], [284.0, 448.0, 284.0, 448.0]])
    out = crop_to_frame(boxes, CROP, (568, 896))
    assert np.allclose(out[0], [84, 161, 996, 1487])
    assert np.allclose(out[1], [540, 824, 540, 824])


def test_weights_without_a_framing_file_see_the_whole_frame(tmp_path):
    weights = tmp_path / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    assert load_framing(weights) is None


def test_the_framing_file_beside_the_weights_is_read(tmp_path):
    weights = tmp_path / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    (tmp_path / "run" / "framing.json").write_text(
        json.dumps({"arena_crop": list(CROP), "crop_size": [568, 896], "imgsz": 896}),
        encoding="utf-8")
    framing = load_framing(weights)
    assert framing == {"arena_crop": CROP, "crop_size": (568, 896), "imgsz": 896}
