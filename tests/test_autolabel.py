"""Turning recorded battle frames into detector labels without a human.

A label is only written when it is safe:

* our own unit: the runner played that card there moments before, and the
  tracked unit has that card's name - certain;
* any other unit: BuildABot's tracked unit (seen for a few frames) and the
  YOLO detector agree on the name at the same place.

A frame with anything left unexplained is not used at all: an unlabelled
troop would teach the detector that troops are background.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from autolabel import Det, label_frame, to_crop  # noqa: E402


def track(name, side, box, x=4.0, y=10.0, hits=4, conf=0.8):
    return {"name": name, "side": side, "box": list(box), "x": x, "y": y, "hits": hits,
            "conf": conf}


def test_our_own_play_labels_its_units_for_certain():
    tracks = [track("musketeer", "ally", (300, 1200, 380, 1320), x=5, y=4)]
    plays = [{"card": "musketeer", "tile": [5, 3], "age": 2.0}]
    labels, unresolved = label_frame(tracks, [], plays)
    assert [(l.name, l.source) for l in labels] == [("musketeer", "own")]
    assert unresolved == []


def test_a_play_names_the_card_s_units():
    tracks = [track("archer", "ally", (300, 1200, 340, 1260), x=5, y=4),
              track("archer", "ally", (350, 1200, 390, 1260), x=6, y=4)]
    plays = [{"card": "archers", "tile": [5, 3], "age": 1.5}]
    labels, unresolved = label_frame(tracks, [], plays)
    assert [l.name for l in labels] == ["archer", "archer"] and unresolved == []


def test_an_enemy_needs_both_detectors_to_agree():
    tracks = [track("giant", "enemy", (400, 500, 520, 640), y=22)]
    yolo = [Det("giant", (405, 505, 515, 645), 0.8)]
    labels, unresolved = label_frame(tracks, yolo, [])
    assert [(l.name, l.source) for l in labels] == [("giant", "agree")]
    assert labels[0].box == (405, 505, 515, 645)            # YOLO's finer box


def test_a_disagreement_leaves_the_frame_unusable():
    tracks = [track("archer", "enemy", (400, 500, 450, 560), y=22)]
    yolo = [Det("princess", (402, 502, 452, 562), 0.7)]
    labels, unresolved = label_frame(tracks, yolo, [])
    assert labels == [] and len(unresolved) == 1


def test_a_unit_only_one_detector_sees_leaves_the_frame_unusable():
    labels, unresolved = label_frame([], [Det("knight", (100, 100, 160, 180), 0.7)], [])
    assert labels == [] and len(unresolved) == 1


def test_a_brand_new_track_is_not_trusted_yet():
    tracks = [track("giant", "enemy", (400, 500, 520, 640), hits=1)]
    yolo = [Det("giant", (405, 505, 515, 645), 0.8)]
    labels, unresolved = label_frame(tracks, yolo, [])
    assert labels == [] and len(unresolved) == 1


def test_faint_detections_do_not_block_a_frame():
    labels, unresolved = label_frame([], [Det("bat", (100, 100, 120, 120), 0.2)], [])
    assert labels == [] and unresolved == []


def test_boxes_move_into_the_dataset_framing():
    # ARENA_CROP (84, 161, 996, 1487) -> 568x896
    assert to_crop((84, 161, 996, 1487)) == (0.0, 0.0, 568.0, 896.0)
    x0, y0, x1, y1 = to_crop((540, 824, 600, 900))
    assert round(x0) == 284 and round(y0) == 448


def test_a_card_being_dropped_that_nobody_sees_yet_blocks_the_frame():
    """The deploy animation: the card name is on screen, no model sees the
    unit yet. Accepting the frame would teach that it is background."""
    labels, unresolved = label_frame([], [], [{"card": "minipekka", "tile": [3, 8], "age": 0.8}])
    assert labels == [] and len(unresolved) == 1 and unresolved[0].box is None


def test_unresolved_units_come_with_their_boxes_for_erasing():
    tracks = [track("archer", "enemy", (400, 500, 450, 560), y=22)]
    _, unresolved = label_frame(tracks, [Det("princess", (402, 502, 452, 562), 0.7)], [])
    assert unresolved[0].box == (400, 500, 450, 560)


def test_an_ambiguous_unit_is_erased_with_the_empty_arena():
    from PIL import Image
    from autolabel import erase
    frame = Image.new("RGB", (1080, 1920), (200, 0, 0))
    background = Image.new("RGB", (1080, 1920), (0, 200, 0))
    out = erase(frame, background, [Det("princess", (400, 500, 450, 560), 0.7)], labels=[])
    assert out is not None and out.getpixel((425, 530)) == (0, 200, 0)
    assert out.getpixel((10, 10)) == (200, 0, 0)


def test_erasing_that_would_cut_into_a_labelled_unit_is_refused():
    from PIL import Image
    from autolabel import Label, erase
    frame = Image.new("RGB", (1080, 1920))
    labels = [Label("giant", (420, 520, 520, 640), "agree")]
    assert erase(frame, frame, [Det("princess", (400, 500, 450, 560), 0.7)], labels) is None


def test_a_faint_but_real_detection_must_still_be_explained():
    """Two Archers at a bridge seen at 0.35 were left as background."""
    _, unresolved = label_frame([], [Det("archer", (100, 100, 130, 140), 0.35)], [])
    assert len(unresolved) == 1
