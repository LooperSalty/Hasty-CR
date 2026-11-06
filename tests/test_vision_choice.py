"""Which detector and side classifier the live bot loads, and how it crops.

Measured on the checked frames of scripts/truth_set.py (2026-11-06): the
detector trained on our own frames found 86% of the units against 76%, and
the side classifier trained on our recordings got 94% of sides against 84%.
The winner is pinned in tmp/yolo/runs/chosen.txt so a run still training -
newer on disk - is never picked up by date.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

import brain.vision as vision  # noqa: E402


def _run(root: Path, name: str) -> Path:
    weights = root / name / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    return weights


def test_the_pinned_run_wins_over_a_newer_one(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "RUNS", tmp_path)
    pinned = _run(tmp_path, "good")
    _run(tmp_path, "newer")
    (tmp_path / "chosen.txt").write_text("good\n", encoding="utf-8")
    assert vision.resolve_weights() == pinned


def test_without_a_pin_the_newest_run_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "RUNS", tmp_path)
    _run(tmp_path, "old")
    newest = _run(tmp_path, "new")
    assert vision.resolve_weights() == newest


def test_a_pin_to_a_missing_run_falls_back_to_the_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "RUNS", tmp_path)
    newest = _run(tmp_path, "only")
    (tmp_path / "chosen.txt").write_text("gone\n", encoding="utf-8")
    assert vision.resolve_weights() == newest


def test_resize_gives_the_size_asked_for_both_ways():
    frame = np.random.default_rng(0).integers(0, 255, (1326, 912, 3), dtype=np.uint8)
    assert vision._resize(frame, (568, 896)).shape == (896, 568, 3)
    assert vision._resize(frame[:20, :30], (48, 48)).shape == (48, 48, 3)


def test_the_side_classifier_keeps_its_default_crop_without_a_checkpoint(tmp_path):
    sides = vision.SideClassifier(tmp_path / "missing.pt")
    assert not sides.ready
    assert sides.pad == (0.125, 0.125, 0.125, 0.125) and sides.crop == 32
