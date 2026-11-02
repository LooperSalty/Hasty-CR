"""The runner keeps battle frames with what it saw and what it played.

Raw material for growing the detector's dataset automatically: each saved
frame has a sidecar JSON with the tracked units visible in that frame (name,
side, box in 1080x1920 pixels, how many frames the track has lasted) and our
recent plays - which say, with certainty, what our own units are.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from PIL import Image  # noqa: E402

from dataset_recorder import DatasetRecorder  # noqa: E402


def track(tid, name, side, missing=False, hits=4):
    return SimpleNamespace(id=tid, name=name, side=side, x=4.0, y=10.0,
                           box=(100, 200, 160, 280), conf=0.8, hits=hits,
                           first_seen=1.0, missing=missing)


def perception(*tracks):
    return SimpleNamespace(tracker=SimpleNamespace(tracks=list(tracks)))


def test_one_frame_a_second_with_its_sidecar(tmp_path):
    recorder = DatasetRecorder(tmp_path, every=1.0)
    frame = Image.new("RGB", (1080, 1920))
    p = perception(track(1, "giant", "enemy"), track(2, "archer", "ally"))
    assert recorder.record(frame, p, now=10.0, match_started=5.0) is True
    assert recorder.record(frame, p, now=10.4, match_started=5.0) is False   # too soon
    assert recorder.record(frame, p, now=11.1, match_started=5.0) is True
    sidecars = sorted(tmp_path.rglob("*.json"))
    assert len(sidecars) == 2 and len(list(tmp_path.rglob("*.jpg"))) == 2
    data = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert data["match_time"] == 5.0
    assert [(t["name"], t["side"], t["box"]) for t in data["tracks"]] == [
        ("giant", "enemy", [100, 200, 160, 280]), ("archer", "ally", [100, 200, 160, 280])]


def test_only_units_seen_in_this_frame_are_written(tmp_path):
    """A held track keeps its last box: in this frame it may be somewhere else."""
    recorder = DatasetRecorder(tmp_path, every=1.0)
    p = perception(track(1, "giant", "enemy"), track(2, "knight", "enemy", missing=True))
    recorder.record(Image.new("RGB", (1080, 1920)), p, now=10.0, match_started=0.0)
    data = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert [t["name"] for t in data["tracks"]] == ["giant"]


def test_our_plays_go_with_the_frames_that_follow(tmp_path):
    recorder = DatasetRecorder(tmp_path, every=1.0)
    recorder.played("musketeer", (5, 3), now=20.0)
    recorder.record(Image.new("RGB", (1080, 1920)), perception(), now=21.0, match_started=0.0)
    data = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert data["plays"] == [{"card": "musketeer", "tile": [5, 3], "age": 1.0}]


def test_old_plays_are_dropped(tmp_path):
    recorder = DatasetRecorder(tmp_path, every=1.0)
    recorder.played("knight", (3, 8), now=0.0)
    recorder.record(Image.new("RGB", (1080, 1920)), perception(), now=30.0, match_started=0.0)
    data = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert data["plays"] == []


def test_a_new_match_gets_its_own_folder(tmp_path):
    recorder = DatasetRecorder(tmp_path, every=1.0)
    frame = Image.new("RGB", (1080, 1920))
    recorder.record(frame, perception(), now=10.0, match_started=5.0)
    recorder.record(frame, perception(), now=400.0, match_started=390.0)
    assert len([d for d in tmp_path.iterdir() if d.is_dir()]) == 2
