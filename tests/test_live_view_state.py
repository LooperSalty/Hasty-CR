"""What the runner publishes for the HUD and the decision panel."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from live_view_state import ViewWriter, snapshot  # noqa: E402


def unit(name, x, y, conf, bbox=(0, 0, 0, 0)):
    return SimpleNamespace(unit=SimpleNamespace(name=name),
                           position=SimpleNamespace(tile_x=x, tile_y=y, conf=conf, bbox=bbox))


def battle_state():
    number = SimpleNamespace
    return SimpleNamespace(
        screen=SimpleNamespace(name="in_game"),
        cards=[SimpleNamespace(name=n) for n in ("knight", "giant", "archers", "arrows", "musketeer")],
        ready=[0, 2],
        allies=[unit("giant", 4, 7, 0.91, bbox=(100, 900, 180, 1010))],
        enemies=[unit("hog_rider", 14.4, 18.2, 0.66)],
        numbers=SimpleNamespace(
            elixir=number(number=7),
            left_enemy_princess_hp=number(number=1.0), right_enemy_princess_hp=number(number=0.4),
            left_ally_princess_hp=number(number=1.0), right_ally_princess_hp=number(number=0.9)))


def make(**overrides):
    args = dict(phase="push", reason="Push voie gauche",
                recent=["12:00:01 giant (8,0) - Géant au fond"], target=(256, 876),
                strategy="beatdown", finished=3, actions=40, enemy_elixir=4.5,
                tile_to_px=lambda x, y: (x * 10, y * 10), box_scale=1.0)
    args.update(overrides)
    return snapshot(battle_state(), **args)


def test_snapshot_carries_sight_hand_elixirs_and_reasoning():
    data = make()
    assert data["screen"] == "in_game" and data["elixir"] == 7
    assert data["enemy_elixir"] == 4.5
    assert data["hand"] == [{"name": "giant", "ready": True}, {"name": "archers", "ready": False},
                            {"name": "arrows", "ready": True}, {"name": "musketeer", "ready": False}]
    assert data["next_card"] == "knight"
    assert data["allies"][0]["box"] == [100, 900, 180, 1010]
    assert data["enemies"][0]["box"] is None          # no box: the HUD falls back to the tile
    assert data["enemies"][0]["px"] == 144
    assert data["towers"]["enemy_right"] == 0.4
    assert data["phase"] == "push" and data["target"] == [256, 876]
    json.dumps(data)


def test_upstream_boxes_are_scaled_from_the_368_wide_frame():
    data = make(box_scale=1080 / 368)
    assert data["allies"][0]["box"] == [293, 2641, 528, 2964]


def test_writer_throttles_and_writes_atomically(tmp_path):
    writer = ViewWriter(tmp_path, every=0.05)
    assert writer.write({"a": 1}, now=0.0)
    assert not writer.write({"a": 2}, now=0.02)
    assert writer.write({"a": 3}, now=0.06)
    assert json.loads((tmp_path / "native_view.json").read_text(encoding="utf-8"))["a"] == 3
    assert not list(tmp_path.glob("*.part*"))


def test_a_file_held_by_the_hud_is_retried_not_dropped(tmp_path, monkeypatch):
    """Windows refuses to replace a file another process is reading: the HUD
    polls it 60 times a second, so the writer retries instead of failing."""
    import os
    import live_view_state
    real_replace, calls = os.replace, []

    def busy_once(src, dst):
        calls.append(dst)
        if len(calls) == 1:
            raise PermissionError(5, "Accès refusé")
        real_replace(src, dst)

    monkeypatch.setattr(live_view_state.os, "replace", busy_once)
    writer = live_view_state.ViewWriter(tmp_path, every=0.0)
    assert writer.write({"elixir": 4}, now=1.0) is True
    assert len(calls) == 2 and (tmp_path / "native_view.json").is_file()
