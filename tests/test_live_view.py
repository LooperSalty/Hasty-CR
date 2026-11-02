"""The live view window: where it docks, and how it reads the bot's files."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from control_panel.live_view_model import (BoxFollower, ViewReader, dock_geometry,  # noqa: E402
                                           read_view)

SCREEN = (0, 0, 2560, 1400)


def unit(name, x0, y0, size=100):
    return {"name": name, "conf": 0.9, "px": x0 + size // 2, "py": y0 + size // 2,
            "box": [x0, y0, x0 + size, y0 + size]}


def frame(allies=(), enemies=(), time=0.0):
    return {"screen": "in_game", "allies": list(allies), "enemies": list(enemies), "time": time}


def test_a_new_unit_is_drawn_where_it_was_seen():
    follower = BoxFollower()
    follower.update(frame(enemies=[unit("giant", 100, 200)], time=1.0), received=10.0)
    [box] = follower.frame(now=10.0)
    assert (box.side, box.name) == ("enemy", "giant")
    assert box.rect == (100, 200, 200, 300)


def test_a_moving_unit_glides_instead_of_jumping():
    follower = BoxFollower()
    follower.update(frame(enemies=[unit("giant", 100, 200)], time=1.0), received=10.0)
    follower.frame(now=10.0)
    follower.update(frame(enemies=[unit("giant", 100, 300)], time=1.1), received=10.1)
    first = follower.frame(now=10.1)[0].rect[1]
    assert 200 < first < 300                               # on its way, not teleported
    later = [follower.frame(now=10.1 + i * 0.016)[0].rect[1] for i in range(1, 8)]
    assert later == sorted(later)                          # keeps moving forward
    assert later[-1] >= 295


def test_between_detections_a_moving_unit_keeps_its_course():
    follower = BoxFollower()
    follower.update(frame(enemies=[unit("giant", 100, 200)], time=1.0), received=10.0)
    follower.update(frame(enemies=[unit("giant", 100, 300)], time=1.1), received=10.1)
    for i in range(10):
        follower.frame(now=10.1 + i * 0.01)
    ahead = follower.frame(now=10.2)[0].rect[1]
    assert ahead > 300                                     # predicted past the last read


def test_prediction_is_capped_when_detections_stop():
    follower = BoxFollower()
    follower.update(frame(enemies=[unit("giant", 100, 200)], time=1.0), received=10.0)
    follower.update(frame(enemies=[unit("giant", 100, 300)], time=1.1), received=10.1)
    for i in range(200):
        follower.frame(now=10.1 + i * 0.016)
    top = follower.frame(now=20.0)[0].rect[1]
    assert top <= 300 + BoxFollower.MAX_SPEED * BoxFollower.MAX_PREDICT_SECONDS + 1


def test_two_units_of_the_same_name_keep_their_own_boxes():
    follower = BoxFollower()
    follower.update(frame(allies=[unit("archer", 100, 900), unit("archer", 700, 900)], time=1.0),
                    received=10.0)
    follower.update(frame(allies=[unit("archer", 705, 890), unit("archer", 104, 890)], time=1.1),
                    received=10.1)
    for i in range(20):
        boxes = follower.frame(now=10.1 + i * 0.016)
    lefts = sorted(round(b.rect[0]) for b in boxes)
    assert lefts[0] < 200 and lefts[1] > 600


def test_the_same_frame_read_twice_does_not_stop_a_moving_unit():
    follower = BoxFollower()
    follower.update(frame(enemies=[unit("giant", 100, 200)], time=1.0), received=10.0)
    follower.update(frame(enemies=[unit("giant", 100, 300)], time=1.1), received=10.1)
    speed = follower.targets[0].velocity
    follower.update(frame(enemies=[unit("giant", 100, 300)], time=1.1), received=10.12)
    assert follower.targets[0].velocity == speed and follower.received == 10.1


def test_a_unit_that_left_the_data_is_gone():
    follower = BoxFollower()
    follower.update(frame(enemies=[unit("giant", 100, 200)], time=1.0), received=10.0)
    follower.update(frame(time=1.1), received=10.1)
    assert follower.frame(now=10.1) == []


def test_a_unit_without_a_box_gets_a_square_on_its_tile():
    follower = BoxFollower()
    data = frame(enemies=[{"name": "minion", "conf": 0.7, "px": 500, "py": 600, "box": None}])
    follower.update(data, received=1.0)
    [box] = follower.frame(now=1.0)
    x0, y0, x1, y1 = box.rect
    assert (x0 + x1) / 2 == 500 and (y0 + y1) / 2 == 600


def test_view_reader_parses_only_when_the_file_changed(tmp_path, monkeypatch):
    path = tmp_path / "native_view.json"
    path.write_text(json.dumps({"elixir": 3}), encoding="utf-8")
    reader = ViewReader(tmp_path)
    first = reader.poll(now=path.stat().st_mtime + 0.1)
    assert first.view.data["elixir"] == 3 and first.changed
    parsed = []
    monkeypatch.setattr(json, "loads", lambda text: parsed.append(text) or {})
    again = reader.poll(now=path.stat().st_mtime + 0.2)
    assert not again.changed and parsed == [] and again.view.data["elixir"] == 3


def test_docks_to_the_right_of_mumu_at_its_height():
    assert dock_geometry((100, 50, 700, 1100), SCREEN, width=460) == (704, 50, 460, 1050)


def test_goes_left_when_there_is_no_room_on_the_right():
    assert dock_geometry((2000, 0, 2550, 1000), SCREEN, width=460) == (1536, 0, 460, 1000)


def test_without_mumu_it_sits_at_the_screen_edge():
    x, y, w, h = dock_geometry(None, SCREEN, width=460)
    assert (x, w) == (2560 - 460, 460) and h > 0


def test_read_view_reports_missing_and_stale_data(tmp_path):
    assert read_view(tmp_path, now=100.0) is None
    (tmp_path / "native_view.json").write_text(json.dumps({"elixir": 7}), encoding="utf-8")
    view = read_view(tmp_path, now=100.0, file_age=1.0)
    assert view.data["elixir"] == 7 and not view.stale
    view = read_view(tmp_path, now=100.0, file_age=12.0)
    assert view.stale


def test_read_view_survives_a_half_written_file(tmp_path):
    (tmp_path / "native_view.json").write_text("{\"elixir\": ", encoding="utf-8")
    assert read_view(tmp_path, now=100.0, file_age=0.5) is None
