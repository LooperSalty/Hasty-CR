"""What a runner does on a screen the detector cannot name.

The order and timings are the ones cr_bot.py learned live: reward screens are
advanced at once, a connection-lost dialog is answered at once, a popup is
closed after 15 s, and the app is relaunched after 75 s as the last resort.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from unknown_screen import Detectors, UnknownScreenHandler  # noqa: E402


class Recorder:
    def __init__(self):
        self.taps, self.relaunches, self.lines = [], 0, []

    def tap(self, xy):
        self.taps.append(xy)

    def relaunch(self):
        self.relaunches += 1

    def log(self, text):
        self.lines.append(text)


def handler(reward=False, reload_at=None, close_at=None, confirm_at=None):
    rec = Recorder()
    detectors = Detectors(
        is_reward=lambda frame: reward,
        advance_point=lambda attempt: (540, 300) if attempt % 2 == 0 else (540, 960),
        find_reload=lambda frame: reload_at,
        find_close=lambda frame: close_at,
        find_confirm=lambda frame: confirm_at,
    )
    return UnknownScreenHandler(rec.tap, rec.relaunch, rec.log, detectors), rec


def test_a_confirm_button_is_pressed_after_the_screen_settles():
    h, rec = handler(confirm_at=(540, 1580))
    assert h.handle(None, 0.0) == "wait"            # transitions flash blue too
    assert h.handle(None, 3.5) == "confirm"
    assert h.handle(None, 4.0) == "wait"            # one press per few seconds
    assert h.handle(None, 7.0) == "confirm"
    assert rec.taps == [(540, 1580), (540, 1580)]


def test_reward_screens_are_advanced_immediately_and_alternate_tap_points():
    h, rec = handler(reward=True)
    assert h.handle(None, 0.0) == "reward"
    assert h.handle(None, 0.5) == "wait"          # too soon for another tap
    assert h.handle(None, 2.0) == "reward"
    assert rec.taps == [(540, 960), (540, 300)]


def test_connection_lost_is_answered_at_once_but_not_spammed():
    h, rec = handler(reload_at=(540, 1100))
    assert h.handle(None, 0.0) == "reload"
    assert h.handle(None, 3.0) == "wait"
    assert h.handle(None, 9.0) == "reload"
    assert rec.taps == [(540, 1100), (540, 1100)]


def test_popup_closed_only_after_fifteen_seconds_unknown():
    h, rec = handler(close_at=(980, 300))
    assert h.handle(None, 0.0) == "wait"
    assert h.handle(None, 10.0) == "wait"
    assert h.handle(None, 16.0) == "popup"
    assert rec.taps == [(980, 300)]
    assert h.handle(None, 20.0) == "wait"          # 25 s between popup attempts


def test_app_is_relaunched_after_seventy_five_seconds_with_cooldown():
    h, rec = handler()
    for t in range(0, 85, 5):
        h.handle(None, float(t))
    assert rec.relaunches == 1
    assert any("RECOVER" in line for line in rec.lines)
    h.handle(None, 120.0)
    assert rec.relaunches == 1                     # 150 s cooldown


def test_a_known_screen_resets_the_unknown_clock():
    h, rec = handler(close_at=(980, 300))
    h.handle(None, 0.0)
    h.known()
    assert h.handle(None, 16.0) == "wait"          # unknown again only since 16 s
    assert h.handle(None, 32.0) == "popup"


def test_the_screen_is_shown_to_the_runner_before_ok_takes_it_away():
    """An end-of-match screen the detector cannot name ends here: the runner
    must read the result off it before the confirm tap."""
    rec = Recorder()
    seen = []
    detectors = Detectors(is_reward=lambda frame: False, advance_point=lambda n: (0, 0),
                          find_reload=lambda frame: None, find_close=lambda frame: None,
                          find_confirm=lambda frame: (704, 1742))
    h = UnknownScreenHandler(rec.tap, rec.relaunch, rec.log, detectors,
                             before_confirm=lambda frame: seen.append((frame, list(rec.taps))))
    h.handle("frame-0", 0.0)
    h.handle("frame-1", 3.5)
    assert seen == [("frame-1", [])]                 # before the tap, with that frame
    assert rec.taps == [(704, 1742)]


def test_a_failing_reader_never_blocks_the_confirm_tap():
    rec = Recorder()

    def broken(frame):
        raise ValueError("bad frame")

    detectors = Detectors(is_reward=lambda frame: False, advance_point=lambda n: (0, 0),
                          find_reload=lambda frame: None, find_close=lambda frame: None,
                          find_confirm=lambda frame: (704, 1742))
    h = UnknownScreenHandler(rec.tap, rec.relaunch, rec.log, detectors, before_confirm=broken)
    h.handle(None, 0.0)
    assert h.handle(None, 3.5) == "confirm"
    assert rec.taps == [(704, 1742)]


def test_another_menu_tab_goes_back_to_battle_instead_of_relaunching():
    """Left on the Cards tab, the runner relaunched the game twice: it reopens
    on the same tab. The Battle tab button is the way back."""
    rec = Recorder()
    detectors = Detectors(is_reward=lambda frame: False, advance_point=lambda n: (0, 0),
                          find_reload=lambda frame: None, find_close=lambda frame: None,
                          find_battle_tab=lambda frame: (626, 1806))
    h = UnknownScreenHandler(rec.tap, rec.relaunch, rec.log, detectors)
    assert h.handle(None, 0.0) == "wait"            # let a transition settle first
    assert h.handle(None, 6.0) == "nav"
    assert rec.taps == [(626, 1806)]
    assert h.handle(None, 7.0) == "wait"            # one tap, then give it time
    assert rec.relaunches == 0


def test_a_battle_tab_tap_that_changes_nothing_still_ends_in_a_relaunch():
    rec = Recorder()
    detectors = Detectors(is_reward=lambda frame: False, advance_point=lambda n: (0, 0),
                          find_reload=lambda frame: None, find_close=lambda frame: None,
                          find_battle_tab=lambda frame: (626, 1806))
    h = UnknownScreenHandler(rec.tap, rec.relaunch, rec.log, detectors)
    for second in range(0, 80):
        h.handle(None, float(second))
    assert rec.relaunches == 1
