"""What a live runner does on a screen the detector cannot name.

Between matches the game shows chests, trophy-road rewards, new-arena and
new-card screens, offers and "connection lost" dialogs. The detector knows
none of them, and a runner that only sleeps there stops playing for good.

This is cr_bot.py's recovery sequence, lifted into one place so the native
BuildABot runner gets the same behaviour (cr_bot keeps its inline copy):

1. a positively identified reward screen is advanced at once;
2. "connection lost" has no close button, only RELOAD - answered at once;
3. after 3 s, a blue confirm button (arena unlock, King Tower level-up) is
   pressed - those screens have no cross, so step 4 never closes them. The
   runner sees the frame first (`before_confirm`): an end-of-match screen
   the detector failed to name ends here, and its result is on it;
4. after 5 s, another main-menu tab (Cards, Shop, Clan) goes back to the
   Battle tab - relaunching reopens the game on the same tab;
5. after 15 s on an unknown screen, look for a popup's close button;
6. after 75 s, relaunch the app, at most once every 150 s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

UNKNOWN_POPUP_SECONDS = 15
UNKNOWN_RECOVER_SECONDS = 75
UNKNOWN_RECOVER_COOLDOWN = 150
REWARD_TAP_GAP = 1.5
RELOAD_GAP = 8
POPUP_GAP = 25
CONFIRM_SETTLE_SECONDS = 3
CONFIRM_GAP = 3
NAV_SETTLE_SECONDS = 5
NAV_GAP = 5
NEVER = float("-inf")

Point = tuple[int, int]


@dataclass(frozen=True)
class Detectors:
    is_reward: Callable[[object], bool]
    advance_point: Callable[[int], Point]
    find_reload: Callable[[object], Optional[Point]]
    find_close: Callable[[object], Optional[Point]]
    find_confirm: Callable[[object], Optional[Point]] = lambda frame: None
    find_battle_tab: Callable[[object], Optional[Point]] = lambda frame: None


def default_detectors() -> Detectors:
    import connection_lost
    import nav_bar
    import ok_button
    import popup_guard
    import reward_screen
    return Detectors(reward_screen.is_reward_screen, reward_screen.advance_point,
                     connection_lost.find_reload, popup_guard.find_close_button,
                     ok_button.find_ok_button, nav_bar.find_battle_tab)


class UnknownScreenHandler:
    def __init__(self, tap: Callable[[Point], None], relaunch: Callable[[], None],
                 log: Callable[[str], None], detectors: Optional[Detectors] = None,
                 before_confirm: Optional[Callable[[object], None]] = None):
        self._tap, self._relaunch, self._log = tap, relaunch, log
        self._before_confirm = before_confirm
        self._detectors = detectors or default_detectors()
        self._unknown_since: Optional[float] = None
        self._last_reward = self._last_reload = NEVER
        self._last_popup = self._last_recover = NEVER
        self._last_confirm = NEVER
        self._last_nav = NEVER
        self._reward_taps = 0

    def known(self) -> None:
        """Call on every recognised screen: the unknown clock starts afresh."""
        self._unknown_since = None

    def handle(self, frame, now: float) -> str:
        """Act on one unknown frame; returns what was done, for the caller's log."""
        if self._unknown_since is None:
            self._unknown_since = now
        found = self._detectors
        if now - self._last_reward > REWARD_TAP_GAP and found.is_reward(frame):
            self._last_reward = now
            self._reward_taps += 1
            spot = found.advance_point(self._reward_taps)
            self._tap(spot)
            self._log(f"REWARD advance tap #{self._reward_taps} at={spot}")
            self._unknown_since = now
            return "reward"
        reload_spot = found.find_reload(frame)
        if reload_spot is not None and now - self._last_reload > RELOAD_GAP:
            self._last_reload = now
            self._tap(reload_spot)
            self._log(f"RELOAD connection_lost at={reload_spot}")
            self._unknown_since = now
            return "reload"
        unknown_for = now - self._unknown_since
        if unknown_for >= CONFIRM_SETTLE_SECONDS and now - self._last_confirm >= CONFIRM_GAP:
            spot = found.find_confirm(frame)
            if spot is not None:
                self._last_confirm = now
                if self._before_confirm is not None:
                    try:
                        self._before_confirm(frame)
                    except Exception as exc:  # reading must never block the tap
                        self._log(f"CONFIRM reader failed {type(exc).__name__}: {exc}")
                self._tap(spot)
                self._log(f"CONFIRM ok_button at={spot}")
                return "confirm"
        if unknown_for >= NAV_SETTLE_SECONDS and now - self._last_nav >= NAV_GAP:
            self._last_nav = now
            spot = found.find_battle_tab(frame)
            if spot is not None:
                self._tap(spot)
                self._log(f"NAV battle_tab at={spot}")
                # The unknown clock keeps running: a tap that changes nothing
                # must not hold off the popup check and the relaunch forever.
                return "nav"
        action = "wait"
        if unknown_for > UNKNOWN_POPUP_SECONDS and now - self._last_popup > POPUP_GAP:
            self._last_popup = now
            spot = found.find_close(frame)
            if spot is not None:
                self._tap(spot)
                self._log(f"POPUP dismiss at={spot}")
                action = "popup"
            else:
                self._log("POPUP no_close_button")
        if (unknown_for > UNKNOWN_RECOVER_SECONDS
                and now - self._last_recover > UNKNOWN_RECOVER_COOLDOWN):
            self._last_recover = now
            try:
                self._relaunch()
                self._log("RECOVER relaunched_app")
            except Exception as exc:      # logged, the loop must keep going
                self._log(f"RECOVER failed {type(exc).__name__}: {exc}")
            action = "recover"
        return action
