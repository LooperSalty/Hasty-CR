"""The native BuildABot runner must not play cards that do nothing.

Upstream BuildABot refuses to act when the best score's first element is 0
("no good actions available"). The runner's own `choose_native` dropped that
check, so a spell that hits nobody - scored [0, 0, inf] - beat the initial
[0] and was cast at an empty tile, in front of our own King Tower.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts"),
             str(ROOT / "vendor" / "ClashRoyaleBuildABot")):
    if path not in sys.path:
        sys.path.insert(0, path)

import pytest  # noqa: E402

native = pytest.importorskip("mumu_buildabot_native")
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402
from clashroyalebuildabot.namespaces.units import Units  # noqa: E402


def make_state(hand, enemies=(), elixir=10):
    number = SimpleNamespace
    return SimpleNamespace(
        enemies=list(enemies),
        allies=[],
        ready=list(range(len(hand))),
        cards=[None, *hand],
        numbers=SimpleNamespace(
            elixir=number(number=elixir),
            left_enemy_princess_hp=number(number=1000),
            right_enemy_princess_hp=number(number=1000),
        ),
    )


def enemy(tile_x, tile_y, unit=Units.KNIGHT):
    return SimpleNamespace(unit=unit, position=SimpleNamespace(tile_x=tile_x, tile_y=tile_y))


@pytest.fixture(autouse=True)
def starter_deck():
    native.use_deck("starter")
    yield
    native.use_deck("default")


def test_spells_are_not_cast_on_an_empty_board():
    action, _ = native.choose_native(make_state([Cards.FIREBALL, Cards.ARROWS]))
    assert action is None


def test_a_spell_that_hits_too_little_is_held():
    lone = [enemy(8, 20)]
    action, _ = native.choose_native(make_state([Cards.FIREBALL], lone))
    assert action is None


def test_bridge_cards_play_at_full_elixir_and_land_at_the_foot_of_the_bridge():
    """Upstream scores Giant/Mini P.E.K.K.A only on the river tiles (3,15) and
    (14,15), which are not deployable while the princess tower stands - so in
    upstream they never played. They must be offered, and placed one row back."""
    state = make_state([Cards.GIANT, Cards.MINIPEKKA], elixir=10)
    action, score = native.choose_native(state)
    assert action is not None and score[0] == 1
    assert (action.tile_x, action.tile_y) in {(3, 15), (14, 15)}
    assert native.deploy_tile(action, state) == (action.tile_x, 14)


def test_bridge_tile_is_used_as_is_once_that_tower_is_down():
    state = make_state([Cards.GIANT], elixir=10)
    state.numbers.left_enemy_princess_hp.number = 0
    # Upstream sends the Giant at the tower still standing...
    action, _ = native.choose_native(state)
    assert (action.tile_x, action.tile_y) == (14, 15)
    assert native.deploy_tile(action, state) == (14, 14)
    # ...and a bridge tile in a fallen lane is deployable as it is.
    left = native.GiantAction(0, 3, 15)
    assert native.deploy_tile(left, state) == (3, 15)


def test_bridge_cards_wait_below_full_elixir():
    action, _ = native.choose_native(make_state([Cards.GIANT], elixir=9))
    assert action is None


def test_a_match_counts_even_when_the_guard_blocked_every_frame(monkeypatch):
    """A deck preset that does not match the game makes the battle guard reject
    every frame. The match must still be counted from the screens, or
    --max-matches and the panel's stop-after-match never fire."""
    screens = iter(["in_game"] * 4 + ["end_of_game"] * 2)

    def detect(*_args):
        name = next(screens)
        state = make_state([Cards.KNIGHT])
        state.screen = SimpleNamespace(name=name, click_xy=(360, 1100))
        return None, state

    monkeypatch.setattr(native.guarded, "detect", detect)
    monkeypatch.setattr(native.guarded, "home_guard", lambda *a: False)
    monkeypatch.setattr(native.guarded, "battle_guard", lambda *a: False)
    monkeypatch.setattr(native.guarded, "tap", lambda *a: None)
    monkeypatch.setattr(native.time, "sleep", lambda *a: None)
    lines = []
    args = SimpleNamespace(adb="adb", serial="s", max_matches=1, no_queue=True, hours=1)
    runner = native.Runner(args, detector=None, vision=None, log=lines.append)
    for _ in range(4):
        assert runner.step() is False
    assert runner.step() is True                  # result screen: counted, limit reached
    assert runner.finished == 1
    assert any(" RESULT exit" in f" {line}" for line in lines)


def end_screen_runner(monkeypatch, tmp_path, outcome="victoire"):
    """A runner sitting on the end-of-game screen of a match it played."""
    end = SimpleNamespace(name="end_of_game", ltrb=(379, 1122, 560, 1201), click_xy=(243, 1158))

    def detect(*_args):
        state = make_state([Cards.KNIGHT])
        state.screen = end
        return None, state

    taps, lines = [], []
    monkeypatch.setattr(native.guarded, "detect", detect)
    monkeypatch.setattr(native.guarded, "tap", lambda adb, serial, xy: taps.append(xy))
    monkeypatch.setattr(native.time, "sleep", lambda *a: None)
    monkeypatch.setattr(native, "banner_outcome", lambda frame: outcome)
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1,
                           log=tmp_path / "bot.log")
    runner = native.Runner(args, detector=None, vision=None, log=lines.append,
                           stop_flag=tmp_path / native.STOP_FLAG_NAME)
    runner.in_battle = True
    return runner, end, taps, lines


OK_BUTTON = (704, 1742)          # centre of the screen's OK box, 720p x1.5
REPLAY_BUTTON = (364, 1737)      # BuildABot's click_xy: "Rejouer"


def test_after_a_match_it_leaves_through_ok_never_replay(monkeypatch, tmp_path):
    runner, end, taps, lines = end_screen_runner(monkeypatch, tmp_path)
    state = make_state([Cards.KNIGHT])
    state.screen = end
    assert runner.on_result(state, None, "end_of_game", now=100.0) is False
    assert taps == [OK_BUTTON] and REPLAY_BUTTON not in taps


def test_a_stop_after_the_match_does_not_queue_another(monkeypatch, tmp_path):
    """"Arrêter après la partie" used to stop *after* tapping Rejouer: the game
    queued a match nobody played, and lost it."""
    runner, end, taps, lines = end_screen_runner(monkeypatch, tmp_path)
    (tmp_path / native.STOP_FLAG_NAME).write_text("stop", encoding="utf-8")
    state = make_state([Cards.KNIGHT])
    state.screen = end
    assert runner.on_result(state, None, "end_of_game", now=100.0) is True
    assert REPLAY_BUTTON not in taps


def test_each_result_is_logged_with_the_running_record(monkeypatch, tmp_path):
    runner, end, taps, lines = end_screen_runner(monkeypatch, tmp_path, outcome="défaite")
    state = make_state([Cards.KNIGHT])
    state.screen = end
    runner.on_result(state, None, "end_of_game", now=100.0)
    result = next(line for line in lines if "RESULT exit" in line)
    assert "outcome=défaite" in result and "record=0V-1D" in result
    assert runner.record == {"victoire": 0, "défaite": 1, "inconnu": 0}


def test_a_stop_survives_a_failed_ok_tap(monkeypatch, tmp_path):
    """adb drops now and then: the tap raises after the flag was consumed.
    The lobby must still stop instead of queueing."""
    runner, end, taps, lines = end_screen_runner(monkeypatch, tmp_path)
    (tmp_path / native.STOP_FLAG_NAME).write_text("stop", encoding="utf-8")

    def broken_tap(*_args):
        raise RuntimeError("adb: device offline")

    monkeypatch.setattr(native.guarded, "tap", broken_tap)
    state = make_state([Cards.KNIGHT])
    state.screen = end
    try:
        runner.on_result(state, None, "end_of_game", now=100.0)
    except RuntimeError:
        pass
    monkeypatch.setattr(native.guarded, "tap", lambda adb, serial, xy: taps.append(xy))
    runner.args.no_queue = False
    assert runner.on_lobby(now=200.0) is True
    assert native.guarded.HOME_BATTLE not in taps


def test_a_result_screen_the_detector_misses_is_still_read(monkeypatch, tmp_path):
    """BuildABot sometimes cannot name the end screen; the unknown-screen
    handler then presses OK. The result is read off it just before."""
    from PIL import Image
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    lines = []
    runner = native.Runner(args, detector=None, vision=None, log=lines.append,
                           results_dir=tmp_path / "results")
    win = Image.open(ROOT / "tests" / "fixtures" / "screens" / "end_of_game_win.png")
    runner.on_confirm_screen(win)                     # not in a match: nothing to count
    assert runner.finished == 0
    runner.in_battle = True
    runner.on_confirm_screen(win)
    assert runner.finished == 1 and not runner.in_battle
    assert runner.record["victoire"] == 1
    assert any(path.name.endswith("_victoire.jpg") for path in (tmp_path / "results").iterdir())


def test_stop_after_match_flag_ends_the_session_and_is_consumed(tmp_path):
    flag = tmp_path / native.STOP_FLAG_NAME
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    runner = native.Runner(args, detector=None, vision=None, log=lambda line: None,
                           stop_flag=flag)
    assert runner.limit_reached() is False
    flag.write_text("stop", encoding="utf-8")
    assert runner.limit_reached() is True
    assert not flag.exists()


def test_a_stop_asked_while_the_runner_was_loading_is_kept(tmp_path):
    import os
    flag = tmp_path / native.STOP_FLAG_NAME
    flag.write_text("old", encoding="utf-8")
    os.utime(flag, (100.0, 100.0))
    native.clear_stale_stop_flag(flag, started=200.0)
    assert not flag.exists()                     # left over from an earlier session
    flag.write_text("new", encoding="utf-8")
    os.utime(flag, (300.0, 300.0))
    native.clear_stale_stop_flag(flag, started=200.0)
    assert flag.exists()                         # asked after this one started


def test_only_one_runner_holds_the_emulator(tmp_path):
    first = native.hold_runner_lock(tmp_path)
    assert first is not None
    try:
        assert native.hold_runner_lock(tmp_path) is None
    finally:
        first.close()
    again = native.hold_runner_lock(tmp_path)
    assert again is not None
    again.close()


def beatdown_runner():
    from beatdown_strategy import BeatdownStrategy
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    return native.Runner(args, detector=None, vision=None, log=lambda line: None,
                         strategy=BeatdownStrategy())


def test_beatdown_runner_drops_the_giant_behind_the_king_at_full_elixir():
    runner = beatdown_runner()
    state = make_state([Cards.KNIGHT, Cards.GIANT, Cards.ARCHERS, Cards.ARROWS], elixir=10)
    action, score, reason = runner.choose(state, now=0.0)
    assert action.CARD == Cards.GIANT and action.index == 1
    assert (action.tile_x, action.tile_y) == (8, 0)
    assert native.deploy_tile(action, state) == (8, 0)
    assert "Géant au fond" in reason


def test_beatdown_runner_still_fires_spells_on_a_cluster_first():
    runner = beatdown_runner()
    cluster = [enemy(8, 20), enemy(8, 20), enemy(8, 20)]
    state = make_state([Cards.GIANT, Cards.FIREBALL], cluster, elixir=10)
    action, _, reason = runner.choose(state, now=0.0)
    assert action.CARD == Cards.FIREBALL
    assert reason.startswith("Sort")


def test_a_spell_is_cast_on_a_real_cluster():
    cluster = [enemy(8, 20), enemy(8, 20), enemy(8, 20)]
    action, score = native.choose_native(make_state([Cards.FIREBALL], cluster))
    assert action is not None and action.CARD == Cards.FIREBALL
    assert score[0] == 1


class _HoldingStrategy:
    """A strategy that casts its own spells (the trained policy) and holds."""

    handles_spells = True
    name = "ia"

    def __init__(self):
        self.started = []

    def start_match(self, now):
        self.started.append(now)

    def decide(self, state, now):
        from beatdown_strategy import Plan
        return Plan(phase="ia", reason="IA : j'attends")

    def played(self, card, tile, now):
        pass


def test_a_policy_that_casts_its_own_spells_is_not_overruled():
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    strategy = _HoldingStrategy()
    runner = native.Runner(args, detector=None, vision=None, log=lambda line: None,
                           strategy=strategy)
    cluster = [enemy(8, 20), enemy(8, 20), enemy(8, 20)]
    action, _, reason = runner.choose(make_state([Cards.FIREBALL], cluster), now=0.0)
    assert action is None and "IA" in reason


def test_the_strategy_hears_when_a_match_starts(monkeypatch):
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    strategy = _HoldingStrategy()
    runner = native.Runner(args, detector=None, vision=None, log=lambda line: None,
                           strategy=strategy)
    monkeypatch.setattr(native.guarded, "home_guard", lambda *a: False)
    monkeypatch.setattr(runner, "on_battle", lambda *a: None)
    state = make_state([Cards.KNIGHT])
    state.screen = SimpleNamespace(name="in_game")
    for now in (1.0, 1.1, 1.2):
        runner.handle(None, state, now)
    assert strategy.started == [1.2]


def test_no_decision_before_the_battle_is_confirmed(monkeypatch):
    """The first two in-game frames still carry the last match's clock."""
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    strategy = _HoldingStrategy()
    decided = []
    strategy.decide = lambda state, now: decided.append(now) or _HoldingStrategy.decide(
        strategy, state, now)
    runner = native.Runner(args, detector=None, vision=None, log=lambda line: None,
                           strategy=strategy)
    monkeypatch.setattr(native.guarded, "home_guard", lambda *a: False)
    monkeypatch.setattr(native.guarded, "battle_guard", lambda *a: True)
    monkeypatch.setattr(native.time, "sleep", lambda *a: None)
    monkeypatch.setattr(runner.perception, "perceive", lambda *a: None)
    state = make_state([Cards.KNIGHT])
    state.screen = SimpleNamespace(name="in_game")
    for now in (101.0, 101.1, 101.2):               # well past the gap between plays
        runner.handle(None, state, now)
    assert decided == [101.2]


class _FakeRecorder:
    def __init__(self):
        self.frames, self.plays = [], []

    def record(self, frame, perception, now, match_started):
        self.frames.append((now, match_started))
        return True

    def played(self, card, tile, now):
        self.plays.append((card, tile, now))


def test_battle_frames_are_recorded_for_the_dataset_once_the_match_is_confirmed(monkeypatch):
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    recorder = _FakeRecorder()
    runner = native.Runner(args, detector=None, vision=None, log=lambda line: None,
                           strategy=_HoldingStrategy(), recorder=recorder)
    monkeypatch.setattr(native.guarded, "home_guard", lambda *a: False)
    monkeypatch.setattr(native.guarded, "battle_guard", lambda *a: True)
    monkeypatch.setattr(native.time, "sleep", lambda *a: None)
    monkeypatch.setattr(runner.perception, "perceive", lambda *a: None)
    state = make_state([Cards.KNIGHT])
    state.screen = SimpleNamespace(name="in_game")
    for now in (101.0, 101.1, 101.2, 102.5):
        runner.handle(object(), state, now)
    assert recorder.frames == [(101.2, 101.2), (102.5, 101.2)]


def test_an_end_screen_mistaken_for_the_lobby_still_gives_the_result(monkeypatch):
    """BuildABot read a result screen as the lobby: the match was counted
    'inconnu' and the result lost. A blue OK button says it is not the lobby."""
    from PIL import Image
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    runner = native.Runner(args, detector=None, vision=None, log=lambda line: None)
    monkeypatch.setattr(native.time, "sleep", lambda *a: None)
    runner.in_battle = True
    win = Image.open(ROOT / "tests" / "fixtures" / "screens" / "end_of_game_win.png").convert("RGB")
    runner.on_lobby(now=100.0, full=win)
    assert runner.finished == 1 and runner.record["victoire"] == 1


def test_a_real_lobby_after_a_match_is_still_counted_unknown(monkeypatch):
    from PIL import Image
    args = SimpleNamespace(adb="adb", serial="s", max_matches=None, no_queue=True, hours=1)
    runner = native.Runner(args, detector=None, vision=None, log=lambda line: None)
    monkeypatch.setattr(native.time, "sleep", lambda *a: None)
    runner.in_battle = True
    lobby = Image.open(ROOT / "tests" / "fixtures" / "screens" / "lobby.png").convert("RGB")
    runner.on_lobby(now=100.0, full=lobby.resize((1080, 1920)))
    assert runner.finished == 1 and runner.record["inconnu"] == 1


def test_the_pekka_deck_preset_is_the_deck_in_play_since_2026_10_31():
    """The user swapped Giant + Fireball for P.E.K.K.A + Inferno Tower."""
    native.use_deck("pekka")
    assert sorted(card.name for card in native.DECK) == sorted(
        ["knight", "pekka", "minions", "arrows", "inferno_tower", "archers", "musketeer", "minipekka"])
