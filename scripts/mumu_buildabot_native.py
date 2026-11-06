from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ClashRoyaleBuildABot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mumu_overnight_bot as guarded  # noqa: E402
from clashroyalebuildabot.actions import (  # noqa: E402
    ArchersAction,
    ArrowsAction,
    BabyDragonAction,
    CannonAction,
    FireballAction,
    GiantAction,
    GoblinBarrelAction,
    KnightAction,
    MinionsAction,
    MinipekkaAction,
    MusketeerAction,
    WitchAction,
)
from emulator import DEFAULT_SERIAL, verify as verify_emulator  # noqa: E402
from beatdown_strategy import BeatdownStrategy  # noqa: E402
from live_perception import configure_unit_detection  # noqa: E402
from live_view_state import ViewWriter, snapshot  # noqa: E402
from match_result import banner_outcome  # noqa: E402
from ok_button import find_ok_button  # noqa: E402
from native_perception import Perception  # noqa: E402
from unknown_screen import UnknownScreenHandler  # noqa: E402
from clashroyalebuildabot.constants import (  # noqa: E402
    ALL_TILES,
    ALLY_TILES,
    DISPLAY_HEIGHT,
    LEFT_PRINCESS_TILES,
    RIGHT_PRINCESS_TILES,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)
from clashroyalebuildabot.detectors.detector import Detector  # noqa: E402
from clashroyalebuildabot.actions.generic.bridge_action import BridgeAction  # noqa: E402
from clashroyalebuildabot.actions.generic.defense_action import DefenseAction  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402


# Upstream has no action for these two: P.E.K.K.A is scored like the Mini
# P.E.K.K.A (a melee tank at the bridge or on defence), the Inferno Tower like
# the Cannon (a defensive building).
class PekkaAction(BridgeAction):
    CARD = Cards.PEKKA


class InfernoTowerAction(DefenseAction):
    CARD = Cards.INFERNO_TOWER


# The card reader only recognises the cards it is given, so the preset has to
# match the deck actually equipped in game - otherwise every slot reads as a
# wrong card or "blank" and the battle guard blocks every play.
DECK_PRESETS = {
    "default": (
        ArchersAction,
        GoblinBarrelAction,
        BabyDragonAction,
        CannonAction,
        KnightAction,
        MinipekkaAction,
        MusketeerAction,
        WitchAction,
    ),
    # What a fresh account is handed after the training camp.
    "starter": (
        KnightAction,
        ArchersAction,
        MinionsAction,
        ArrowsAction,
        FireballAction,
        GiantAction,
        MusketeerAction,
        MinipekkaAction,
    ),
    # The user's deck since 2026-10-31: P.E.K.K.A + Inferno Tower in place of
    # Giant + Fireball.
    "pekka": (
        KnightAction,
        PekkaAction,
        MinionsAction,
        ArrowsAction,
        InfernoTowerAction,
        ArchersAction,
        MusketeerAction,
        MinipekkaAction,
    ),
}
ACTIONS = DECK_PRESETS["default"]
DECK = [action.CARD for action in ACTIONS]
CARDS_TO_ACTIONS = {action.CARD: action for action in ACTIONS}
YOLO_CONFIDENCE = 0.2           # --vision yolo: detection threshold (see build_vision)


def use_deck(preset: str) -> None:
    global ACTIONS, DECK, CARDS_TO_ACTIONS
    ACTIONS = DECK_PRESETS[preset]
    DECK = [action.CARD for action in ACTIONS]
    CARDS_TO_ACTIONS = {action.CARD: action for action in ACTIONS}


def tile_centre(tile_x: int, tile_y: int) -> tuple[int, int]:
    x720 = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
    y1280 = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
    return round(x720 * 1.5), round(y1280 * 1.5)


# Upstream's BridgeAction (Mini P.E.K.K.A) and GiantAction score only these two
# tiles - on the river, deployable only once that lane's princess tower has
# fallen. They were never in the candidate tiles before that, so in upstream
# those cards were simply never played: the bot sat on 10 elixir with a Giant
# in hand. They are evaluated here and deployed at the foot of the bridge.
BRIDGE_TILES = ((3, 15), (14, 15))
LAST_ALLY_ROW = 14
SCREEN_LABELS = {"lobby": "salon", "end_of_game": "fin de match",
                 "bypass_end_of_game": "fin de match", "unknown": "écran inconnu",
                 "in_game": "match"}


@dataclass(frozen=True)
class PlannedAction:
    """A strategy's choice, shaped like a BuildABot action for the tap path."""
    index: int
    CARD: object
    tile_x: int
    tile_y: int
# Consecutive `in_game` frames before a match counts as played. From the
# screen, not from the battle guard: a guard that blocks a whole match (a deck
# preset that does not match the game) must still end in a counted result.
IN_GAME_FRAMES_FOR_MATCH = 3
# Minimum time between two plays: the card animation and the elixir bar need
# this long to settle before the next read is trusted.
PLAY_GAP_SECONDS = 2.0
# Created by the panel's "arrêter après la partie": checked at each match end.
STOP_FLAG_NAME = "stop_after_match.flag"
# Held for the whole session: two runners on one emulator fight for every tap.
LOCK_NAME = "runner.lock"
_IMPORTED_AT = time.time()


def process_started() -> float:
    """When this process was created (import time if psutil cannot say)."""
    try:
        import psutil
        return psutil.Process().create_time()
    except Exception:
        return _IMPORTED_AT


def clear_stale_stop_flag(flag: Path, started: float) -> None:
    """Drop a stop left over from an earlier session, but keep one asked for
    while this runner was still loading its models (10-30 s)."""
    try:
        if flag.stat().st_mtime < started:
            flag.unlink()
    except FileNotFoundError:
        pass


def hold_runner_lock(folder: Path):
    """An open handle holding the session lock, or None if another runner has it.

    The OS releases the lock when the process dies, killed or not, so a crash
    never leaves the emulator locked.
    """
    folder.mkdir(parents=True, exist_ok=True)
    handle = open(folder / LOCK_NAME, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _tower_down(state, tile_x: int) -> bool:
    numbers = state.numbers
    hp = numbers.left_enemy_princess_hp if tile_x < 9 else numbers.right_enemy_princess_hp
    return hp.number == 0


def deploy_tile(action, state) -> tuple[int, int]:
    """Where to actually place `action`: a bridge tile becomes the last row on
    our side while the princess tower in that lane still stands."""
    tile = (action.tile_x, action.tile_y)
    if tile in BRIDGE_TILES and not _tower_down(state, action.tile_x):
        return (action.tile_x, LAST_ALLY_ROW)
    return tile


SPELL_CARDS = frozenset({"arrows", "fireball", "zap", "the_log", "goblin_barrel"})


def choose_native(state, exclude=frozenset(), only=None):
    """BuildABot's own scoring; `exclude`/`only` restrict which cards it may use."""
    valid_tiles = list(ALLY_TILES)
    if state.numbers.left_enemy_princess_hp.number == 0:
        valid_tiles += LEFT_PRINCESS_TILES
    if state.numbers.right_enemy_princess_hp.number == 0:
        valid_tiles += RIGHT_PRINCESS_TILES
    known = {tuple(tile) for tile in valid_tiles}
    valid_tiles += [list(tile) for tile in BRIDGE_TILES if tile not in known]

    best_score = [0]
    best_action = None
    for slot in state.ready:
        card = state.cards[slot + 1]
        action_type = CARDS_TO_ACTIONS.get(card)
        if action_type is None or state.numbers.elixir.number < card.cost:
            continue
        if card.name in exclude or (only is not None and card.name not in only):
            continue
        tiles = ALL_TILES if card.target_anywhere else valid_tiles
        for tile_x, tile_y in tiles:
            candidate = action_type(slot, tile_x, tile_y)
            score = candidate.calculate_score(state)
            if score > best_score:
                best_score = score
                best_action = candidate
    # Upstream's "no good actions available": a leading 0 means the action
    # achieves nothing. Without this a spell hitting nobody ([0, 0, inf])
    # beats the initial [0] and is cast at an empty tile - in practice onto
    # the bot's own side, in front of its King Tower.
    if best_score[0] == 0:
        return None, best_score
    return best_action, best_score


def validate_battle(detector, adb_path, serial, slot=None, card_name=None, count=2):
    latest = None
    for _ in range(count):
        full, state = guarded.detect(detector, adb_path, serial)
        if state is None or not guarded.battle_guard(state):
            return None
        if slot is not None and state.cards[slot + 1].name != card_name:
            return None
        latest = (full, state)
        time.sleep(0.22)
    return latest


FAST_CAPTURE_SIZE = "540x960"
LIVE_LOG = ROOT / "tmp" / "live" / "buildabot-native.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded native BuildABot policy on MuMu")
    parser.add_argument("--adb", required=True, type=Path)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--deck", choices=sorted(DECK_PRESETS), default="default",
                        help="must match the deck equipped in game")
    parser.add_argument("--max-matches", type=int,
                        help="stop once this many matches have been played to the end "
                             "(counted from battle to result screen, queued or not)")
    parser.add_argument("--no-queue", action="store_true",
                        help="never press Battle: play only matches started by hand, "
                             "e.g. a friendly 1v1")
    parser.add_argument("--vision", choices=("auto", "yolo", "buildabot"), default="auto",
                        help="units from the YOLO detector trained by scripts/train_detector.py "
                             "(auto: when its weights exist), or BuildABot's own model")
    parser.add_argument("--fast-capture", action="store_true",
                        help="hold the device at 540x960 while running (4x lighter "
                             "screencaps); reset on a normal exit - after a forced "
                             "kill, scripts/climb.ps1 -Stop resets it")
    parser.add_argument("--rl-checkpoint", type=Path,
                        default=ROOT / "tmp" / "rl" / "starter_long_best.pt",
                        help="with --strategy ia: a checkpoint trained on this deck "
                             "(sim.train_ppo --deck)")
    parser.add_argument("--strategy", choices=("buildabot", "beatdown", "pekka", "pekka2", "pekka2c",
                                               "pekka_hier", "ia", "hybride", "hybride_sorts"),
                        default="buildabot",
                        help="buildabot = upstream's per-card rules; beatdown = bank to 10, "
                             "Giant behind the King Tower, support troops behind it")
    parser.add_argument("--record-dataset", type=Path, default=None,
                        help="keep a battle frame a second with what was seen and played "
                             "(scripts/autolabel.py turns them into detector data)")
    parser.add_argument("--no-view", action="store_true",
                        help="do not publish tmp/live/native_view.json for the HUD")
    parser.add_argument("--capture", choices=("auto", "window", "adb"), default="auto",
                        help="window = grab MuMu's render window on the PC (~9 ms, no "
                             "device load); adb = screencap; auto = window when found")
    parser.add_argument("--instance", default=os.environ.get("HASTYCR_INSTANCE", "Android Device"),
                        help="title of the MuMu window to grab with --capture window")
    parser.add_argument("--log", type=Path, default=LIVE_LOG)
    return parser.parse_args()


def build_vision(choice: str, log):
    """YoloVision when asked for (or available under auto), else None."""
    if choice == "buildabot":
        log("VISION buildabot (requested)")
        return None
    from brain.vision import YoloVision
    # 0.2: on the checked frames (scripts/eval_truth.py) it finds 83% of the
    # units with 2% ghosts, against 74% and none at the old 0.40.
    vision = YoloVision(confidence=YOLO_CONFIDENCE)
    if vision.ready:
        sides = vision.sides.path.name if vision.sides.ready else "NO"
        log(f"VISION yolo weights={vision.weights.parent.parent.name} conf={YOLO_CONFIDENCE} "
            f"side_classifier={sides} ({vision.sides.crop}px)")
        return vision
    log(f"VISION yolo unavailable ({vision.error}); staying on buildabot")
    return None


class Runner:
    """One session: lobby -> battle -> result -> lobby, plus unknown screens."""

    def __init__(self, args, detector, vision, log, upstream_units=None,
                 strategy=None, view=None, stop_flag=None, results_dir=None,
                 recorder=None):
        self.args, self.detector, self.log = args, detector, log
        # End-of-game frames, kept so a logged victory or defeat can be checked.
        self.results_dir = results_dir
        # Battle frames + what was seen and played, to grow the detector's
        # dataset (scripts/dataset_recorder.py, scripts/autolabel.py).
        self.recorder = recorder
        self.recorder_failed = False
        self.match_started_at: Optional[float] = None
        self.record = {"victoire": 0, "défaite": 0, "inconnu": 0}
        self.strategy = strategy            # Beatdown / RL strategy, or None for plain BuildABot
        self.view = view                    # ViewWriter for the live view, or None
        # Detect -> track -> sides -> enemy elixir; owns the YOLO fallback too.
        # A strategy may ask to see enemies from their first frame (pekka2c).
        min_hits = getattr(strategy, "enemy_min_hits", None)
        self.perception = Perception(DECK, vision, detector, upstream_units, log,
                                     own_structures=getattr(strategy, "own_structures", True),
                                     **({"enemy_min_hits": min_hits, "deep_born_min_hits": min_hits}
                                        if min_hits else {}))
        # A file the panel creates to ask for a stop once the match is over.
        self.stop_flag = stop_flag
        # Once asked, a stop stays asked: the flag is deleted when read, and a
        # failed tap right after must not let the lobby queue another match.
        self.stop_requested = False
        self.plan_phase, self.plan_reason = "", "démarrage"
        self.last_state = None
        self.last_target = None
        self.recent_plays: list[str] = []
        self.view_failed = False
        self.last_action = self.last_nav = 0.0
        self.actions = self.matches = self.finished = 0
        # Set once a battle is seen, cleared when it is counted as finished, so a
        # match counts once however many times its result screen is tapped - and
        # a missed Battle tap never counts as a match played.
        self.in_battle = False
        self.in_game_frames = 0
        self.last_idle_log = 0.0
        self.unknown = UnknownScreenHandler(
            tap=lambda xy: guarded.tap(args.adb, args.serial, xy),
            relaunch=lambda: guarded.adb(args.adb, args.serial, "shell", "monkey", "-p",
                                         "com.supercell.clashroyale", "-c",
                                         "android.intent.category.LAUNCHER", "1"),
            log=log, before_confirm=self.on_confirm_screen)

    def limit_reached(self) -> bool:
        """Checked at a match's end and in the lobby: the match cap, or a stop
        the panel asked for ("arrêter après la partie")."""
        if self.stop_requested:
            return True
        if self.stop_flag is not None and self.stop_flag.exists():
            try:
                self.stop_flag.unlink()
            except OSError:
                pass
            self.log("STOP requested from the panel, no match in progress")
            self.stop_requested = True
        elif self.args.max_matches is not None and self.finished >= self.args.max_matches:
            self.stop_requested = True
        return self.stop_requested

    def count_finished(self, screen: str, outcome: str = "inconnu") -> None:
        self.in_battle = False
        self.finished += 1
        self.record[outcome] = self.record.get(outcome, 0) + 1
        self.log(f"RESULT exit screen={screen} finished={self.finished} outcome={outcome} "
                 f"record={self.record['victoire']}V-{self.record['défaite']}D")

    def on_confirm_screen(self, full) -> None:
        """An OK button on a screen BuildABot cannot name, about to be pressed.

        Straight after a match it is the result screen in a variant the
        detector misses (seen live: the match then ended 'inconnu' in the
        lobby). Read the crowns before OK takes it away.
        """
        if not self.in_battle:
            return
        outcome = banner_outcome(full)
        if outcome == "inconnu":
            return
        self.keep_result_frame(full, outcome)
        self.count_finished("result_screen", outcome)
        self.limit_reached()        # a stop asked for now is kept for the lobby

    def record_frame(self, full, now: float) -> None:
        """A battle frame for the dataset; never allowed to stop the bot."""
        if self.recorder is None or not self.in_battle or full is None:
            return
        try:
            self.recorder.record(full, self.perception, now, self.match_started_at or now)
        except Exception as exc:
            if not self.recorder_failed:
                self.recorder_failed = True
                self.log(f"DATASET frame not saved: {type(exc).__name__}: {exc}")

    def keep_result_frame(self, full, outcome: str) -> None:
        if self.results_dir is None or full is None:
            return
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            name = f"{datetime.now():%Y%m%d-%H%M%S}_{outcome}.jpg"
            full.convert("RGB").resize((540, 960)).save(self.results_dir / name, quality=70)
        except Exception as exc:                 # a lost picture must not stop the bot
            self.log(f"RESULT frame not saved: {type(exc).__name__}: {exc}")

    def run(self) -> int:
        deadline = time.monotonic() + self.args.hours * 3600
        while time.monotonic() < deadline:
            try:
                if self.step():
                    return 0
            except Exception as exc:
                self.log(f"ERROR {type(exc).__name__}: {exc}")
                time.sleep(2)
        self.log(f"STOP deadline matches={self.matches} actions={self.actions}")
        return 0

    def step(self) -> bool:
        """One detection and the action it calls for. True means stop."""
        args = self.args
        full, state = guarded.detect(self.detector, args.adb, args.serial)
        if state is None:
            time.sleep(1)
            return False
        now = time.monotonic()
        try:
            return self.handle(full, state, now)
        finally:
            self.publish(full, state, now)

    def publish(self, full, state, now: float) -> None:
        """Feed the live view; a failure here must never stop the bot."""
        if self.view is None or full is None:
            return
        screen = state.screen.name
        phase = self.plan_phase if screen == "in_game" else SCREEN_LABELS.get(screen, screen)
        reason = self.plan_reason if screen == "in_game" else f"Écran : {phase}"
        try:
            data = snapshot(state, phase=phase, reason=reason, recent=self.recent_plays,
                            target=self.last_target,
                            strategy=getattr(self.strategy, "name", "beatdown")
                            if self.strategy else "buildabot",
                            finished=self.finished, actions=self.actions,
                            enemy_elixir=(self.perception.enemy.value if self.in_battle
                                          else None),
                            tile_to_px=tile_centre,
                            # Perception keeps every box in the 1080x1920 frame.
                            box_scale=1.0, record=self.record)
            self.view.write(data, now)
        except Exception as exc:
            if not self.view_failed:
                self.view_failed = True
                self.log(f"VIEW error {type(exc).__name__}: {exc}")

    def handle(self, full, state, now: float) -> bool:
        screen = state.screen.name
        self.in_game_frames = self.in_game_frames + 1 if screen == "in_game" else 0
        if self.in_game_frames >= IN_GAME_FRAMES_FOR_MATCH and not self.in_battle:
            self.in_battle = True
            self.perception.start_match(now)  # tracks forgotten, opponent back on 5
            self.match_started_at = now
            if hasattr(self.strategy, "start_match"):
                self.strategy.start_match(now)    # the policy's match clock
        if screen == "unknown":
            self.unknown.handle(full, now)
            time.sleep(0.3)
            return False
        self.unknown.known()
        if guarded.home_guard(full, state) and now - self.last_nav > 8:
            return self.on_lobby(now, full)
        if screen in {"end_of_game", "bypass_end_of_game"} and now - self.last_nav > 5:
            return self.on_result(state, full, screen, now)
        self.on_battle(state, full, now)
        return False

    def on_lobby(self, now: float, full=None) -> bool:
        args = self.args
        if self.in_battle:
            # BuildABot sometimes reads a result screen as the lobby; a blue
            # OK button at the bottom says it is not, and the crowns are there.
            if full is not None and find_ok_button(full) is not None:
                outcome = banner_outcome(full)
                self.keep_result_frame(full, outcome)
                self.count_finished("result_as_lobby", outcome)
            else:
                self.count_finished("lobby")  # back without a recognised result screen
        # Back in the lobby after the last allowed match: stop here rather
        # than queue one more.
        if self.limit_reached():
            self.log(f"STOP max-matches finished={self.finished} actions={self.actions}")
            return True
        if not args.no_queue:
            full2, state2 = guarded.detect(self.detector, args.adb, args.serial)
            if state2 is not None and guarded.home_guard(full2, state2):
                guarded.tap(args.adb, args.serial, guarded.HOME_BATTLE)
                self.last_nav = now
                self.matches += 1
                self.log(f"QUEUE match={self.matches}")
        time.sleep(1)
        return False

    def on_result(self, state, full, screen: str, now: float) -> bool:
        """Record the result, then leave through OK - never "Rejouer".

        BuildABot's click point on this screen is "Rejouer", which queues the
        next match at once. Tapped before the stop check, a stop asked for
        after the match queued one nobody played (and lost). OK goes back to
        the lobby, where on_lobby decides whether to queue.
        """
        args = self.args
        _, state2 = guarded.detect(self.detector, args.adb, args.serial)
        if state2 is None or state2.screen.name != screen:
            time.sleep(1)
            return False
        stop = False
        if self.in_battle:
            outcome = banner_outcome(full)
            self.keep_result_frame(full, outcome)
            self.count_finished(screen, outcome)
            stop = self.limit_reached()
        else:
            self.log(f"RESULT tap screen={screen}")
        guarded.tap(args.adb, args.serial, guarded.ok_button(state.screen))
        self.last_nav = now
        if stop:
            self.log(f"STOP max-matches finished={self.finished} actions={self.actions}")
            return True
        time.sleep(1)
        return False

    def log_idle(self, state, now: float, every: float = 8.0) -> None:
        """Why nothing was played: BuildABot acts only on enemies or at 10 elixir,
        so a silent stretch is either an unseen board or a misread elixir bar."""
        if now - self.last_idle_log < every:
            return
        self.last_idle_log = now

        def units(group) -> str:
            return ",".join(f"{d.unit.name}@{d.position.tile_x:.0f}/{d.position.tile_y:.0f}"
                            for d in group[:6]) or "-"

        hand = [card.name for card in state.cards[1:5]]
        self.log(f"IDLE elixir={state.numbers.elixir.number} hand={hand} ready={list(state.ready)} "
                 f"enemies={len(state.enemies)}[{units(state.enemies)}] "
                 f"allies={len(state.allies)}[{units(state.allies)}] why={self.plan_reason}")

    def choose(self, state, now: float):
        """(action, score, reason) under the configured strategy."""
        if self.strategy is None:
            action, score = choose_native(state)
            return action, score, "règles BuildABot"
        # Spells on a real cluster whatever the phase: a good trade now is
        # worth more than banking for the push.
        # A trained policy casts its own spells: the cluster rule would overrule it.
        handles_spells = getattr(self.strategy, "handles_spells", False)
        action, score = (None, [0]) if handles_spells else choose_native(state, only=SPELL_CARDS)
        if action is not None:
            self.plan_reason = f"Sort : {action.CARD.name} sur un groupe d'ennemis"
            return action, score, self.plan_reason
        plan = self.strategy.decide(state, now)
        self.plan_reason, self.plan_phase = plan.reason, plan.phase
        if plan.delegate:
            action, score = choose_native(state, exclude=plan.exclude)
            if action is not None:
                return action, score, plan.reason
            plan = plan.then
            self.plan_reason = f"{self.plan_reason} - rien d'utile, {plan.reason}"
        if plan.card is None:
            return None, [0], plan.reason
        planned = PlannedAction(index=plan.slot, CARD=state.cards[plan.slot + 1],
                                tile_x=plan.tile[0], tile_y=plan.tile[1])
        return planned, [1], plan.reason

    def on_battle(self, state, full, now: float) -> None:
        # Short sleeps on purpose: every pass refreshes what the HUD shows and
        # what the bot knows. With window capture a pass is ~0.1 s; the old
        # 0.4-0.7 s naps were sized for a 120 ms screencap and left the view a
        # second behind the troops.
        args = self.args
        if not guarded.battle_guard(state):
            time.sleep(0.1)
            return
        self.perception.perceive(state, full, now)
        self.last_state = state
        self.record_frame(full, now)
        # Publish before acting: a play (validation + two taps) takes ~0.5 s,
        # and boxes published after it would already be that far behind.
        self.publish(full, state, now)
        if not self.in_battle:
            # The first in-game frames, before the match is confirmed: a
            # strategy's clock (and a trained policy's towers) still belong
            # to the last match.
            time.sleep(0.03)
            return
        if now - self.last_action < PLAY_GAP_SECONDS:
            time.sleep(0.02)
            return
        action, score, reason = self.choose(state, now)
        if action is None:
            self.log_idle(state, now)
            time.sleep(0.03)
            return
        card_name = action.CARD.name
        if validate_battle(self.detector, args.adb, args.serial, action.index,
                           card_name, count=2) is None:
            self.log("BLOCK pre-card battle validation failed")
            time.sleep(1)
            return
        guarded.tap(args.adb, args.serial, guarded.CARD_CENTRES[action.index])
        time.sleep(0.15)
        if validate_battle(self.detector, args.adb, args.serial, count=1) is None:
            self.log("BLOCK pre-target battle validation failed")
            time.sleep(1)
            return
        tile_x, tile_y = deploy_tile(action, state)
        target = tile_centre(tile_x, tile_y)
        guarded.tap(args.adb, args.serial, target)
        self.actions += 1
        self.last_action = time.monotonic()
        self.perception.played(card_name, now, (tile_x, tile_y))
        if self.recorder is not None:
            self.recorder.played(card_name, (tile_x, tile_y), now)
        if self.strategy is not None:
            self.strategy.played(card_name, (tile_x, tile_y), now)
        self.last_target = target
        self.recent_plays = ([f"{datetime.now():%H:%M:%S} {card_name} ({tile_x},{tile_y}) - {reason}"]
                             + self.recent_plays)[:8]
        self.log(f"PLAY #{self.actions} {card_name} tile=({tile_x},{tile_y}) "
                 f"target={target} score={score} elixir={state.numbers.elixir.number} "
                 f"enemy_elixir={self.perception.enemy.value:.1f} why={reason}")
        time.sleep(0.2)


def main() -> int:
    args = parse_args()
    # Same guard as cr_bot.py: refuse to tap anything but the Clash Royale
    # instance. This runner used to skip it, and it is the panel's default bot.
    verify_emulator(args.serial, args.adb)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    lock = hold_runner_lock(args.log.parent)
    if lock is None:
        raise SystemExit("another native runner is already playing on this machine "
                         "(tmp/live/runner.lock is held); stop it first")

    def log(message: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {message}"
        print(line, flush=True)
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    use_deck(args.deck)
    # BuildABot's ONNX models fall back to the CPU here and took 22 of 24
    # cores: a few non-spinning threads are faster and leave the PC usable.
    import lean_onnx
    lean_onnx.install()
    detector = Detector(DECK)
    upstream_units = detector.unit_detector
    vision = build_vision(args.vision, log)
    configure_unit_detection(detector, vision)
    vision_name = "yolo" if vision else "buildabot"
    strategy = None
    if args.strategy == "beatdown":
        strategy = BeatdownStrategy()
    elif args.strategy == "pekka":
        # The user's P.E.K.K.A game plan as rules (scripts/pekka_strategy.py).
        from pekka_strategy import PekkaStrategy
        strategy = PekkaStrategy()
    elif args.strategy == "pekka2":
        # The same plan with a counter per threat (Wizard, Valkyrie, dragons...).
        from pekka_strategy import PekkaCountersStrategy
        strategy = PekkaCountersStrategy()
    elif args.strategy == "pekka_hier":
        # The rules as they were on 2026-11-04, for an A/B against today's.
        from pekka_strategy import PekkaYesterdayStrategy
        strategy = PekkaYesterdayStrategy()
    elif args.strategy == "pekka2c":
        # pekka2 with the 2026-11-05 defence additions off, for an A/B.
        from pekka_strategy import PekkaClassicStrategy
        strategy = PekkaClassicStrategy()
    elif args.strategy in ("ia", "hybride", "hybride_sorts"):
        from native_rl import RLStrategy
        policy = RLStrategy(args.rl_checkpoint, [card.name for card in DECK])
        log(f"IA checkpoint={args.rl_checkpoint.name} deck={policy.deck}")
        strategy = policy
        if args.strategy in ("hybride", "hybride_sorts"):
            # The policy attacks, the beatdown defence answers what crosses;
            # hybride_sorts also lets the cluster rule cast Fireball/Arrows.
            from hybrid_strategy import HybridStrategy
            rules = BeatdownStrategy                  # module-level import: no local rebinding
            if args.deck == "pekka":
                # The user's P.E.K.K.A rules bind the policy too.
                from pekka_strategy import PekkaStrategy
                rules = PekkaStrategy
            strategy = HybridStrategy(policy, cluster_spells=args.strategy == "hybride_sorts",
                                      rules_factory=rules)
            log(f"hybride: règles {rules.__name__}")
    view = None if args.no_view else ViewWriter(args.log.parent)
    # A stop asked for in an earlier session must not end this one at once.
    stop_flag = args.log.parent / STOP_FLAG_NAME
    clear_stale_stop_flag(stop_flag, process_started())
    capture = "adb"
    if args.capture != "adb":
        problem = guarded.use_window_capture(args.instance)
        if not problem:
            capture = "window"
        elif args.capture == "window":
            raise SystemExit(f"window capture unavailable: {problem}")
        else:
            log(f"CAPTURE window unavailable ({problem}); using adb screencap")
    log(f"START native BuildABot scoring with guarded MuMu controls "
        f"deck={args.deck} serial={args.serial} vision={vision_name} "
        f"strategy={args.strategy} capture={capture}")
    # Shrinking the device only speeds up screencap; with window capture it
    # would just blur what is grabbed.
    resized = args.fast_capture and capture == "adb" and set_fast_capture(args, log)
    try:
        recorder = None
        if args.record_dataset:
            from dataset_recorder import DatasetRecorder
            recorder = DatasetRecorder(args.record_dataset)
            log(f"DATASET recording to {args.record_dataset}")
        return Runner(args, detector, vision, log, upstream_units, strategy, view,
                      stop_flag=stop_flag, results_dir=args.log.parent / "results",
                      recorder=recorder).run()
    finally:
        if resized:
            reset_display(args, log)


def _wm_size(args, *size: str) -> str:
    """`wm size [...]` with a timeout: a hung adb must not hang the exit too."""
    result = subprocess.run([str(args.adb), "-s", args.serial, "shell", "wm", "size", *size],
                            capture_output=True, text=True, timeout=15)
    return result.stdout


def set_fast_capture(args, log) -> bool:
    """Drop to 540x960 unless something else already forced a size.

    Only a size this runner set is reset on exit, so a size the panel or
    run.ps1 chose is left alone. A forced kill skips `finally`: the panel and
    scripts/climb.ps1 -Stop reset the size themselves.
    """
    if "Override size" in _wm_size(args):
        log("DISPLAY already overridden; leaving it as it is")
        return False
    _wm_size(args, FAST_CAPTURE_SIZE)
    time.sleep(2)     # a capture taken mid-switch would scale taps by 2x
    log(f"DISPLAY {FAST_CAPTURE_SIZE} for capture speed (reset on exit)")
    return True


def reset_display(args, log) -> None:
    try:
        _wm_size(args, "reset")
        log("DISPLAY reset")
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"DISPLAY reset failed {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
