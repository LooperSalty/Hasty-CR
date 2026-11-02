"""The P.E.K.K.A rules play a whole simulated match through RulesSeat."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from pekka_strategy import PekkaStrategy  # noqa: E402
from sim.gamedata import load_gamedata  # noqa: E402
from sim.match import Match  # noqa: E402
from sim.opponents import ScriptedOpponent  # noqa: E402
from sim.runner import resolve_deck  # noqa: E402
from sim.spells import load_spells  # noqa: E402
from sim_rules_seat import PEKKA_DECK, RulesSeat  # noqa: E402

THEIRS = ["giant", "musketeer", "minions", "arrows", "knight", "archer", "fireball", "minipekka"]


def test_the_pekka_rules_play_a_whole_simulated_match():
    full = load_gamedata(level=11)
    cards = resolve_deck(full, sorted(set(PEKKA_DECK) | set(THEIRS)))
    match = Match(cards=cards, decks=(list(PEKKA_DECK), list(THEIRS)), seed=5,
                  spells=load_spells(level=11))
    bot = RulesSeat(PekkaStrategy())
    them = ScriptedOpponent(cards, side=-1, deck=THEIRS, style="beatdown", seed=6)
    bot.reset()
    them.reset()
    plays = []
    next_decision = 0
    while not match.finished and match.elapsed_ms < 300_000:
        match.step()
        if match.elapsed_ms >= next_decision:
            next_decision = match.elapsed_ms + 500
            played = bot.act(match)
            if played:
                plays.append(played)
            them.act(match)
    pekkas = [(x, y) for card, x, y, _why in plays if card == "pekka"]
    assert pekkas and all(tile in {(8, 0), (9, 0)} for tile in pekkas)
    assert all("Défense" in why for card, _x, _y, why in plays if card == "knight")
