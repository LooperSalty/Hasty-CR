"""Rebuild the opponents' decks the live bot met, for the simulator's `live` pool.

The `meta` pool is Path of Legends: top-ladder decks. At 600-1100 trophies
the bot met other cards, and lost most against some the pool barely holds
(Lumberjack 0 decks of 204, Elite Barbarians 0, Wizard 5, P.E.K.K.A 9). The
runner logs every unit it sees (`IDLE ... enemies=N[name@x/y,...]`) and each
result, so each match gives the opponent's troops; spells are never seen as
units, so the deck is padded with the common ones. Decks that beat us are
written LOSS_WEIGHT times: that is the weight `sim.meta_decks.live_deck_pool`
draws them with.

    python scripts/build_live_decks.py [--since 2026-10-13]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "tmp" / "live" / "buildabot-native.log"
OUT = ROOT / "data" / "live_decks.json"
FIRST_DAY = datetime(2026, 10, 10)          # the log's first line; it carries no dates
MIN_FRAMES = 3                              # fewer is a misread, not a card
MAX_UNITS = 6
LOSS_WEIGHT = 2
SPELLS = ("fireball", "arrows", "zap")
FILLERS = ("knight", "archer", "minions", "skeletons", "goblins", "musketeer", "valkyrie", "bomber")

# BuildABot unit name -> simulator card names to try, first one present wins.
UNIT_TO_CARD = {
    "minion": ("minions",), "skeleton": ("skeletons",), "goblin": ("goblins", "goblin_gang"),
    "spear_goblin": ("spear_goblins",), "barbarian": ("barbarians",),
    "skeleton_dragon": ("skeleton_dragons",), "ice_golem": ("ice_golemite", "ice_golem"),
    "lava_pup": ("lava_hound",), "lumberjack": ("axe_man",), "bandit": ("assassin",),
    "royal_recruit": ("royal_recruits",), "elite_barbarian": ("angry_barbarians",),
    "bat": ("bats",), "rascal_girl": ("rascals",), "phoenix_egg": ("phoenix",),
    "guard": ("skeleton_warriors", "guards"), "wall_breaker": ("wallbreakers",),
    "giant_snowball": ("snowball", "giant_snowball"), "x_bow": ("xbow",),
    "royal_hog": ("royal_hogs",), "fire_spirit": ("fire_spirits",),
    "elixir_golem_small": ("elixir_golem",), "elixir_golem_medium": ("elixir_golem",),
    "elixir_golem_large": ("elixir_golem",), "sparky": ("zap_machine", "sparky"),
    "electro_spirit": ("electro_spirit",), "heal_spirit": ("heal_spirit",),
}
IGNORED = {"hog", "golemite", "royal_guardian", "dagger_duchess_tower"}

TIME = re.compile(r"^(\d\d):(\d\d):(\d\d) ")
ENEMIES = re.compile(r"enemies=\d+\[([^\]]*)\]")
RESULT = re.compile(r"RESULT exit .*outcome=(victoire|défaite)")


def card_for(unit: str, cards: dict) -> str | None:
    if unit in IGNORED:
        return None
    for name in UNIT_TO_CARD.get(unit, (unit, unit + "s")):
        if cards.get(name) is not None:
            return name
    return None


def deck_from_units(seen: Counter, cards: dict) -> list[str]:
    """Eight distinct simulator cards from units seen (name -> frames)."""
    deck: list[str] = []
    for unit, frames in seen.most_common():
        card = card_for(unit, cards)
        if frames >= MIN_FRAMES and card and card not in deck and len(deck) < MAX_UNITS:
            deck.append(card)
    for name in SPELLS + FILLERS:
        if len(deck) == 8:
            break
        if name not in deck and cards.get(name) is not None:
            deck.append(name)
    return deck


def matches_from_log(text: str, since: datetime | None = None):
    """(units seen, outcome) per finished match, in log order."""
    out, seen = [], Counter()
    day, previous = FIRST_DAY, None
    for line in text.splitlines():
        stamp = TIME.match(line)
        if not stamp:
            continue
        seconds = int(stamp.group(1)) * 3600 + int(stamp.group(2)) * 60 + int(stamp.group(3))
        if previous is not None and seconds < previous - 3600:
            day += timedelta(days=1)
        previous = seconds
        if " QUEUE " in line:
            seen = Counter()
        if found := ENEMIES.search(line):
            seen.update({item.split("@")[0] for item in found.group(1).split(",") if "@" in item})
        if result := RESULT.search(line):
            if since is None or day + timedelta(seconds=seconds) >= since:
                out.append((seen, result.group(1)))
            seen = Counter()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", help="YYYY-MM-DD: only matches from that day on")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from sim.gamedata import load_gamedata
    cards = load_gamedata(level=11)
    since = datetime.fromisoformat(args.since) if args.since else None
    matches = matches_from_log(LOG.read_text(encoding="utf-8", errors="replace"), since)
    decks = []
    for seen, outcome in matches:
        deck = deck_from_units(seen, cards)
        if len(deck) == 8:
            decks += [deck] * (LOSS_WEIGHT if outcome == "défaite" else 1)
    OUT.write_text(json.dumps({
        "schema": 1, "source": str(LOG.relative_to(ROOT)), "built_at": datetime.now().isoformat(timespec="seconds"),
        "matches": len(matches), "losses": sum(o == "défaite" for _s, o in matches),
        "loss_weight": LOSS_WEIGHT, "decks": decks}, indent=1), encoding="utf-8")
    common = Counter(card for deck in decks for card in deck).most_common(20)
    print(f"{len(matches)} matches -> {len(decks)} weighted decks in {OUT.relative_to(ROOT)}")
    print("most common cards:", ", ".join(f"{c} {n}" for c, n in common))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
