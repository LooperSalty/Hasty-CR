"""data/unit_stats.json: what one enemy unit is worth and how much life it has.

For every unit BuildABot can name: the cheapest card that brings it, that
card's cost, how many of the unit it deploys and the unit's hit points at
LEVEL - read from the game data through the simulator's card table. The live
bot uses it to price a group of enemies (two Skeletons are not worth Archers)
and to guess how much life a unit our towers are shooting has left, without
loading the simulator itself.

    PYTHONPATH=".;scripts;vendor/ClashRoyaleBuildABot" python scripts/build_unit_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "unit_stats.json"
# Opponents met between 1600 and 2300 trophies: king level 8, cards about 9.
LEVEL = 9
# BuildABot card name -> the simulator's (the game's internal name).
SIM_NAMES = {"archers": "archer", "elite_barbarians": "angry_barbarians",
             "wall_breakers": "wallbreakers", "ice_golem": "ice_golemite",
             "ice_spirit": "ice_spirits", "fire_spirit": "fire_spirits",
             "lumberjack": "axe_man", "furnace": "firespirit_hut", "zappies": "mini_sparkys",
             "x_bow": "xbow", "mother_witch": "witch_mother", "night_witch": "dark_witch",
             "royal_ghost": "ghost", "bandit": "assassin", "dart_goblin": "blowdart_goblin",
             "guards": "skeleton_warriors", "magic_archer": "elite_archer"}


def _units(spec, card) -> list[tuple[str, int, int]]:
    """(BuildABot unit name, count, hit points) for each unit `card` brings."""
    names = [unit.name for unit in getattr(card, "units", ())]
    found = []
    for name, sim_unit, count in ((names[0] if names else None, spec.unit, spec.summon_number),
                                  (names[1] if len(names) > 1 else None, spec.secondary_unit,
                                   spec.secondary_summon_number)):
        hitpoints = getattr(sim_unit, "hitpoints", 0) if sim_unit is not None else 0
        if name and hitpoints:
            found.append((name, max(1, int(count or 1)), int(hitpoints)))
    return found


def build() -> dict:
    from clashroyalebuildabot.namespaces.cards import Cards
    from sim.gamedata import load_gamedata

    data = load_gamedata(level=LEVEL)
    stats: dict[str, dict] = {}
    for attr in sorted(dir(Cards)):
        card = getattr(Cards, attr)
        if attr.startswith("_") or not isinstance(getattr(card, "cost", None), (int, float)):
            continue
        spec = data.get(SIM_NAMES.get(card.name, card.name))
        if spec is None:
            continue
        for unit, count, hitpoints in _units(spec, card):
            known = stats.get(unit)
            if known is None or card.cost < known["cost"]:
                stats[unit] = {"card": card.name, "cost": int(card.cost), "count": count,
                               "hp": hitpoints}
    return {"level": LEVEL, "units": dict(sorted(stats.items()))}


def main() -> int:
    table = build()
    OUT.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"{len(table['units'])} units -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
