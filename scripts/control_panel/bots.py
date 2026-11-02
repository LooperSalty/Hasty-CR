"""Which bots the panel can drive, and the exact command each one is started with.

Bots are launched with Python directly rather than through `run.ps1`: the
panel owns the process, so it can stream its output and kill it, and the
native BuildABot runner has no PowerShell launcher at all. The trade-off is
that the emulator stays at its native 1080x1920 (slower capture than the
540x960 `run.ps1` sets, but nothing to restore if the panel is closed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

NATIVE_SCRIPT = "mumu_buildabot_native.py"
LIVE_SCRIPT = "cr_bot.py"
SUPERVISOR_SCRIPT = "supervisor.py"

DECKS = {
    "starter": "Deck de départ (Chevalier, Archères, Gargouilles, Flèches, "
               "Boule de feu, Géant, Mousquetaire, Mini P.E.K.K.A)",
    "default": "Deck BuildABot (Archères, Tonneau à gobelins, Bébé dragon, "
               "Canon, Chevalier, Mini P.E.K.K.A, Mousquetaire, Sorcière)",
    "pekka": "Deck P.E.K.K.A (Chevalier, P.E.K.K.A, Gargouilles, Flèches, "
             "Tour de l'enfer, Archères, Mousquetaire, Mini P.E.K.K.A)",
}


@dataclass(frozen=True)
class BotSpec:
    key: str
    label: str
    description: str
    script: str
    uses_deck: bool
    needs_checkpoint: bool
    match_end_marker: str
    extra_args: tuple = ()


BOTS = {
    "beatdown": BotSpec(
        key="beatdown",
        label="Gros push Géant (deck au choix)",
        description="Économise jusqu'à 10 élixirs, pose le Géant au fond derrière la tour "
                    "du roi, puis ses troupes derrière lui. Défense et sorts gardés.",
        script=NATIVE_SCRIPT,
        uses_deck=True,
        needs_checkpoint=False,
        match_end_marker=" RESULT exit",
        extra_args=("--strategy", "beatdown"),
    ),
    "buildabot": BotSpec(
        key="buildabot",
        label="BuildABot (règles, deck au choix)",
        description="Le bot de ClashRoyaleBuildABot. Joue le deck choisi, "
                    "qui doit être le deck actif en jeu.",
        script=NATIVE_SCRIPT,
        uses_deck=True,
        needs_checkpoint=False,
        match_end_marker=" RESULT exit",
    ),
    "rules": BotSpec(
        key="rules",
        label="Hasty-CR règles (Hog 2.6)",
        description="La stratégie écrite à la main de Hasty-CR. Exige le deck "
                    "Hog 2.6 en jeu, sinon elle bloque ses actions.",
        script=LIVE_SCRIPT,
        uses_deck=False,
        needs_checkpoint=False,
        match_end_marker=" MATCH_END",
    ),
    "rl": BotSpec(
        key="rl",
        label="Hasty-CR IA entraînée (Hog 2.6)",
        description="Le réseau entraîné dans le simulateur. Exige un "
                    "checkpoint .pt et le deck Hog 2.6 en jeu.",
        script=LIVE_SCRIPT,
        uses_deck=False,
        needs_checkpoint=True,
        match_end_marker=" MATCH_END",
    ),
}


@dataclass(frozen=True)
class LaunchOptions:
    bot: str
    deck: str
    matches: int          # 0 means no limit
    hours: float
    no_queue: bool
    checkpoint: Optional[Path]
    adb: Path
    serial: str


def _validate(options: LaunchOptions) -> BotSpec:
    spec = BOTS.get(options.bot)
    if spec is None:
        raise ValueError(f"bot inconnu : {options.bot!r}")
    if spec.uses_deck and options.deck not in DECKS:
        raise ValueError(f"deck inconnu : {options.deck!r}")
    if options.matches < 0:
        raise ValueError("le nombre de matchs doit être positif ou nul")
    if options.hours <= 0:
        raise ValueError("la durée doit être supérieure à zéro")
    if not options.serial.strip():
        raise ValueError("l'adresse adb de l'émulateur est vide")
    if spec.needs_checkpoint:
        if options.checkpoint is None or not Path(options.checkpoint).is_file():
            raise ValueError("ce bot a besoin d'un fichier checkpoint .pt existant")
    return spec


def build_command(options: LaunchOptions, python: Path, root: Path) -> list[str]:
    """The argv for `options`, or ValueError naming what is wrong."""
    spec = _validate(options)
    command = [str(python), "-u", str(root / "scripts" / spec.script),
               "--adb", str(options.adb), "--serial", options.serial,
               "--hours", str(float(options.hours))]
    if options.matches:
        command += ["--max-matches", str(options.matches)]
    if options.no_queue:
        command.append("--no-queue")
    if spec.uses_deck:
        command += ["--deck", options.deck]
    if spec.script == LIVE_SCRIPT:
        command += ["--vision", live_vision(root)]
    if spec.needs_checkpoint:
        command += ["--rl", str(options.checkpoint)]
    return command + list(spec.extra_args)


def live_vision(root: Path) -> str:
    """'yolo' once scripts/train_detector.py has produced weights, else 'buildabot'.

    The weights are not shipped with the repo. Asking cr_bot for yolo without
    them only logs "VISION unavailable" and falls back, but naming the vision
    that will actually run keeps the panel's command honest.
    """
    trained = any((Path(root) / "tmp" / "yolo" / "runs").glob("*/weights/best.pt"))
    return "yolo" if trained else "buildabot"


def is_match_end(bot: str, line: str) -> bool:
    spec = BOTS.get(bot)
    return spec is not None and spec.match_end_marker in line


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def bot_for_process(cmdline: Sequence[str], cwd: Path, root: Path) -> Optional[str]:
    """Which bot a process is, only if it runs a script of *this* checkout.

    The panel kills what this returns, so a name match is not enough: a copy
    of `cr_bot.py` elsewhere, or `python -m pytest scripts/cr_bot.py`, is not
    a bot. The script is the first `.py` argument, unless `-m` comes first.
    """
    for arg in cmdline[1:]:
        if arg == "-m":
            return None
        if arg.endswith(".py"):
            script = Path(arg)
            if not script.is_absolute():
                script = Path(cwd) / script
            for name in (NATIVE_SCRIPT, LIVE_SCRIPT, SUPERVISOR_SCRIPT):
                if _same_path(script, Path(root) / "scripts" / name):
                    return bot_for_cmdline(cmdline)
            return None
    return None


def bot_for_cmdline(cmdline: Sequence[str]) -> Optional[str]:
    """Which bot a running process is, from its argv, or None if not ours."""
    joined = " ".join(cmdline).replace("\\", "/")
    if f"scripts/{NATIVE_SCRIPT}" in joined:
        return "beatdown" if "beatdown" in cmdline else "buildabot"
    if f"scripts/{LIVE_SCRIPT}" in joined:
        return "rl" if "--rl" in cmdline else "rules"
    if f"scripts/{SUPERVISOR_SCRIPT}" in joined:
        return "rules"
    return None
