"""Unattended night: train in the simulator, then play on MuMu, and again.

One thing at a time, as the user asked:

1. wait for any training already running to finish;
2. start MuMu and Clash Royale, play MATCHES ladder matches with the newest
   simulator-best checkpoint (`climb.ps1 -Strategy ia -Record -Matches N`);
3. shut MuMu down and train STEPS more steps in the simulator from that
   checkpoint (`train_starter_deck.ps1 -Init ... -Name starter_longN`);
4. repeat until the deadline.

Each cycle's simulator score and live record go to tmp/live/overnight.json;
progress to tmp/logs/overnight.log. Create tmp/live/stop_overnight.flag to
stop after the current step.

    python scripts/overnight_cycle.py --until 09:00
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "tmp" / "rl"
LIVE = ROOT / "tmp" / "live"
RUNNER_LOG = LIVE / "buildabot-native.log"
SUMMARY = LIVE / "overnight.json"
STOP = LIVE / "stop_overnight.flag"
MUMU = Path(r"C:\Program Files\Netease\MuMuPlayer\nx_main\MuMuManager.exe")
ADB = Path(r"C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe")
SERIAL = "127.0.0.1:16384"
RESULT = re.compile(r"RESULT exit .*outcome=(victoire|défaite|inconnu)")
SERIES = "starter_long"
MINUTES_PER_MATCH = 8                    # generous: queue + match + result screens


def log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def parse_results(lines) -> dict:
    counts = {"victoire": 0, "défaite": 0, "inconnu": 0}
    for line in lines:
        match = RESULT.search(line)
        if match:
            counts[match.group(1)] += 1
    return counts


def _runs(folder: Path, prefix: str) -> list:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d*)_best\.pt$")
    return [(int(m.group(1) or 1), p) for p in folder.glob(f"{prefix}*_best.pt")
            if (m := pattern.match(p.name))]


def next_run_name(folder: Path, prefix: str = SERIES) -> str:
    return f"{prefix}{max((n for n, _p in _runs(folder, prefix)), default=1) + 1}"


def newest_best(folder: Path, prefix: str = SERIES) -> Optional[Path]:
    runs = _runs(folder, prefix)
    return max(runs)[1] if runs else None


def processes(pattern: str) -> int:
    import psutil
    count = 0
    for process in psutil.process_iter(["cmdline"]):
        try:
            if pattern in " ".join(process.info["cmdline"] or []):
                count += 1
        except Exception:
            continue
    return count


def wait_gone(pattern: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if processes(pattern) == 0:
            return True
        time.sleep(20)
    return False


def adb(*args, timeout=30) -> str:
    try:
        result = subprocess.run([str(ADB), "-s", SERIAL, *args], capture_output=True,
                                text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception:
        return ""


def mumu_start() -> bool:
    subprocess.run([str(MUMU), "control", "-v", "0", "launch"], capture_output=True, timeout=60)
    for _ in range(80):
        subprocess.run([str(ADB), "connect", SERIAL], capture_output=True, timeout=30)
        if adb("shell", "getprop", "sys.boot_completed") == "1":
            break
        time.sleep(3)
    else:
        return False
    time.sleep(10)
    # MuMu's own store draws full-screen ads over the game (as a system app,
    # denying its overlay permission does not hold): keep it disabled.
    adb("shell", "am", "force-stop", "com.mumu.store")
    adb("shell", "pm", "disable-user", "--user", "0", "com.mumu.store")
    # MuMu reopens where it last was, and stays there: the user keeps it on
    # the second monitor. The capture reads it in physical pixels on any
    # monitor (studio.capture.ensure_dpi_awareness), so it is not moved.
    adb("shell", "monkey", "-p", "com.supercell.clashroyale", "-c",
        "android.intent.category.LAUNCHER", "1")
    time.sleep(40)
    return "clashroyale" in adb("shell", "dumpsys", "activity", "activities", timeout=30)


def mumu_stop() -> None:
    subprocess.run([str(MUMU), "control", "-v", "0", "shutdown"], capture_output=True, timeout=60)
    time.sleep(10)


def climb_command(checkpoint: Path, matches: int, strategy: str = "ia",
                  deck: str = "starter") -> list:
    """climb.ps1 for `matches` games of `strategy` on the runner's `deck` preset."""
    return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(ROOT / "scripts" / "climb.ps1"), "-Strategy", strategy, "-Deck", deck,
            "-Checkpoint", str(checkpoint.relative_to(ROOT)), "-Record",
            "-Matches", str(matches)]


def play(checkpoint: Path, matches: int, strategy: str = "ia", deck: str = "starter") -> dict:
    start = len(RUNNER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()) \
        if RUNNER_LOG.exists() else 0
    (LIVE / "stop_after_match.flag").unlink(missing_ok=True)
    subprocess.run(climb_command(checkpoint, matches, strategy, deck), cwd=str(ROOT), timeout=120)
    time.sleep(30)
    if not wait_gone("mumu_buildabot_native.py", matches * MINUTES_PER_MATCH * 60):
        log("runner still going past its time: asking it to stop after the match")
        (LIVE / "stop_after_match.flag").write_text("overnight", encoding="utf-8")
        wait_gone("mumu_buildabot_native.py", 15 * 60)
    lines = RUNNER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[start:]
    return parse_results(lines)


def train_command(init: Path, name: str, steps: int, league: int = 0, extra=(),
                  sim_deck: str = "") -> list:
    """train_starter_deck.ps1 from `init`; `sim_deck` = simulator card names, in slot order."""
    deck = ["-Deck", sim_deck] if sim_deck else []
    return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(ROOT / "scripts" / "train_starter_deck.ps1"), "-Init",
            str(init.relative_to(ROOT)), "-Name", name, "-Steps", str(steps),
            "-Envs", "16", "-EvalEvery", "1000000", "-League", str(league), *deck, *extra]


def train(init: Path, name: str, steps: int, league: int = 0, extra=(),
          sim_deck: str = "") -> Optional[dict]:
    with (ROOT / "tmp" / "logs" / f"{name}.log").open("w") as out:
        subprocess.run(train_command(init, name, steps, league, extra, sim_deck),
                       cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    best = RL / f"{name}_best.json"
    return json.loads(best.read_text(encoding="utf-8")) if best.exists() else None


def save(cycles: list) -> None:
    SUMMARY.write_text(json.dumps(cycles, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--until", default="09:00", help="HH:MM, next occurrence")
    parser.add_argument("--steps", type=int, default=6_000_000)
    parser.add_argument("--matches", type=int, default=12)
    parser.add_argument("--league", type=int, default=0,
                        help="past selves kept as training opponents (0 = off)")
    parser.add_argument("--train-first", action="store_true",
                        help="wait for a runner already playing, then start with training")
    parser.add_argument("--series", default=SERIES,
                        help="run name prefix: starter_long, or starter_weak for the "
                             "weak-point series (live decks, stronger opponents)")
    parser.add_argument("--fallback", default="",
                        help="checkpoint to play while the series has none yet")
    parser.add_argument("--weak-points", action="store_true",
                        help="train against the decks met live with x1.0-1.45 opponent "
                             "units, and rebuild those decks after every series of matches")
    parser.add_argument("--strategy", default="ia",
                        help="climb.ps1 -Strategy for the live series (ia, hybride...)")
    parser.add_argument("--deck", default="starter", help="climb.ps1 -Deck preset")
    parser.add_argument("--sim-deck", default="",
                        help="simulator card names for training, in slot order (P.E.K.K.A: "
                             "knight,pekka,minions,arrows,inferno_tower,archer,musketeer,minipekka)")
    args = parser.parse_args()
    extra = (["-Opponent", "live", "-Boost", "1.0-1.45", "-BrainShare", "0.25",
              "-EvalOpponent", "live@1.33,meta@1.21"] if args.weak_points else [])
    hour, minute = map(int, args.until.split(":"))
    now = datetime.now()
    deadline = now.replace(hour=hour, minute=minute, second=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    log(f"overnight until {deadline:%d/%m %H:%M}")
    STOP.unlink(missing_ok=True)
    cycles = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else []

    log("waiting for the training already running")
    wait_gone("sim.train_ppo", 6 * 3600)
    skip_play = args.train_first
    if skip_play:
        log("waiting for the runner already playing")
        wait_gone("mumu_buildabot_native.py", 3 * 3600)
    while datetime.now() < deadline and not STOP.exists():
        checkpoint = newest_best(RL, args.series) or (ROOT / args.fallback if args.fallback else None)
        if checkpoint is None or not checkpoint.exists():
            log("no checkpoint to play")
            return 1
        cycle = {"checkpoint": checkpoint.name, "started": f"{datetime.now():%H:%M}"}
        sim = checkpoint.with_suffix(".json")
        if sim.exists():
            cycle["sim"] = json.loads(sim.read_text(encoding="utf-8"))
        if skip_play:
            cycle["live"] = "played before this run (see the runner log)"
        else:
            log(f"play {args.matches} matches with {checkpoint.name} "
                f"({args.strategy}, deck {args.deck})")
            if mumu_start():
                cycle["live"] = play(checkpoint, args.matches, args.strategy, args.deck)
                log(f"live: {cycle['live']}")
            else:
                cycle["live"] = None
                log("MuMu or the game did not start: no matches this cycle")
        skip_play = False
        mumu_stop()
        if args.weak_points:
            # The matches just played join the decks the next run trains against.
            built = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_live_decks.py"),
                                    "--since", "2026-10-13"], cwd=str(ROOT),
                                   capture_output=True, text=True)
            log(f"live decks: {built.stdout.strip().splitlines()[0] if built.stdout else built.stderr[-200:]}")
        cycles.append(cycle)
        save(cycles)
        remaining = (deadline - datetime.now()).total_seconds()
        if STOP.exists() or remaining < 50 * 60:
            break
        name = next_run_name(RL, args.series)
        steps = min(args.steps, int(remaining / 60 * 100_000))   # ~10 min per 1M steps
        log(f"train {name}: {steps:,} steps from {checkpoint.name}")
        cycle["trained"] = {"name": name, "steps": steps, "league": args.league,
                            "weak_points": args.weak_points,
                            "best": train(checkpoint, name, steps, args.league, extra,
                                          args.sim_deck)}
        save(cycles)
    log("overnight done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
