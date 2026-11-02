"""Record MuMu's render surface at native resolution while a bot plays.

Raw material for the detector: real battle frames from the arenas the
account actually plays in (the published dataset's frames come from another
phone, framed and scaled differently). Window capture is read-only, so this
runs beside the bot without slowing it.

    python scripts/record_frames.py tmp/frames/live --minutes 12 --every 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from PIL import Image  # noqa: E402

from studio.capture import SurfaceGrabber, find_surface  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--every", type=float, default=1.0, help="seconds between frames")
    parser.add_argument("--instance", default="Android Device")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grabber = SurfaceGrabber(find_surface(instance=args.instance))
    deadline, saved = time.time() + args.minutes * 60, 0
    while time.time() < deadline:
        started = time.time()
        try:
            array = grabber.grab()
            if array is not None:
                name = f"{datetime.now():%Y%m%d-%H%M%S-%f}.jpg"
                Image.fromarray(array[:, :, 2::-1]).save(args.out / name, quality=95)
                saved += 1
        except Exception as exc:              # a dropped frame is not worth stopping for
            print(f"grab failed: {exc}", flush=True)
        time.sleep(max(0.0, args.every - (time.time() - started)))
    print(f"saved {saved} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
