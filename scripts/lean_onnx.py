"""Keep onnxruntime from taking every core of the machine.

BuildABot's detectors run on onnxruntime. Its CUDA provider needs a CUDA the
machine does not have, so it silently falls back to the CPU - where each of
its sessions builds a thread pool the size of the machine and spins those
threads between frames. Measured live: the runner used 22 of 24 cores.

`install()` makes every InferenceSession created afterwards use a few threads
that sleep when idle. Call it before BuildABot's Detector is built; it patches
`onnxruntime.InferenceSession`, which BuildABot looks up at call time.
"""

from __future__ import annotations

import os

THREADS = int(os.environ.get("HASTYCR_ONNX_THREADS", "4"))
_installed = False


def options(base=None):
    import onnxruntime as ort
    opts = base or ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return opts


def install() -> None:
    global _installed
    if _installed:
        return
    import onnxruntime as ort
    original = ort.InferenceSession

    class LeanSession(original):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
            super().__init__(path_or_bytes, sess_options=options(sess_options),
                             providers=providers, **kwargs)

    ort.InferenceSession = LeanSession
    _installed = True
