"""onnxruntime sessions limited to a few threads that sleep when idle."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lean_onnx  # noqa: E402


def test_sessions_get_few_threads_and_no_spinning():
    opts = lean_onnx.options()
    assert opts.intra_op_num_threads == lean_onnx.THREADS
    assert opts.inter_op_num_threads == 1
    assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "0"


def test_install_patches_the_session_class_once():
    import onnxruntime as ort
    lean_onnx.install()
    patched = ort.InferenceSession
    lean_onnx.install()
    assert ort.InferenceSession is patched and patched.__name__ == "LeanSession"
