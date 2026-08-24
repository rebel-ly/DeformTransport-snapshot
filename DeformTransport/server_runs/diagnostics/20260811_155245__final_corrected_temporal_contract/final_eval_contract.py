import copy
import importlib.util
from pathlib import Path

ROOT = Path("/workspace/DeformTransport")

BASE = ROOT / (
    "server_runs/wan_move_method_eval/"
    "20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/"
    "eval_v3.py"
)

spec = importlib.util.spec_from_file_location(
    "_frozen_eval",
    BASE,
)

_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_base)

# Re-export frozen evaluator implementation.
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

CASES = copy.deepcopy(_base.CASES)

RERUN = Path(
    (
        ROOT /
        "server_runs/wan_move_formal/"
        "current_santa_corrected_v3d_rerun.txt"
    ).read_text().strip()
)

BRIDGE = ROOT / (
    "server_runs/wan_move_bridge/"
    "20260811_024330__santa_corrected_physical_visibility"
)

# Santa comparison semantics:
# rw  = Identity-Shuffled
# v3d = Correct
# therefore rw-v3d > 0 means Correct wins.
CASES["santa"]["rw"] = str(
    RERUN / "shuffled" /
    "santa_v3d_corrected_visibility_shuffled_seed0.mp4"
)

CASES["santa"]["tracks"] = str(
    BRIDGE / "santa_material_tracks_correct.npy"
)

CASES["santa"]["vis"] = str(
    BRIDGE / "santa_material_visibility_correct.npy"
)

# Tree remains:
# rw  = RealWonder A0
# v3d = Persistent-Noise A2
# therefore rw-v3d > 0 means A2 wins.
