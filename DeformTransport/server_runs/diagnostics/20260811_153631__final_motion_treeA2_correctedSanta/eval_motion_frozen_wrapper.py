import sys
import importlib.util
from pathlib import Path

ROOT = Path("/workspace/DeformTransport")

SRC = ROOT / (
    "server_runs/wan_move_method_eval/"
    "20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/"
    "eval_v3.py"
)

spec = importlib.util.spec_from_file_location("frozen_eval", SRC)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

# Only evaluate one candidate; all metric implementation remains frozen.
ev.CANDIDATES = ["v3d"]

if "--case" not in sys.argv:
    raise RuntimeError("--case required")

case = sys.argv[sys.argv.index("--case") + 1]

if case == "tree":
    # Make old_correct identical to A0 so both frozen comparison branches
    # reduce to the intended A0 vs A2 experiment.
    ev.CASES["tree"]["old_correct"] = ev.CASES["tree"]["rw"]

elif case == "santa":
    rerun = Path(
        (
            ROOT /
            "server_runs/wan_move_formal/"
            "current_santa_corrected_v3d_rerun.txt"
        ).read_text().strip()
    )

    correct = (
        rerun / "correct" /
        "santa_v3d_corrected_visibility_correct_seed0.mp4"
    )

    # rw and old_correct both map to corrected Correct.
    # v3d candidate maps to corrected Shuffled through the custom suite.
    ev.CASES["santa"]["rw"] = str(correct)
    ev.CASES["santa"]["old_correct"] = str(correct)

else:
    raise RuntimeError(case)

ev.main()
