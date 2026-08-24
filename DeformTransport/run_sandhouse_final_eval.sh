#!/usr/bin/env bash
set -euo pipefail

DT=/workspace/DeformTransport
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python

FROZEN_EVAL="$DT/server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/eval_v3.py"

DEV_EVAL="$DT/server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval"

FORMAL="$DT/server_runs/wan_move_formal/20260810_124701__v3d_mechanism_and_sandhouse_heldout_seed0"

MECH_CURRENT="$DT/server_runs/wan_move_mechanism_eval/current_v3d_identity_eval.txt"

STAMP=$(date +%Y%m%d_%H%M%S)

RUN="$DT/server_runs/wan_move_heldout_eval/${STAMP}__sandhouse_v3d_final_heldout_seed0"

mkdir -p "$RUN"

echo "$RUN" > \
"$DT/server_runs/wan_move_heldout_eval/current_sandhouse_final_eval.txt"

echo "RUN=$RUN"

# ============================================================
# 0. Input integrity
# ============================================================

CORRECT="$FORMAL/sandhouse_correct/sandhouse_v3d_correct_seed0.mp4"
SHUFFLED="$FORMAL/sandhouse_shuffled/sandhouse_v3d_identity_shuffled_seed0.mp4"

TRACKS="$FORMAL/artifacts/sandhouse/sandhouse_v3d_correct_tracks.npy"
VIS="$FORMAL/artifacts/sandhouse/sandhouse_v3d_visibility.npy"
SOURCE=$(cat "$FORMAL/artifacts/sandhouse/input_image_path.txt")

for f in \
"$FROZEN_EVAL" \
"$CORRECT" \
"$SHUFFLED" \
"$TRACKS" \
"$VIS" \
"$SOURCE"
do
    test -s "$f" || {
        echo "MISSING: $f"
        exit 10
    }
done

echo
echo "===== FROZEN SANDHOUSE INPUTS ====="

sha256sum \
"$CORRECT" \
"$SHUFFLED" \
"$TRACKS" \
"$VIS" \
"$SOURCE" \
| tee "$RUN/input_sha256.txt"


# ============================================================
# 1. Audit SandHouse arrays
# ============================================================

"$PY" - <<'PY'
from pathlib import Path
import numpy as np

DT = Path("/workspace/DeformTransport")

FORMAL = (
    DT
    / "server_runs/wan_move_formal/"
      "20260810_124701__v3d_mechanism_and_sandhouse_heldout_seed0"
)

t = np.load(
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_correct_tracks.npy"
)

s = np.load(
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_identity_shuffled_tracks.npy"
)

v = np.load(
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_visibility.npy"
)

d = np.load(
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_depth.npy"
)

ids = np.load(
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_ids.npy"
)

frames = np.load(
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_frame_ids_165_to_81.npy"
)

assert t.shape == (1,81,1791,2), t.shape
assert s.shape == t.shape
assert v.shape == (1,81,1791)
assert d.shape == (1,81,1791)
assert ids.shape == (1791,)

assert np.array_equal(
    s[:,1:],
    t[:,1:],
)

assert not np.array_equal(
    s[:,0],
    t[:,0],
)

assert np.array_equal(
    frames,
    np.arange(0,161,2)
)

print("SANDHOUSE_ARRAY_CONTRACT_OK")
print("tracks =", t.shape)
print("visibility =", v.shape)
print("selected material tracks =", t.shape[2])
print("physical states =", frames.tolist())
PY


# ============================================================
# 2. Auto-discover compatible SandHouse RealWonder baseline
#
# Strict:
# - must contain Sand + RealWonder/baseline in path
# - must be an 81-frame video
# - exclude Wan-Move / current V3D outputs
#
# If no unique compatible RW is found:
# mechanism evaluation STILL proceeds.
# ============================================================

"$PY" - <<'PY'
from pathlib import Path
import cv2
import hashlib
import json

DT = Path("/workspace/DeformTransport")
base = DT / "server_runs"

RUN = Path(
    (
        DT
        / "server_runs/wan_move_heldout_eval/"
          "current_sandhouse_final_eval.txt"
    ).read_text().strip()
)

candidates = []

for p in base.rglob("*.mp4"):

    s = str(p).lower()

    if "sand" not in s:
        continue

    if (
        "wan_move_formal" in s
        or "wan_move_method" in s
        or "wan_move_mechanism" in s
        or "wan_move_heldout" in s
    ):
        continue

    if not (
        "realwonder" in s
        or "/baseline/" in s
        or "__baseline" in s
        or "_baseline_" in s
    ):
        continue

    cap = cv2.VideoCapture(str(p))

    frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    rank = 2

    if "realwonder" in s:
        rank = 0
    elif "sand_house" in s and "baseline" in s:
        rank = 1

    candidates.append({
        "path": str(p),
        "frames": frames,
        "width": width,
        "height": height,
        "rank": rank,
    })


compatible = [
    x for x in candidates
    if x["frames"] == 81
]

compatible.sort(
    key=lambda x: (
        x["rank"],
        len(x["path"]),
        x["path"],
    )
)


chosen = None
status = None

if not compatible:

    status = "NO_COMPATIBLE_81F_RW_FOUND"

else:

    best_rank = compatible[0]["rank"]

    best = [
        x for x in compatible
        if x["rank"] == best_rank
    ]

    if len(best) == 1:

        chosen = best[0]
        status = "UNIQUE_RW_FOUND"

    else:

        # Multiple paths are accepted automatically only if
        # they are byte-identical.
        groups = {}

        for x in best:

            h = hashlib.sha256()

            with open(
                x["path"],
                "rb"
            ) as f:

                for b in iter(
                    lambda: f.read(4 << 20),
                    b"",
                ):
                    h.update(b)

            groups.setdefault(
                h.hexdigest(),
                []
            ).append(x)

        if len(groups) == 1:

            chosen = sorted(
                best,
                key=lambda x: (
                    len(x["path"]),
                    x["path"],
                )
            )[0]

            status = "MULTIPLE_IDENTICAL_RW_FOUND"

        else:

            status = "MULTIPLE_DISTINCT_RW_CANDIDATES"


report = {
    "status": status,
    "chosen": chosen,
    "compatible_candidates": compatible,
    "all_candidates": candidates,
}

(
    RUN
    / "realwonder_discovery.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    )
    + "\n"
)

if chosen is not None:

    (
        RUN
        / "realwonder_path.txt"
    ).write_text(
        chosen["path"]
        + "\n"
    )

    print(
        "REALWONDER_AUTO_DISCOVERY_OK"
    )

    print(
        "RW =",
        chosen["path"]
    )

    print(
        "shape =",
        chosen["width"],
        "x",
        chosen["height"],
        "frames =",
        chosen["frames"],
    )

else:

    print(
        "REALWONDER_AUTO_DISCOVERY_NOT_RESOLVED"
    )

    print(
        "status =",
        status,
    )

    print(
        "Mechanism evaluation will continue."
    )

    print(
        "Candidate report:",
        RUN / "realwonder_discovery.json"
    )
PY


# ============================================================
# 3. Create frozen evaluator copy with ONLY old numerical
#    reproduction guards disabled.
#
# Metric code itself stays unchanged.
# ============================================================

"$PY" - <<'PY'
from pathlib import Path
import ast
import json

DT = Path("/workspace/DeformTransport")

SRC = (
    DT
    / "server_runs/wan_move_method_eval/"
      "20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/"
      "eval_v3.py"
)

RUN = Path(
    (
        DT
        / "server_runs/wan_move_heldout_eval/"
          "current_sandhouse_final_eval.txt"
    ).read_text().strip()
)

tree = ast.parse(
    SRC.read_text()
)


class T(ast.NodeTransformer):

    def __init__(self):
        self.removed = 0

    def visit_Raise(self, node):

        txt = ast.unparse(
            node
        ).lower()

        if (
            "reproduction" in txt
            and "mismatch" in txt
        ):

            self.removed += 1

            return ast.Pass()

        return self.generic_visit(node)


tr = T()

tree = tr.visit(tree)

ast.fix_missing_locations(
    tree
)

if tr.removed < 1:

    raise RuntimeError(
        "Frozen reproduction guard not found."
    )

out = RUN / "eval_v3_sandhouse_core.py"

out.write_text(
    "# FROZEN V3 EVALUATOR CORE FOR SANDHOUSE\n"
    "# Only historical numerical reproduction Raise statements\n"
    "# are disabled. Metric implementation is inherited unchanged.\n\n"
    +
    ast.unparse(tree)
    +
    "\n"
)

(
    RUN
    / "evaluator_transform.json"
).write_text(
    json.dumps(
        {
            "source":
                str(SRC),

            "reproduction_guards_removed":
                tr.removed,

            "metric_code":
                "frozen eval_v3.py",
        },
        indent=2,
    )
    + "\n"
)

print(
    "SANDHOUSE_FROZEN_EVALUATOR_CORE_OK",
    "guards_removed=",
    tr.removed,
)
PY


"$PY" -m py_compile \
"$RUN/eval_v3_sandhouse_core.py"

cp "$FROZEN_EVAL" \
"$RUN/eval_v3_original.py"

sha256sum \
"$RUN/eval_v3_original.py" \
"$RUN/eval_v3_sandhouse_core.py" \
> "$RUN/evaluator_sha256.txt"


# ============================================================
# 4. SandHouse wrapper
# ============================================================

cat > "$RUN/run_eval.py" <<'PY'
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np


DT = Path(
    "/workspace/DeformTransport"
)

FORMAL = (
    DT
    / "server_runs/wan_move_formal/"
      "20260810_124701__v3d_mechanism_and_sandhouse_heldout_seed0"
)

RUN = Path(
    (
        DT
        / "server_runs/wan_move_heldout_eval/"
          "current_sandhouse_final_eval.txt"
    ).read_text().strip()
)


spec = importlib.util.spec_from_file_location(
    "frozen_eval",
    RUN
    / "eval_v3_sandhouse_core.py",
)

ev = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    ev
)


correct = (
    FORMAL
    / "sandhouse_correct/"
      "sandhouse_v3d_correct_seed0.mp4"
)

shuffled = (
    FORMAL
    / "sandhouse_shuffled/"
      "sandhouse_v3d_identity_shuffled_seed0.mp4"
)

source = Path(
    (
        FORMAL
        / "artifacts/sandhouse/"
          "input_image_path.txt"
    ).read_text().strip()
)

tracks = (
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_correct_tracks.npy"
)

vis = (
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_visibility.npy"
)


rw_file = (
    RUN
    / "realwonder_path.txt"
)

rw_available = (
    rw_file.is_file()
)

if rw_available:

    rw = Path(
        rw_file.read_text().strip()
    )

else:

    # Placeholder ONLY so the frozen evaluator can compute
    # the Correct-vs-Shuffled mechanism result.
    #
    # It is never reported as a real RW comparison.
    rw = correct


ev.CANDIDATES = [
    "v3d",
]


ev.CASES[
    "sandhouse"
] = {
    "source":
        str(source),

    "rw":
        str(rw),

    "old_correct":
        str(shuffled),

    "tracks":
        str(tracks),

    "vis":
        str(vis),

    "expect": {
        "n":
            1791,

        # Historical reproduction fields are irrelevant to
        # held-out data; guards are disabled.
        "valid_tracks":
            -1,

        "obs":
            -1,

        "lab":
            float("nan"),

        "rgb":
            float("nan"),

        "tcme":
            float("nan"),
    },
}


def formal_method_paths(
    root,
    suite,
    case,
):

    assert case == "sandhouse"

    return {
        # actual RealWonder if found;
        # otherwise placeholder Correct,
        # ignored by final reporting.
        "rw":
            rw,

        # Relabel for causal comparison:
        # frozen evaluator defines
        # OldCorrect - candidate
        # for appearance.
        "old_correct":
            shuffled,

        "v3d":
            correct,
    }


ev.method_paths = (
    formal_method_paths
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    required=True,
    choices=[
        "appearance",
        "motion",
    ],
)

parser.add_argument(
    "--batch",
    type=int,
    default=4,
)

args = parser.parse_args()


if args.mode == "appearance":

    r = ev.appearance_case(
        DT,
        DT,
        "sandhouse",
    )

    report = {
        "protocol":
            "SandHouse held-out frozen TC-MAR",

        "rw_available":
            rw_available,

        "cases": {
            "sandhouse":
                r
        },
    }

    (
        RUN
        / "appearance_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    print(
        "SANDHOUSE_APPEARANCE_DONE"
    )


else:

    r = ev.motion_case(
        DT,
        DT,
        "sandhouse",
        args.batch,
    )

    r[
        "rw_available"
    ] = rw_available

    (
        RUN
        / "sandhouse_motion_report.json"
    ).write_text(
        json.dumps(
            r,
            indent=2,
        )
        + "\n"
    )

    print(
        "SANDHOUSE_MOTION_DONE"
    )
PY


"$PY" -m py_compile \
"$RUN/run_eval.py"

echo "SANDHOUSE_WRAPPER_OK"


# ============================================================
# 5. Pick free evaluation GPU.
#    Prefer GPU1 -> GPU2 -> GPU0.
# ============================================================

GPU=$(
"$PY" - <<'PY'
import subprocess

text = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)

rows = []

for line in text.strip().splitlines():

    a = [
        int(x.strip())
        for x in line.split(",")
    ]

    rows.append(a)


for wanted in [1,2,0]:

    row = next(
        x for x in rows
        if x[0] == wanted
    )

    idx, used, free, util = row

    if (
        free >= 25000
        and util <= 50
    ):

        print(idx)
        raise SystemExit


raise SystemExit(
    "NO_SAFE_GPU"
)
PY
)

echo
echo "===== SELECTED EVAL GPU ====="
echo "GPU=$GPU"

nvidia-smi \
--query-gpu=index,memory.used,memory.free,utilization.gpu \
--format=csv,noheader


# ============================================================
# 6. Run appearance CPU + motion GPU in parallel
# ============================================================

echo
echo "===== START SANDHOUSE TC-MAR ====="

"$PY" \
"$RUN/run_eval.py" \
--mode appearance \
> "$RUN/appearance_stdout.log" \
2> "$RUN/appearance_stderr.log" &

APID=$!

echo "$APID" \
> "$RUN/appearance_pid.txt"


echo
echo "===== START SANDHOUSE TC-ME GPU${GPU} ====="

CUDA_VISIBLE_DEVICES="$GPU" \
"$PY" \
"$RUN/run_eval.py" \
--mode motion \
--batch 4 \
> "$RUN/motion_stdout.log" \
2> "$RUN/motion_stderr.log" &

MPID=$!

echo "$MPID" \
> "$RUN/motion_pid.txt"


set +e

wait "$APID"
AEC=$?

wait "$MPID"
MEC=$?

set -e


echo "$AEC" \
> "$RUN/appearance_exit_code.txt"

echo "$MEC" \
> "$RUN/motion_exit_code.txt"


if [ "$AEC" -ne 0 ]; then

    echo "SANDHOUSE_APPEARANCE_FAILED"

    tail -60 \
    "$RUN/appearance_stderr.log"

    exit "$AEC"
fi


if [ "$MEC" -ne 0 ]; then

    echo "SANDHOUSE_MOTION_FAILED"

    tail -60 \
    "$RUN/motion_stderr.log"

    exit "$MEC"
fi


echo
echo "SANDHOUSE_RAW_EVAL_DONE"


# ============================================================
# 7. Final SandHouse result + 3-case master table
# ============================================================

"$PY" - <<'PY'
from pathlib import Path
import json


DT = Path(
    "/workspace/DeformTransport"
)

RUN = Path(
    (
        DT
        / "server_runs/wan_move_heldout_eval/"
          "current_sandhouse_final_eval.txt"
    ).read_text().strip()
)

DEV = (
    DT
    / "server_runs/wan_move_method_eval/"
      "20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval"
)

MECH = Path(
    (
        DT
        / "server_runs/wan_move_mechanism_eval/"
          "current_v3d_identity_eval.txt"
    ).read_text().strip()
)


app = json.loads(
    (
        RUN
        / "appearance_report.json"
    ).read_text()
)

motion = json.loads(
    (
        RUN
        / "sandhouse_motion_report.json"
    ).read_text()
)

A = app[
    "cases"
][
    "sandhouse"
]

M = motion


# ------------------------------------------------------------
# SandHouse Correct vs Shuffled
# ------------------------------------------------------------

shuf_app = (
    A[
        "methods"
    ][
        "old_correct"
    ][
        "tc_mar_lab"
    ][
        "mean"
    ]
)

corr_app = (
    A[
        "methods"
    ][
        "v3d"
    ][
        "tc_mar_lab"
    ][
        "mean"
    ]
)

ax = (
    A[
        "vs_old_correct"
    ][
        "v3d"
    ]
)

app_diff = (
    ax[
        "paired_mean_difference"
    ]
)

app_ci = (
    ax[
        "bootstrap_95_ci"
    ]
)

if app_ci[0] > 0:
    app_dec = "CORRECT_WIN"
elif app_ci[1] < 0:
    app_dec = "CORRECT_LOSS"
else:
    app_dec = "TIE"


shuf_me = (
    M[
        "methods"
    ][
        "old_correct"
    ][
        "transition_mean_epe_mean"
    ]
)

corr_me = (
    M[
        "methods"
    ][
        "v3d"
    ][
        "transition_mean_epe_mean"
    ]
)

mx = (
    M[
        "vs_old_correct"
    ][
        "v3d"
    ]
)

motion_diff = (
    mx[
        "paired_mean_difference"
    ]
)

motion_ci = (
    mx[
        "bootstrap_95_ci"
    ]
)

# frozen motion:
# candidate - old_correct
# = Correct - Shuffled

if motion_ci[1] < 0:
    motion_dec = "CORRECT_WIN"
elif motion_ci[0] > 0:
    motion_dec = "CORRECT_LOSS"
else:
    motion_dec = "TIE"


rw_available = bool(
    app[
        "rw_available"
    ]
)


sand = {
    "mechanism": {
        "tc_mar_lab": {
            "shuffled":
                shuf_app,

            "correct":
                corr_app,

            "shuffled_minus_correct":
                app_diff,

            "ci95":
                app_ci,

            "decision":
                app_dec,
        },

        "tc_me": {
            "shuffled":
                shuf_me,

            "correct":
                corr_me,

            "correct_minus_shuffled":
                motion_diff,

            "ci95":
                motion_ci,

            "decision":
                motion_dec,
        },
    },

    "vs_realwonder":
        None,
}


# ------------------------------------------------------------
# SandHouse V3D vs RealWonder if exact compatible RW found
# ------------------------------------------------------------

if rw_available:

    rw_app = (
        A[
            "methods"
        ][
            "rw"
        ][
            "tc_mar_lab"
        ][
            "mean"
        ]
    )

    rax = (
        A[
            "vs_realwonder"
        ][
            "v3d"
        ]
    )

    rw_me = (
        M[
            "methods"
        ][
            "rw"
        ][
            "transition_mean_epe_mean"
        ]
    )

    rmx = (
        M[
            "vs_realwonder"
        ][
            "v3d"
        ]
    )

    sand[
        "vs_realwonder"
    ] = {
        "tc_mar_lab": {
            "realwonder":
                rw_app,

            "v3d":
                corr_app,

            "rw_minus_v3d":
                rax[
                    "paired_mean_difference"
                ],

            "ci95":
                rax[
                    "bootstrap_95_ci"
                ],

            "decision":
                rax[
                    "decision"
                ],
        },

        "tc_me": {
            "realwonder":
                rw_me,

            "v3d":
                corr_me,

            "rw_minus_v3d":
                rmx[
                    "paired_mean_difference"
                ],

            "ci95":
                rmx[
                    "bootstrap_95_ci"
                ],

            "decision":
                rmx[
                    "decision"
                ],
        },
    }


(
    RUN
    / "sandhouse_final_summary.json"
).write_text(
    json.dumps(
        sand,
        indent=2,
    )
    + "\n"
)


# ------------------------------------------------------------
# Existing Santa / Tree mechanism
# ------------------------------------------------------------

mech = json.loads(
    (
        MECH
        / "mechanism_summary.json"
    ).read_text()
)


# ------------------------------------------------------------
# Existing Santa / Tree V3D vs RW
# ------------------------------------------------------------

dev_app = json.loads(
    (
        DEV
        / "appearance_report.json"
    ).read_text()
)

dev_motion = {
    c:
        json.loads(
            (
                DEV
                / f"{c}_motion_report.json"
            ).read_text()
        )
    for c in [
        "santa",
        "tree",
    ]
}


master = {
    "method":
        "V3D",

    "development_cases": [
        "santa",
        "tree",
    ],

    "heldout_case":
        "sandhouse",

    "cases": {},
}


for case in [
    "santa",
    "tree",
]:

    da = dev_app[
        "cases"
    ][
        case
    ]

    dm = dev_motion[
        case
    ]

    master[
        "cases"
    ][
        case
    ] = {
        "split":
            "development",

        "mechanism":
            mech[
                "cases"
            ][
                case
            ],

        "vs_realwonder": {
            "tc_mar_lab": {
                "realwonder":
                    da[
                        "methods"
                    ][
                        "rw"
                    ][
                        "tc_mar_lab"
                    ][
                        "mean"
                    ],

                "v3d":
                    da[
                        "methods"
                    ][
                        "v3d"
                    ][
                        "tc_mar_lab"
                    ][
                        "mean"
                    ],

                "rw_minus_v3d":
                    da[
                        "vs_realwonder"
                    ][
                        "v3d"
                    ][
                        "paired_mean_difference"
                    ],

                "ci95":
                    da[
                        "vs_realwonder"
                    ][
                        "v3d"
                    ][
                        "bootstrap_95_ci"
                    ],

                "decision":
                    da[
                        "vs_realwonder"
                    ][
                        "v3d"
                    ][
                        "decision"
                    ],
            },

            "tc_me": {
                "realwonder":
                    dm[
                        "methods"
                    ][
                        "rw"
                    ][
                        "transition_mean_epe_mean"
                    ],

                "v3d":
                    dm[
                        "methods"
                    ][
                        "v3d"
                    ][
                        "transition_mean_epe_mean"
                    ],

                "rw_minus_v3d":
                    dm[
                        "vs_realwonder"
                    ][
                        "v3d"
                    ][
                        "paired_mean_difference"
                    ],

                "ci95":
                    dm[
                        "vs_realwonder"
                    ][
                        "v3d"
                    ][
                        "bootstrap_95_ci"
                    ],

                "decision":
                    dm[
                        "vs_realwonder"
                    ][
                        "v3d"
                    ][
                        "decision"
                    ],
            },
        },
    }


master[
    "cases"
][
    "sandhouse"
] = {
    "split":
        "held-out",

    **sand,
}


(
    RUN
    / "three_case_master_summary.json"
).write_text(
    json.dumps(
        master,
        indent=2,
    )
    + "\n"
)


print()
print("=" * 84)
print("SANDHOUSE HELD-OUT FINAL RESULT")
print("=" * 84)

print()
print("[Correct vs Identity-Shuffled]")

print(
    "TC-MAR Shuffled =",
    shuf_app
)

print(
    "TC-MAR Correct  =",
    corr_app
)

print(
    "Shuf-Correct    =",
    app_diff
)

print(
    "95% CI          =",
    app_ci
)

print(
    "TC-MAR decision =",
    app_dec
)

print()

print(
    "TC-ME Shuffled  =",
    shuf_me
)

print(
    "TC-ME Correct   =",
    corr_me
)

print(
    "Correct-Shuf    =",
    motion_diff
)

print(
    "95% CI          =",
    motion_ci
)

print(
    "TC-ME decision  =",
    motion_dec
)


print()
print("[V3D vs RealWonder]")

if sand[
    "vs_realwonder"
] is None:

    print(
        "NOT AUTOMATICALLY RESOLVED."
    )

    print(
        "Mechanism result is still VALID."
    )

    print(
        "See realwonder_discovery.json."
    )

else:

    x = sand[
        "vs_realwonder"
    ]

    print(
        "TC-MAR RW      =",
        x[
            "tc_mar_lab"
        ][
            "realwonder"
        ]
    )

    print(
        "TC-MAR V3D     =",
        x[
            "tc_mar_lab"
        ][
            "v3d"
        ]
    )

    print(
        "TC-MAR decision=",
        x[
            "tc_mar_lab"
        ][
            "decision"
        ]
    )

    print()

    print(
        "TC-ME RW       =",
        x[
            "tc_me"
        ][
            "realwonder"
        ]
    )

    print(
        "TC-ME V3D      =",
        x[
            "tc_me"
        ][
            "v3d"
        ]
    )

    print(
        "TC-ME decision =",
        x[
            "tc_me"
        ][
            "decision"
        ]
    )


print()
print("=" * 84)
print("THREE-CASE MASTER SUMMARY")
print("=" * 84)

for case in [
    "santa",
    "tree",
    "sandhouse",
]:

    x = master[
        "cases"
    ][
        case
    ]

    print()
    print(
        case.upper(),
        f"[{x['split']}]"
    )

    mech_x = x[
        "mechanism"
    ]

    if case == "sandhouse":

        print(
            "Identity TC-MAR =",
            mech_x[
                "tc_mar_lab"
            ][
                "decision"
            ]
        )

        print(
            "Identity TC-ME  =",
            mech_x[
                "tc_me"
            ][
                "decision"
            ]
        )

    else:

        print(
            "Identity TC-MAR =",
            mech_x[
                "TC_MAR_Lab"
            ][
                "decision"
            ]
        )

        print(
            "Identity TC-ME  =",
            mech_x[
                "TC_ME"
            ][
                "decision"
            ]
        )


    rw = x.get(
        "vs_realwonder"
    )

    if rw is None:

        print(
            "vs RW          = N/A"
        )

    else:

        print(
            "vs RW TC-MAR   =",
            rw[
                "tc_mar_lab"
            ][
                "decision"
            ]
        )

        print(
            "vs RW TC-ME    =",
            rw[
                "tc_me"
            ][
                "decision"
            ]
        )


print()
print("=" * 84)
print("HELDOUT_EVAL_DONE")
print("=" * 84)

(
    RUN
    / "EVAL_DONE.txt"
).write_text(
    "done\n"
)
PY


echo
echo "RUN=$RUN"
echo "SANDHOUSE_FINAL_EVALUATION_COMPLETE"

