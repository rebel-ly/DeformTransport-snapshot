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
            1791,

        "obs":
            11383,

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
