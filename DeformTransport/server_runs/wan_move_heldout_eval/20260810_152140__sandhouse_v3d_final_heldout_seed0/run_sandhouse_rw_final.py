from pathlib import Path
import argparse, importlib.util, json
import numpy as np
import torch
import torch.nn.functional as F

DT = Path("/workspace/DeformTransport")
RUN = DT / "server_runs/wan_move_heldout_eval/20260810_152140__sandhouse_v3d_final_heldout_seed0"
FORMAL = DT / "server_runs/wan_move_formal/20260810_124701__v3d_mechanism_and_sandhouse_heldout_seed0"

RW = RUN / "rw81/realwonder_81_rgb.npy"
CORRECT = FORMAL / "sandhouse_correct/sandhouse_v3d_correct_seed0.mp4"
SHUFFLED = FORMAL / "sandhouse_shuffled/sandhouse_v3d_identity_shuffled_seed0.mp4"
TRACKS = FORMAL / "artifacts/sandhouse/sandhouse_v3d_correct_tracks.npy"
VIS = FORMAL / "artifacts/sandhouse/sandhouse_v3d_visibility.npy"
SOURCE = Path((FORMAL / "artifacts/sandhouse/input_image_path.txt").read_text().strip())

spec = importlib.util.spec_from_file_location(
    "frozen_eval",
    RUN / "eval_v3_sandhouse_core.py"
)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

# ------------------------------------------------------------
# Exact frozen common-domain transform:
# 832x480 -> 832x464 bicubic
# ------------------------------------------------------------
raw = np.load(RW)
assert raw.shape == (81,480,832,3)
assert raw.dtype == np.uint8

x = torch.from_numpy(raw).permute(0,3,1,2).float() / 255.0
x = F.interpolate(
    x,
    size=(464,832),
    mode="bicubic",
    align_corners=False,
    antialias=False,
).clamp(0,1)

RW_COMMON = x.permute(0,2,3,1).numpy().astype(np.float32)

assert RW_COMMON.shape == (81,464,832,3)
print("RW_COMMON_OK", RW_COMMON.shape, flush=True)

# ------------------------------------------------------------
# Patch ONLY video reader for .npy.
# Everything else remains frozen.
# ------------------------------------------------------------
reader_name = None

for name in ("read_video_common", "read_video"):
    if hasattr(ev, name):
        reader_name = name
        break

if reader_name is None:
    raise RuntimeError("cannot locate frozen video reader")

original_reader = getattr(ev, reader_name)

def patched_reader(path):
    p = Path(path)
    if p == RW:
        return RW_COMMON.copy()
    return original_reader(path)

setattr(ev, reader_name, patched_reader)

print("PATCHED_READER =", reader_name, flush=True)

# ------------------------------------------------------------
# SandHouse case
# ------------------------------------------------------------
ev.CANDIDATES = ["v3d"]

ev.CASES["sandhouse"] = {
    "source": str(SOURCE),
    "rw": str(RW),
    "old_correct": str(SHUFFLED),
    "tracks": str(TRACKS),
    "vis": str(VIS),
    "expect": {
        "n": 1791,
        "valid_tracks": 1791,
        "obs": 11383,
        "lab": float("nan"),
        "rgb": float("nan"),
        "tcme": float("nan"),
    },
}

def paths(root, suite, case):
    assert case == "sandhouse"
    return {
        "rw": RW,
        "old_correct": SHUFFLED,
        "v3d": CORRECT,
    }

ev.method_paths = paths

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["appearance","motion"], required=True)
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()

if args.mode == "appearance":
    r = ev.appearance_case(DT, DT, "sandhouse")
    out = {"cases":{"sandhouse":r}}
    (RUN / "sandhouse_rw_appearance.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    # Critical reproduction check
    v = r["methods"]["v3d"]["tc_mar_lab"]["mean"]
    assert abs(v - 23.294537291684335) < 1e-5, v

    print("V3D_TCMAR_REPRODUCTION_OK", v)
    print("RW_TCMAR =", r["methods"]["rw"]["tc_mar_lab"]["mean"])
    print("V3D_TCMAR =", v)
    print("RW_MINUS_V3D =", r["vs_realwonder"]["v3d"]["paired_mean_difference"])
    print("CI95 =", r["vs_realwonder"]["v3d"]["bootstrap_95_ci"])
    print("DECISION =", r["vs_realwonder"]["v3d"]["decision"])
    print("SANDHOUSE_RW_APPEARANCE_DONE")

else:
    r = ev.motion_case(DT, DT, "sandhouse", args.batch)
    (RUN / "sandhouse_rw_motion.json").write_text(
        json.dumps(r, indent=2) + "\n"
    )

    v = r["methods"]["v3d"]["transition_mean_epe_mean"]
    assert abs(v - 2.8482601992328838) < 5e-4, v

    print("V3D_TCME_REPRODUCTION_OK", v)
    print("RW_TCME =", r["methods"]["rw"]["transition_mean_epe_mean"])
    print("V3D_TCME =", v)
    print("RW_MINUS_V3D =", r["vs_realwonder"]["v3d"]["paired_mean_difference"])
    print("CI95 =", r["vs_realwonder"]["v3d"]["bootstrap_95_ci"])
    print("DECISION =", r["vs_realwonder"]["v3d"]["decision"])
    print("SANDHOUSE_RW_MOTION_DONE")
