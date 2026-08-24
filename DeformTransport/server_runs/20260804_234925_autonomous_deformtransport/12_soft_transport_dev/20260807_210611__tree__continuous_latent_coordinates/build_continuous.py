from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from deform_transport.transport_ready import (
    validate_transport_ready,
)


src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
report_path = Path(sys.argv[3]).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


if not src.is_file():
    raise FileNotFoundError(src)

if dst.exists():
    raise FileExistsError(dst)

state = torch.load(
    src,
    map_location="cpu",
    weights_only=False,
)

validate_transport_ready(state)

required = [
    "source_points_2d_video",
    "points_2d_video",
    "source_points_2d_latent",
    "points_2d_latent",
    "video_width",
    "video_height",
    "latent_width",
    "latent_height",
]

missing = [k for k in required if k not in state]
if missing:
    raise KeyError(
        f"missing required keys: {missing}"
    )

video_w = int(state["video_width"])
video_h = int(state["video_height"])
latent_w = int(state["latent_width"])
latent_h = int(state["latent_height"])

scale_x = latent_w / video_w
scale_y = latent_h / video_h

print(
    "video_size =",
    (video_w, video_h),
)
print(
    "latent_size =",
    (latent_w, latent_h),
)
print(
    "scale_xy =",
    (scale_x, scale_y),
)

# Tree should preserve the exact Wan 8x spatial mapping.
assert video_w == 832, video_w
assert video_h == 480, video_h
assert latent_w == 104, latent_w
assert latent_h == 60, latent_h

assert abs(scale_x - 0.125) < 1e-12
assert abs(scale_y - 0.125) < 1e-12

source_video = state[
    "source_points_2d_video"
].to(
    dtype=torch.float32,
    device="cpu",
)

future_video = state[
    "points_2d_video"
].to(
    dtype=torch.float32,
    device="cpu",
)

source_cont = source_video.clone()
future_cont = future_video.clone()

source_cont[..., 0] *= scale_x
source_cont[..., 1] *= scale_y

future_cont[..., 0] *= scale_x
future_cont[..., 1] *= scale_y

source_disc = state[
    "source_points_2d_latent"
].to(
    dtype=torch.long,
    device="cpu",
)

future_disc = state[
    "points_2d_latent"
].to(
    dtype=torch.long,
    device="cpu",
)

source_floor = torch.floor(
    source_cont
).to(torch.long)

future_floor = torch.floor(
    future_cont
).to(torch.long)

source_floor_match = torch.equal(
    source_floor,
    source_disc,
)

future_floor_match = torch.equal(
    future_floor,
    future_disc,
)

if not source_floor_match:
    mismatch = (
        source_floor != source_disc
    ).any(dim=-1)

    raise RuntimeError(
        "source continuous/discrete floor "
        f"contract failed for "
        f"{int(mismatch.sum())} points"
    )

if not future_floor_match:
    mismatch = (
        future_floor != future_disc
    ).any(dim=-1)

    raise RuntimeError(
        "future continuous/discrete floor "
        f"contract failed for "
        f"{int(mismatch.sum())} point-states"
    )

if not bool(
    torch.isfinite(source_cont).all()
):
    raise RuntimeError(
        "source continuous coordinates "
        "contain NaN/Inf"
    )

if not bool(
    torch.isfinite(future_cont).all()
):
    raise RuntimeError(
        "future continuous coordinates "
        "contain NaN/Inf"
    )

source_frac = torch.abs(
    source_cont
    - torch.floor(source_cont)
)

future_frac = torch.abs(
    future_cont
    - torch.floor(future_cont)
)

source_noninteger_points = int(
    (
        source_frac > 1e-6
    ).any(dim=-1).sum()
)

future_noninteger_states = int(
    (
        future_frac > 1e-6
    ).any(dim=-1).sum()
)

if source_noninteger_points == 0:
    raise RuntimeError(
        "continuous source coordinates "
        "unexpectedly contain no fractions"
    )

if future_noninteger_states == 0:
    raise RuntimeError(
        "continuous future coordinates "
        "unexpectedly contain no fractions"
    )

# Add only new fields; preserve every existing field unchanged.
state[
    "source_points_2d_latent_continuous"
] = source_cont.contiguous()

state[
    "points_2d_latent_continuous"
] = future_cont.contiguous()

state[
    "continuous_coordinate_extension_version"
] = 1

state[
    "continuous_coordinate_mapping"
] = {
    "source": "source_points_2d_video",
    "target": "points_2d_video",
    "formula": (
        "latent_xy = video_xy * "
        "[latent_width/video_width, "
        "latent_height/video_height]"
    ),
    "scale_xy": [
        float(scale_x),
        float(scale_y),
    ],
    "expected_floor_contract": (
        "floor(latent_continuous) "
        "== discrete latent coordinate"
    ),
}

# Core transport contract must still validate.
validate_transport_ready(state)

dst.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torch.save(
    state,
    dst,
)

loaded = torch.load(
    dst,
    map_location="cpu",
    weights_only=False,
)

validate_transport_ready(loaded)

checks = {
    "source_floor_match":
        source_floor_match,

    "future_floor_match":
        future_floor_match,

    "source_continuous_shape":
        tuple(source_cont.shape)
        == tuple(source_disc.shape),

    "future_continuous_shape":
        tuple(future_cont.shape)
        == tuple(future_disc.shape),

    "source_finite":
        bool(
            torch.isfinite(
                source_cont
            ).all()
        ),

    "future_finite":
        bool(
            torch.isfinite(
                future_cont
            ).all()
        ),

    "source_has_fractional_points":
        source_noninteger_points > 0,

    "future_has_fractional_states":
        future_noninteger_states > 0,

    "saved_source_exact":
        torch.equal(
            loaded[
                "source_points_2d_latent_continuous"
            ],
            source_cont,
        ),

    "saved_future_exact":
        torch.equal(
            loaded[
                "points_2d_latent_continuous"
            ],
            future_cont,
        ),
}

report = {
    "status":
        "TREE_CONTINUOUS_COORDINATES_READY",

    "source_artifact":
        str(src),

    "source_sha256":
        sha256(src),

    "output_artifact":
        str(dst),

    "output_sha256":
        sha256(dst),

    "video_size": [
        video_h,
        video_w,
    ],

    "latent_size": [
        latent_h,
        latent_w,
    ],

    "scale_xy": [
        scale_x,
        scale_y,
    ],

    "source_shape":
        list(source_cont.shape),

    "future_shape":
        list(future_cont.shape),

    "source_noninteger_points":
        source_noninteger_points,

    "future_noninteger_states":
        future_noninteger_states,

    "checks":
        checks,

    "all_checks_pass":
        all(checks.values()),
}

report_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        report,
        indent=2,
    )
)

if not report[
    "all_checks_pass"
]:
    raise SystemExit(1)

print(
    "TREE_CONTINUOUS_COORDINATES_OK"
)
