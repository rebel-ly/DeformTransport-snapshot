import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from wan.modules.trajectory import (
    create_pos_feature_map,
    replace_feature,
)


def run(
    mode,
    tracks,
    visibility,
    depth,
    ids,
    feature,
):

    with tempfile.TemporaryDirectory() as td:

        td = Path(td)

        depth_path = td / "depth.npy"
        ids_path = td / "ids.npy"

        np.save(
            depth_path,
            depth[
                None
            ].astype(
                np.float32
            ),
        )

        np.save(
            ids_path,
            ids.astype(
                np.int64
            ),
        )

        os.environ[
            "DT_TRANSPORT_VARIANT"
        ] = mode

        os.environ[
            "DT_TRACK_DEPTH_PATH"
        ] = str(
            depth_path
        )

        os.environ[
            "DT_TRACK_IDS_PATH"
        ] = str(
            ids_path
        )

        torch.manual_seed(
            0
        )

        _, pos = (
            create_pos_feature_map(
                torch.from_numpy(
                    tracks
                ),
                torch.from_numpy(
                    visibility
                ),
                [
                    4,
                    8,
                    8,
                ],
                16,
                16,
                feature.shape[1],
                track_num=tracks.shape[1],
                device=feature.device,
            )
        )

        return replace_feature(
            feature.clone(),
            pos.unsqueeze(0),
        )


T = 81
N = 2

tracks = np.zeros(
    (
        T,
        N,
        2,
    ),
    np.float32,
)

visibility = np.ones(
    (
        T,
        N,
    ),
    bool,
)

depth = np.ones(
    (
        T,
        N,
    ),
    np.float32,
)

ids = np.array(
    [
        10,
        20,
    ],
    dtype=np.int64,
)

# Distinct source cells:
#
# track 0 -> source latent cell (0,0)
# track 1 -> source latent cell (0,1)
tracks[:, 0] = [
    4,
    4,
]

tracks[:, 1] = [
    12,
    4,
]

# At t>=4 both collide at latent cell (1,1).
tracks[
    4:,
    0,
] = [
    9,
    9,
]

tracks[
    4:,
    1,
] = [
    15,
    15,
]


feature = torch.zeros(
    1,
    2,
    21,
    2,
    2,
    dtype=torch.float32,
)

feature[
    0,
    :,
    0,
    0,
    0,
] = torch.tensor(
    [
        1.0,
        10.0,
    ]
)

feature[
    0,
    :,
    0,
    0,
    1,
] = torch.tensor(
    [
        3.0,
        30.0,
    ]
)


# ------------------------------------------------------------
# V3B
#
# target latent center = (12,12).
# track0 position = (9,9)
# track1 position = (15,15)
#
# Equal squared distance.
# Material ID 10 must win.
# ------------------------------------------------------------

out = run(
    "v3b",
    tracks,
    visibility,
    depth,
    ids,
    feature,
)

got = float(
    out[
        0,
        0,
        1,
        1,
        1,
    ]
)

assert abs(
    got
    - 1.0
) < 1e-6

print(
    "V3B_UNIT_TEST_OK"
)


# ------------------------------------------------------------
# V3C
#
# Track 1 is closer in camera depth.
# Must transport feature value 3.
# ------------------------------------------------------------

depth[:, 0] = 2.0
depth[:, 1] = 1.0

out = run(
    "v3c",
    tracks,
    visibility,
    depth,
    ids,
    feature,
)

got = float(
    out[
        0,
        0,
        1,
        1,
        1,
    ]
)

assert abs(
    got
    - 3.0
) < 1e-6

print(
    "V3C_UNIT_TEST_OK"
)


# ------------------------------------------------------------
# V3D
#
# Continuous source sampling smoke test.
# ------------------------------------------------------------

out = run(
    "v3d",
    tracks,
    visibility,
    depth,
    ids,
    feature,
)

assert torch.isfinite(
    out
).all()

print(
    "V3D_UNIT_TEST_OK"
)


# ------------------------------------------------------------
# V3E
#
# Local 3x3 transport smoke test.
# ------------------------------------------------------------

out = run(
    "v3e",
    tracks,
    visibility,
    depth,
    ids,
    feature,
)

assert torch.isfinite(
    out
).all()

print(
    "V3E_UNIT_TEST_OK"
)

print(
    "V3_SUITE_UNIT_TEST_OK"
)
