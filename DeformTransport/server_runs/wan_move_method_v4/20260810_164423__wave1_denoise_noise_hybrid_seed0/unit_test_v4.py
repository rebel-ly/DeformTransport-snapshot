import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from wan.modules.trajectory import (
    create_pos_feature_map,
    dt_v4_transport,
)

# ----------------------------------------------------------
# Exact flow-prediction algebra.
# ----------------------------------------------------------

torch.manual_seed(123)

z = torch.randn(3, 4, 5)
v = torch.randn_like(z)

sigma = torch.tensor(
    0.73,
    dtype=torch.float32,
)

x0 = z - sigma * v

target = x0 + 0.03125

v2 = (z - target) / sigma
recover = z - sigma * v2

err = float(
    (recover - target)
    .abs()
    .max()
)

assert err < 1e-6, err

print(
    "V4_FLOW_X0_ALGEBRA_OK",
    "max_error=",
    err,
)

# ----------------------------------------------------------
# Tiny material transport test.
# ----------------------------------------------------------

T = 81
N = 2

tracks = np.zeros(
    (T, N, 2),
    dtype=np.float32,
)

vis = np.ones(
    (T, N),
    dtype=bool,
)

depth = np.ones(
    (T, N),
    dtype=np.float32,
)

ids = np.array(
    [10, 20],
    dtype=np.int64,
)

# Source positions.
tracks[:, 0] = [4, 4]
tracks[:, 1] = [20, 4]

# Future target cells.
tracks[4:, 0] = [12, 12]
tracks[4:, 1] = [28, 12]

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    np.save(
        td / "depth.npy",
        depth[None],
    )

    np.save(
        td / "ids.npy",
        ids,
    )

    os.environ[
        "DT_TRANSPORT_VARIANT"
    ] = "v3d"

    os.environ[
        "DT_TRACK_DEPTH_PATH"
    ] = str(td / "depth.npy")

    os.environ[
        "DT_TRACK_IDS_PATH"
    ] = str(td / "ids.npy")

    torch.manual_seed(0)

    _, pos = create_pos_feature_map(
        torch.from_numpy(tracks),
        torch.from_numpy(vis),
        [4, 8, 8],
        32,
        32,
        4,
        track_num=N,
        device=torch.device("cpu"),
    )

    x = torch.zeros(
        4,
        21,
        4,
        4,
        dtype=torch.float32,
    )

    # Non-zero source frame.
    x[:, 0] = torch.arange(
        4 * 4 * 4,
        dtype=torch.float32,
    ).reshape(4, 4, 4)

    y = dt_v4_transport(
        x,
        pos.unsqueeze(0),
        patch_radius=0,
        mix=1.0,
    )

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    changed = float(
        (y[:, 1:] - x[:, 1:])
        .abs()
        .sum()
    )

    assert changed > 0.0

    ypatch = dt_v4_transport(
        x,
        pos.unsqueeze(0),
        patch_radius=1,
        mix=0.2,
    )

    assert ypatch.shape == x.shape
    assert torch.isfinite(ypatch).all()

print(
    "V4_MATERIAL_TRANSPORT_UNIT_TEST_OK"
)

print(
    "V4_WAVE1_UNIT_TEST_OK"
)
