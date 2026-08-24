import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append(os.path.abspath('.'))
import math
import random
from io import BytesIO

import imageio.v3 as iio
import numpy as np

import torch
import decord
from decord import VideoReader
from PIL import Image, ImageDraw
from torchvision import transforms



SKIP_ZERO = False

def get_pos_emb(
    pos_k: torch.Tensor,
    pos_emb_dim: int,
    theta_func: callable = lambda i, d: torch.pow(10000, torch.mul(2, torch.div(i.to(torch.float32), d))),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate batch position embeddings.
    
    Args:
        pos_k (torch.Tensor): A 1D tensor containing positions for which to generate embeddings.
        pos_emb_dim (int): The dimension of position embeddings.
        theta_func (callable): Function to compute thetas based on position and embedding dimensions.
        device (torch.device): Device to store the position embeddings.
        dtype (torch.dtype): Desired data type for computations.
    
    Returns:
        torch.Tensor: The position embeddings with shape (batch_size, pos_emb_dim).
    """
    assert pos_emb_dim % 2 == 0, "The dimension of position embeddings must be even."
    pos_k = pos_k.to(device, dtype)
    if SKIP_ZERO:
        pos_k = pos_k + 1
    batch_size = pos_k.size(0)

    denominator = torch.arange(0, pos_emb_dim // 2, device=device, dtype=dtype)
    # Expand denominator to match the shape needed for broadcasting
    denominator_expanded = denominator.view(1, -1).expand(batch_size, -1)
    
    thetas = theta_func(denominator_expanded, pos_emb_dim)
    
    # Ensure pos_k is in the correct shape for broadcasting
    pos_k_expanded = pos_k.view(-1, 1).to(dtype)
    sin_thetas = torch.sin(torch.div(pos_k_expanded, thetas))
    cos_thetas = torch.cos(torch.div(pos_k_expanded, thetas))

    # Concatenate sine and cosine embeddings along the last dimension
    pos_emb = torch.cat([sin_thetas, cos_thetas], dim=-1)

    return pos_emb


_DT_CONTEXT = None


def _dt_mode():
    import os

    return (
        os.environ
        .get(
            "DT_TRANSPORT_VARIANT",
            "baseline",
        )
        .strip()
        .lower()
    )


def _dt_load_sidecars(
    n,
    device,
):
    import os
    import numpy as np

    depth_path = os.environ.get(
        "DT_TRACK_DEPTH_PATH",
        "",
    )

    ids_path = os.environ.get(
        "DT_TRACK_IDS_PATH",
        "",
    )

    if (
        not depth_path
        or not ids_path
    ):
        raise RuntimeError(
            "DT custom transport requires "
            "DT_TRACK_DEPTH_PATH and "
            "DT_TRACK_IDS_PATH"
        )

    depth_np = np.load(
        depth_path
    )

    ids_np = np.load(
        ids_path
    )

    if (
        depth_np.ndim == 3
        and depth_np.shape[0] == 1
    ):
        depth_np = depth_np[0]

    if depth_np.shape != (
        81,
        n,
    ):
        raise RuntimeError(
            f"depth sidecar shape "
            f"{depth_np.shape}, "
            f"expected (81,{n})"
        )

    if ids_np.shape != (
        n,
    ):
        raise RuntimeError(
            f"ID sidecar shape "
            f"{ids_np.shape}, "
            f"expected ({n},)"
        )

    depth = (
        torch
        .from_numpy(
            depth_np
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    ids = (
        torch
        .from_numpy(
            ids_np.astype(
                np.int64
            )
        )
        .to(
            device=device,
        )
    )

    return (
        depth,
        ids,
    )


def create_pos_feature_map(
    pred_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    downsample_ratios: list[int],
    height: int,
    width: int,
    pos_emb_dim: int,
    track_num: int = -1,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
    dtype: torch.dtype = torch.float32,
):

    global _DT_CONTEXT

    assert (
        t_down_strategy
        in [
            "sample",
            "average",
        ]
    ), "Invalid strategy"

    t, n, _ = pred_tracks.shape

    (
        t_down,
        h_down,
        w_down,
    ) = downsample_ratios

    feature_map = torch.zeros(
        (
            (t - 1)
            // t_down
            + 1,

            height
            // h_down,

            width
            // w_down,

            pos_emb_dim,
        ),
        device=device,
        dtype=dtype,
    )

    track_pos = (
        -torch.ones(
            n,
            (t - 1)
            // t_down
            + 1,
            2,
            dtype=torch.long,
        )
    )

    if track_num == -1:
        track_num = n

    # Preserve original Wan-Move RNG consumption.
    tracks_idx = (
        torch.randperm(n)[
            :track_num
        ]
    )

    tracks = pred_tracks[
        :,
        tracks_idx,
    ]

    visibility = pred_visibility[
        :,
        tracks_idx,
    ]

    # Preserve second original randperm.
    tracks_embs = get_pos_emb(
        torch.randperm(n)[
            :track_num
        ],
        pos_emb_dim,
        device=device,
        dtype=dtype,
    )

    for t_idx in range(
        0,
        t,
        t_down,
    ):

        if (
            t_down_strategy
            == "sample"
            or t_idx == 0
        ):
            cur_tracks = tracks[
                t_idx
            ]

            cur_visibility = visibility[
                t_idx
            ]

        else:
            cur_tracks = (
                tracks[
                    t_idx:
                    t_idx
                    + t_down
                ]
                .mean(
                    dim=0
                )
            )

            cur_visibility = (
                torch.any(
                    visibility[
                        t_idx:
                        t_idx
                        + t_down
                    ],
                    dim=0,
                )
            )

        for i in range(
            track_num
        ):

            if (
                not cur_visibility[i]
                or cur_tracks[i][0] < 0
                or cur_tracks[i][1] < 0
                or cur_tracks[i][0] >= width
                or cur_tracks[i][1] >= height
            ):
                continue

            x, y = cur_tracks[i]

            x = int(
                x
                // w_down
            )

            y = int(
                y
                // h_down
            )

            feature_map[
                t_idx
                // t_down,
                y,
                x,
            ] += tracks_embs[i]

            track_pos[
                i,
                t_idx
                // t_down,
                0,
            ] = y

            track_pos[
                i,
                t_idx
                // t_down,
                1,
            ] = x


    mode = _dt_mode()

    if mode in {
        "v3b",
        "v3c",
        "v3d",
        "v3e",
    }:

        if (
            t_down_strategy
            != "sample"
        ):
            raise RuntimeError(
                "DT suite is frozen "
                "to Wan-Move sample "
                "temporal downsampling"
            )

        depth, ids = (
            _dt_load_sidecars(
                n,
                pred_tracks.device,
            )
        )

        idx_device = (
            tracks_idx
            .to(
                pred_tracks.device
            )
        )

        _DT_CONTEXT = {
            "tracks":
                tracks,

            "visibility":
                visibility,

            "depth":
                depth[
                    :,
                    idx_device,
                ],

            "ids":
                ids[
                    idx_device
                ],

            "t_down":
                int(
                    t_down
                ),

            "h_down":
                int(
                    h_down
                ),

            "w_down":
                int(
                    w_down
                ),

            "height":
                int(
                    height
                ),

            "width":
                int(
                    width
                ),
        }

    else:
        _DT_CONTEXT = None

    return (
        feature_map,
        track_pos,
    )


def _dt_original_replace(
    vae_feature,
    track_pos,
):

    b, _, t, h, w = (
        vae_feature.shape
    )

    assert (
        b
        == track_pos.shape[0]
    ), "Batch size mismatch."

    n = track_pos.shape[1]

    # Preserve original third randperm.
    track_pos = track_pos[
        :,
        torch.randperm(n),
        :,
        :,
    ]

    current_pos = (
        track_pos[
            :,
            :,
            1:,
            :,
        ]
    )

    mask = (
        (current_pos[..., 0] >= 0)
        &
        (current_pos[..., 1] >= 0)
    )

    valid_indices = (
        mask
        .nonzero(
            as_tuple=False
        )
    )

    if (
        valid_indices.shape[0]
        == 0
    ):
        return vae_feature

    batch_idx = valid_indices[
        :,
        0,
    ]

    track_idx = valid_indices[
        :,
        1,
    ]

    t_rel = valid_indices[
        :,
        2,
    ]

    t_target = (
        t_rel
        + 1
    )

    h_target = (
        current_pos[
            batch_idx,
            track_idx,
            t_rel,
            0,
        ]
        .long()
    )

    w_target = (
        current_pos[
            batch_idx,
            track_idx,
            t_rel,
            1,
        ]
        .long()
    )

    h_source = (
        track_pos[
            batch_idx,
            track_idx,
            0,
            0,
        ]
        .long()
    )

    w_source = (
        track_pos[
            batch_idx,
            track_idx,
            0,
            1,
        ]
        .long()
    )

    src_features = (
        vae_feature[
            batch_idx,
            :,
            0,
            h_source,
            w_source,
        ]
    )

    vae_feature[
        batch_idx,
        :,
        t_target,
        h_target,
        w_target,
    ] = src_features

    return vae_feature


def _dt_bilinear_source_features(
    vae_feature,
    source_xy,
    stride_y,
    stride_x,
):

    # Treat the discrete VAE condition latent as a
    # cell-centered continuous feature field.
    #
    # latent index:
    # pixel / stride - 0.5
    #
    # This is a fixed geometric convention.
    # No tunable alpha is introduced.

    source_xy = (
        source_xy
        .to(
            device=vae_feature.device,
            dtype=torch.float32,
        )
    )

    _, c, _, h, w = (
        vae_feature.shape
    )

    lx = (
        source_xy[:, 0]
        / float(
            stride_x
        )
        - 0.5
    )

    ly = (
        source_xy[:, 1]
        / float(
            stride_y
        )
        - 0.5
    )

    if w > 1:
        gx = (
            lx
            * (
                2.0
                / (
                    w - 1
                )
            )
            - 1.0
        )
    else:
        gx = torch.zeros_like(
            lx
        )

    if h > 1:
        gy = (
            ly
            * (
                2.0
                / (
                    h - 1
                )
            )
            - 1.0
        )
    else:
        gy = torch.zeros_like(
            ly
        )

    grid = torch.stack(
        [
            gx,
            gy,
        ],
        dim=-1,
    ).view(
        1,
        1,
        -1,
        2,
    )

    src = (
        vae_feature[
            0,
            :,
            0,
        ]
        .unsqueeze(0)
    )

    sampled = (
        torch.nn.functional.grid_sample(
            src,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    )

    # [1,C,1,N]
    # -> [N,C]
    return (
        sampled[
            0,
            :,
            0,
            :,
        ]
        .transpose(
            0,
            1
        )
        .contiguous()
    )


def replace_feature(
    vae_feature: torch.Tensor,
    track_pos: torch.Tensor,
) -> torch.Tensor:

    import math

    global _DT_CONTEXT

    mode = _dt_mode()

    if mode in {
        "baseline",
        "v3s",
        "",
    }:
        return _dt_original_replace(
            vae_feature,
            track_pos,
        )

    if mode not in {
        "v3b",
        "v3c",
        "v3d",
        "v3e",
    }:
        raise RuntimeError(
            f"unknown "
            f"DT_TRANSPORT_VARIANT="
            f"{mode}"
        )

    if _DT_CONTEXT is None:
        raise RuntimeError(
            "DT transport context missing; "
            "create_pos_feature_map "
            "must run first"
        )

    if (
        vae_feature.shape[0]
        != 1
        or track_pos.shape[0]
        != 1
    ):
        raise RuntimeError(
            "DT suite freezes "
            "batch size 1"
        )

    n = track_pos.shape[1]

    # Preserve original Wan-Move third randperm
    # RNG consumption.
    #
    # Reorder every associated material field
    # identically.
    perm = torch.randperm(
        n
    )

    track_pos = track_pos[
        :,
        perm,
        :,
        :,
    ]

    perm_device = perm.to(
        _DT_CONTEXT[
            "tracks"
        ].device
    )

    tracks = (
        _DT_CONTEXT[
            "tracks"
        ][
            :,
            perm_device,
        ]
    )

    visibility = (
        _DT_CONTEXT[
            "visibility"
        ][
            :,
            perm_device,
        ]
    )

    depth = (
        _DT_CONTEXT[
            "depth"
        ][
            :,
            perm_device,
        ]
    )

    ids = (
        _DT_CONTEXT[
            "ids"
        ][
            perm_device
        ]
    )

    t_down = (
        _DT_CONTEXT[
            "t_down"
        ]
    )

    stride_y = (
        _DT_CONTEXT[
            "h_down"
        ]
    )

    stride_x = (
        _DT_CONTEXT[
            "w_down"
        ]
    )

    tracks_sampled = (
        tracks[
            ::t_down
        ]
    )

    vis_sampled = (
        visibility[
            ::t_down
        ]
    )

    depth_sampled = (
        depth[
            ::t_down
        ]
    )

    latent_t = (
        track_pos.shape[2]
    )

    assert (
        tracks_sampled.shape[0]
        == latent_t
    )

    assert (
        depth_sampled.shape[0]
        == latent_t
    )

    src_h = (
        track_pos[
            0,
            :,
            0,
            0,
        ]
        .to(
            tracks.device
        )
    )

    src_w = (
        track_pos[
            0,
            :,
            0,
            1,
        ]
        .to(
            tracks.device
        )
    )

    src_valid = (
        (src_h >= 0)
        &
        (src_w >= 0)
    )

    if mode == "v3d":

        source_features = (
            _dt_bilinear_source_features(
                vae_feature,
                tracks_sampled[0],
                stride_y,
                stride_x,
            )
        )

    else:
        source_features = None

    H = vae_feature.shape[-2]
    W = vae_feature.shape[-1]

    for tau in range(
        1,
        latent_t,
    ):

        target_h = (
            track_pos[
                0,
                :,
                tau,
                0,
            ]
            .to(
                tracks.device
            )
        )

        target_w = (
            track_pos[
                0,
                :,
                tau,
                1,
            ]
            .to(
                tracks.device
            )
        )

        valid = (
            src_valid
            &
            (target_h >= 0)
            &
            (target_w >= 0)
            &
            vis_sampled[tau]
        )

        indices = torch.where(
            valid
        )[0]

        if (
            indices.numel()
            == 0
        ):
            continue


        # ====================================================
        # V3B / V3C / V3D:
        # exactly one source material identity per target cell.
        # ====================================================

        if mode in {
            "v3b",
            "v3c",
            "v3d",
        }:

            groups = {}

            for ii in indices.tolist():

                hh = int(
                    target_h[
                        ii
                    ].item()
                )

                ww = int(
                    target_w[
                        ii
                    ].item()
                )

                groups.setdefault(
                    (
                        hh,
                        ww,
                    ),
                    [],
                ).append(
                    ii
                )

            for (
                hh,
                ww,
            ), members in groups.items():

                best = None
                best_key = None

                for ii in members:

                    material_id = int(
                        ids[
                            ii
                        ].item()
                    )

                    if mode == "v3b":

                        x = float(
                            tracks_sampled[
                                tau,
                                ii,
                                0,
                            ].item()
                        )

                        y = float(
                            tracks_sampled[
                                tau,
                                ii,
                                1,
                            ].item()
                        )

                        # Squared quantization error
                        # to target latent-cell center.
                        score = (
                            (
                                x
                                - (
                                    ww
                                    + 0.5
                                )
                                * stride_x
                            ) ** 2
                            +
                            (
                                y
                                - (
                                    hh
                                    + 0.5
                                )
                                * stride_y
                            ) ** 2
                        )

                    else:

                        score = float(
                            depth_sampled[
                                tau,
                                ii,
                            ].item()
                        )

                        if (
                            not math.isfinite(
                                score
                            )
                            or score <= 0
                        ):
                            continue

                    key = (
                        score,
                        material_id,
                    )

                    if (
                        best_key is None
                        or key
                        < best_key
                    ):
                        best_key = key
                        best = ii

                if best is None:
                    continue

                if mode == "v3d":

                    feat = (
                        source_features[
                            best
                        ]
                    )

                else:

                    source_h = int(
                        src_h[
                            best
                        ].item()
                    )

                    source_w = int(
                        src_w[
                            best
                        ].item()
                    )

                    feat = (
                        vae_feature[
                            0,
                            :,
                            0,
                            source_h,
                            source_w,
                        ]
                    )

                vae_feature[
                    0,
                    :,
                    tau,
                    hh,
                    ww,
                ] = feat


        # ====================================================
        # V3E:
        #
        # depth-aware 3x3 local source-condition patch.
        #
        # Every target latent cell still has exactly one
        # winning material contributor.
        # ====================================================

        else:

            groups = {}

            for ii in indices.tolist():

                source_h0 = int(
                    src_h[
                        ii
                    ].item()
                )

                source_w0 = int(
                    src_w[
                        ii
                    ].item()
                )

                target_h0 = int(
                    target_h[
                        ii
                    ].item()
                )

                target_w0 = int(
                    target_w[
                        ii
                    ].item()
                )

                z = float(
                    depth_sampled[
                        tau,
                        ii,
                    ].item()
                )

                if (
                    not math.isfinite(
                        z
                    )
                    or z <= 0
                ):
                    continue

                material_id = int(
                    ids[
                        ii
                    ].item()
                )

                for dy in (
                    -1,
                    0,
                    1,
                ):
                    for dx in (
                        -1,
                        0,
                        1,
                    ):

                        source_h = (
                            source_h0
                            + dy
                        )

                        source_w = (
                            source_w0
                            + dx
                        )

                        hh = (
                            target_h0
                            + dy
                        )

                        ww = (
                            target_w0
                            + dx
                        )

                        if not (
                            0 <= source_h < H
                            and 0 <= source_w < W
                            and 0 <= hh < H
                            and 0 <= ww < W
                        ):
                            continue

                        groups.setdefault(
                            (
                                hh,
                                ww,
                            ),
                            [],
                        ).append(
                            (
                                z,
                                material_id,
                                source_h,
                                source_w,
                            )
                        )

            for (
                hh,
                ww,
            ), candidates in groups.items():

                (
                    z,
                    material_id,
                    source_h,
                    source_w,
                ) = min(
                    candidates,
                    key=lambda item: (
                        item[0],
                        item[1],
                    ),
                )

                vae_feature[
                    0,
                    :,
                    tau,
                    hh,
                    ww,
                ] = (
                    vae_feature[
                        0,
                        :,
                        0,
                        source_h,
                        source_w,
                    ]
                )

    return vae_feature

def get_video_track_video(
    model,
    video_tensor: torch.Tensor, # [T, C, H, W]
    downsample_ratios: list[int],
    pos_emb_dim: int,
    grid_size: int = 32,
    track_num: int = -1,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the track video from the video tensor.

    Args:
    - model: torch.nn.Module, the model for tracking, CoTracker
    - video_tensor: torch.Tensor, the video tensor, [T, C, H, W]
    - downsample_ratios: list[int], the ratios for downsampling time, height, and width
    - height: int, the height of the feature map
    - width: int, the width of the feature map
    - pos_emb_dim: int, the dimension of the position embeddings
    - grid_size: int, the size of the grid
    - track_num: int, the number of tracks to use
    - t_down_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    -  track_video: torch.Tensor, the track video, [pos_emb_dim, T', H', W']
    -  track_pos: torch.Tensor, the position embeddings, [N, T', 2], 2 = height, width
    -  pred_tracks: the predicted point trajectories
    -  pred_visibility: visibility of the predicted point trajectories
    """

    t, c, height, width = video_tensor.shape
    with (
        torch.autocast(device_type=device.type, dtype=dtype),
        torch.no_grad(),
    ):
        pred_tracks, pred_visibility = model(
            video_tensor.unsqueeze(0),
            grid_size=grid_size,
            backward_tracking=False,
        ) 
    
    track_video, track_pos = create_pos_feature_map(
        pred_tracks[0], pred_visibility[0], downsample_ratios, height, width, pos_emb_dim, track_num, t_down_strategy, device, dtype
    )

    return track_video.permute(3, 0, 1, 2), track_pos, pred_tracks, pred_visibility


# === user input tracks ===

def resize_tracks(
    img_tracks: torch.Tensor, # [T, N, height, width]
    target_frame_num: int,
    t_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resize tracks to a specified number of frames.

    Args:
    - img_tracks: torch.Tensor, the tracks, [T, N, height, width]
    - target_frame_num: int, the number of frames to resize to
    - t_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    - resized_tracks: torch.Tensor, the resized tracks, [target_frame_num, N, 2]
    - resized_visibility: torch.Tensor, the resized visibility, [target_frame_num, N]
    """

    assert t_strategy in ["sample", "average"], "Invalid strategy for downsampling time dimension."
    assert t_strategy in ["sample"], "only support sample strategy."

    def get_xy_from_hw(hw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the x and y coordinates from the height and width tensor.

        Args:
        - hw_tensor: torch.Tensor, the tensor of height and width, [N, height, width]

        Returns:
        - xy_tensor: torch.Tensor, the tensor of x and y coordinates, [N, 2]
        """

        h, w = hw_tensor.shape[-2:]
        _, y, x = torch.nonzero(hw_tensor, as_tuple=True)
        xy_tensor = torch.stack((x, y), dim=-1)
        assert xy_tensor.shape[0] == hw_tensor.shape[0], "The number of points should be the same."
        return xy_tensor

    def get_average_xy_from_batch_hw(hw_tensor: torch.Tensor) -> torch.Tensor:
        for b in range(hw_tensor.shape[0]):
            xy_tensor = get_xy_from_hw(hw_tensor[b])
            if b == 0:
                xy_tensors = xy_tensor
            else:
                xy_tensors += xy_tensor
        xy_tensors /= hw_tensor.shape[0]
        return xy_tensors
    
    # Get the number of frames in the input tracks
    num_frames, num_tracks, _, _ = img_tracks.shape

    new_tracks = torch.zeros(target_frame_num, num_tracks, 2, device=device, dtype=dtype)
    new_visibility = torch.ones(target_frame_num, num_tracks, device=device, dtype=torch.bool)

    new_tracks[0] = get_xy_from_hw(img_tracks[0])
    # -1 for removing the first frame
    num_frames -= 1
    target_frame_num -= 1

    new_frame_idx = 1
    if target_frame_num <= num_frames:
        t_down = num_frames / target_frame_num
        frame_idxs = [int((i - 1) * t_down + 1) for i in range(1, target_frame_num + 1)]
        for i, frame_idx in enumerate(frame_idxs):
            if t_strategy == "sample":
                new_tracks[new_frame_idx] = get_xy_from_hw(img_tracks[frame_idx])
            else:
                next_frame_idx = frame_idxs[i + 1] if i + 1 < len(frame_idxs) else num_frames + 1 # +1 as compensation for the -1
                new_tracks[new_frame_idx] = get_average_xy_from_batch_hw(img_tracks[frame_idx:next_frame_idx])

            new_frame_idx += 1
    else:
        t_repeat = target_frame_num / num_frames
        target_frame_idxs = [int((i - 1) * t_repeat + 1) for i in range(1, num_frames + 1)]
        for i, target_frame_idx in enumerate(target_frame_idxs):
            next_target_frame_idx = target_frame_idxs[i + 1] if i + 1 < len(target_frame_idxs) else target_frame_num + 1
            if t_strategy == "sample":
                new_tracks[target_frame_idx:next_target_frame_idx] = get_xy_from_hw(img_tracks[new_frame_idx])
            else:
                if target_frame_idx == next_target_frame_idx:
                    new_tracks[target_frame_idx] = get_xy_from_hw(img_tracks[new_frame_idx])
                else:
                    next_new_frame_idx = new_frame_idx + 1 if new_frame_idx + 1 < num_frames else new_frame_idx
                    for j in range(target_frame_idx, next_target_frame_idx):
                        new_tracks[j] = (1 - (next_target_frame_idx - j) / (next_target_frame_idx - target_frame_idx)) * get_xy_from_hw(img_tracks[new_frame_idx]) + (next_target_frame_idx - j) / (next_target_frame_idx - target_frame_idx) * get_xy_from_hw(img_tracks[next_new_frame_idx])

            new_frame_idx += 1

    # print(new_tracks[1])
    return new_tracks, new_visibility

def generate_custom_feature_map(
    img_tracks: torch.Tensor, # [T, N, height, width]
    target_frame_num: int,
    downsample_ratios: list[int],
    pos_emb_dim: int,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a custom feature map from the tracks.

    Args:
    - img_tracks: torch.Tensor, the tracks, [T, N, height, width]
    - target_frame_num: int, the number of frames to resize to
    - downsample_ratios: List[int], the ratios for downsampling time, height, and width
    - pos_emb_dim: int, the dimension of the position embeddings
    - t_down_strategy: str, the strategy for downsampling time dimension
    - device: torch.device, the device
    - dtype: torch.dtype, the data type

    Returns:
    - feature_map: torch.Tensor, the feature map, [T', H', W', pos_emb_dim]
    """

    height, width = img_tracks.shape[-2:]
    resized_tracks, resized_visibility = resize_tracks(img_tracks, target_frame_num, t_down_strategy, device, dtype)
    feature_map, track_pos = create_pos_feature_map(
        resized_tracks, 
        resized_visibility, 
        downsample_ratios, 
        height, 
        width, 
        pos_emb_dim, 
        track_num=-1, 
        t_down_strategy=t_down_strategy, 
        device=device, 
        dtype=dtype
    )

    return feature_map, track_pos



# ---------------------------
# Visualize functions
# --------------------------


def draw_overall_gradient_polyline_on_image(image, line_width, points, start_color):
    """
    - image (Image): target image to draw on.
    - line_width (int): initial line width.
    - points (list of tuples): list of points forming the polyline, each point is (x, y).
    - start_color (tuple): starting color of the line (R, G, B).

    Return:
    - Image: original image with the gradient polyline drawn.
    """
    
    def get_distance(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    # Create a new image with the same size as the original
    new_image = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(new_image, 'RGBA')
    points = points[::-1]

    # Compute total length
    total_length = sum(get_distance(points[i], points[i+1]) for i in range(len(points)-1))

    # Accumulated length
    accumulated_length = 0

    # Draw the gradient polyline
    for start_point, end_point in zip(points[:-1], points[1:]):
        segment_length = get_distance(start_point, end_point)
        steps = int(segment_length)

        for i in range(steps):
            # Current accumulated length
            current_length = accumulated_length + (i / steps) * segment_length

            # Alpha from fully opaque to fully transparent
            alpha = int(255 * (1 - current_length / total_length))
            color = (*start_color, alpha)

            # Interpolated coordinates
            x = int(start_point[0] + (end_point[0] - start_point[0]) * i / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * i / steps)

            # Dynamic line width, decreasing from initial width to 1
            dynamic_line_width = int(line_width * (1 - (current_length / total_length)))
            dynamic_line_width = max(dynamic_line_width, 1)  # minimum width is 1 to avoid 0

            draw.line([(x, y), (x + 1, y)], fill=color, width=dynamic_line_width)

        accumulated_length += segment_length

    return new_image
   
def add_weighted(rgb, track):
    rgb = np.array(rgb) # [H, W, C] "RGB"
    track = np.array(track) # [H, W, C] "RGBA"
    
    # Compute weights from the alpha channel
    alpha = track[:, :, 3] / 255.0

    # Expand alpha to 3 channels to match RGB
    alpha = np.stack([alpha] * 3, axis=-1)

    # Blend the two images
    blend_img = track[:, :, :3] * alpha + rgb * (1 - alpha)
    
    return Image.fromarray(blend_img.astype(np.uint8))
        
def draw_tracks_on_video(video, tracks, visibility=None, track_frame=24):
    color_map = [
        (102, 153, 255), # Blue-ish
        (0, 255, 255),   # Cyan
        (255, 255, 0),   # Yellow
        (255, 102, 204), # Pink
        (0, 255, 0),     # Green
        (255, 0, 0),     # Red
        (128, 0, 128),   # Purple
        (255, 165, 0),   # Orange
        (255, 255, 255), # White
        (165, 42, 42)    # Brown
    ]
    circle_size = 12
    line_width = 16
    
    video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy() # (81, 480, 832, 3), uint8
    tracks = tracks[0].long().detach().cpu().numpy()
    if visibility is not None:
        visibility = visibility[0].detach().cpu().numpy()
    # print(video.shape, tracks.shape)
    
    output_frames = []
    # Process the video
    for t in range(video.shape[0]):
        # Extract current frame
        frame = video[t]
        frame = Image.fromarray(frame).convert("RGB")
        
        # Draw tracks
        for n in range(tracks.shape[1]):
            if visibility is not None and visibility[t, n] == 0:
                continue
            
            # Track coordinate at current frame
            track_coord = tracks[t, n]
            tracks_coord = tracks[max(t-track_frame, 0):t+1, n]
            
            # Draw a circle
            draw = ImageDraw.Draw(frame)
            draw.ellipse((track_coord[0] - circle_size, track_coord[1] - circle_size, track_coord[0] + circle_size, track_coord[1] + circle_size), fill=color_map[n % len(color_map)])
            # Draw the polyline
            track_image = draw_overall_gradient_polyline_on_image(frame, line_width, tracks_coord, color_map[n % len(color_map)])
            frame = add_weighted(frame, track_image)
        
        # Save current frame
        output_frames.append(frame.convert("RGB"))
        
    return output_frames
