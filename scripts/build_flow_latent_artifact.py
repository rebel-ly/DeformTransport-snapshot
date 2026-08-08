#!/usr/bin/env python3
"""Build a RAFT dense-flow latent-transport artifact for RealWonder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_flow(flow: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a [2,H,W] flow field at floating-point XY positions."""
    h, w = flow.shape[-2:]

    if w > 1:
        gx = xy[:, 0] * (2.0 / (w - 1)) - 1.0
    else:
        gx = torch.zeros_like(xy[:, 0])

    if h > 1:
        gy = xy[:, 1] * (2.0 / (h - 1)) - 1.0
    else:
        gy = torch.zeros_like(xy[:, 1])

    grid = torch.stack((gx, gy), dim=-1).reshape(1, 1, -1, 2)

    sampled = F.grid_sample(
        flow.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return sampled[0, :, 0, :].T.contiguous()


def nearest_splat(
    features: torch.Tensor,
    xy: torch.Tensor,
    valid: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-cell forward splat with mean aggregation for collisions."""

    finite = torch.isfinite(xy).all(dim=-1)

    continuous_in_bounds = (
        finite
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )

    cells = torch.floor(xy + 0.5).long()

    rounded_in_bounds = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < width)
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < height)
    )

    keep = valid & continuous_in_bounds & rounded_in_bounds

    channels = features.shape[1]
    flat_size = height * width

    sums = torch.zeros(
        (channels, flat_size),
        dtype=features.dtype,
        device=features.device,
    )

    counts = torch.zeros(
        (1, flat_size),
        dtype=torch.long,
        device=features.device,
    )

    if bool(keep.any()):
        kept_cells = cells[keep]
        linear = kept_cells[:, 1] * width + kept_cells[:, 0]

        sums.scatter_add_(
            1,
            linear.unsqueeze(0).expand(channels, -1),
            features[keep].T,
        )

        counts.scatter_add_(
            1,
            linear.unsqueeze(0),
            torch.ones(
                (1, linear.numel()),
                dtype=torch.long,
                device=features.device,
            ),
        )

    mask = counts > 0
    grid = sums / counts.clamp_min(1).to(features.dtype)

    return (
        grid.reshape(channels, height, width),
        mask.reshape(1, height, width),
        counts.reshape(1, height, width),
    )


def build_transport(
    source_grid: torch.Tensor,
    source_support: torch.Tensor,
    flows: torch.Tensor,
    endpoints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    channels, height, width = source_grid.shape
    flow_h, flow_w = flows.shape[-2:]

    latent_flows = F.interpolate(
        flows,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    latent_flows[:, 0] *= width / flow_w
    latent_flows[:, 1] *= height / flow_h

    yx = torch.nonzero(source_support[0], as_tuple=False)

    if yx.numel() == 0:
        raise ValueError("source support is empty")

    xy = torch.stack(
        (yx[:, 1], yx[:, 0]),
        dim=-1,
    ).float()

    features = source_grid[
        :,
        yx[:, 0],
        yx[:, 1],
    ].T.contiguous()

    valid = torch.ones(
        xy.shape[0],
        dtype=torch.bool,
        device=xy.device,
    )

    endpoint_to_slot = {
        int(frame): slot
        for slot, frame in enumerate(endpoints.tolist())
    }

    transported = [None] * endpoints.numel()
    masks = [None] * endpoints.numel()
    counts = [None] * endpoints.numel()
    valid_counts = []

    grid, mask, count = nearest_splat(
        features,
        xy,
        valid,
        height,
        width,
    )

    transported[0] = grid
    masks[0] = mask
    counts[0] = count
    valid_counts.append(int(valid.sum()))

    for step in range(flows.shape[0]):
        displacement = sample_flow(
            latent_flows[step],
            xy,
        )

        valid = (
            valid
            & torch.isfinite(displacement).all(dim=-1)
        )

        xy = xy + torch.where(
            valid[:, None],
            displacement,
            torch.zeros_like(displacement),
        )

        valid = (
            valid
            & torch.isfinite(xy).all(dim=-1)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )

        pixel_frame = step + 1

        if pixel_frame in endpoint_to_slot:
            slot = endpoint_to_slot[pixel_frame]

            grid, mask, count = nearest_splat(
                features,
                xy,
                valid,
                height,
                width,
            )

            transported[slot] = grid
            masks[slot] = mask
            counts[slot] = count
            valid_counts.append(int(valid.sum()))

    if any(
        item is None
        for item in transported + masks + counts
    ):
        raise RuntimeError(
            "not every latent endpoint was generated"
        )

    metadata = {
        "source_supported_cells": int(source_support.sum()),
        "valid_source_cells_at_endpoints": valid_counts,
        "flow_input_shape": list(flows.shape),
        "flow_latent_shape": list(latent_flows.shape),
        "flow_vector_scale_xy": [
            width / flow_w,
            height / flow_h,
        ],
        "flow_direction_assumption": (
            "consecutive previous-frame to next-frame"
        ),
        "advection": (
            "sample each next flow at the current "
            "advected XY position"
        ),
        "splatting": (
            "nearest-cell forward splat with collision mean"
        ),
    }

    return (
        torch.stack(transported),
        torch.stack(masks),
        torch.stack(counts),
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-artifact",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--flows",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )

    args = parser.parse_args()

    base_path = args.base_artifact.resolve()
    flows_path = args.flows.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()

    for path in (base_path, flows_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite: {path}"
            )

    device_name = args.device

    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    device = torch.device(device_name)

    base = torch.load(
        base_path,
        map_location="cpu",
        weights_only=False,
    )

    required = (
        "source_latent",
        "target_latent",
        "latent_frame_indices",
        "transport_mask",
        "contribution_count",
    )

    missing = [
        key
        for key in required
        if key not in base
    ]

    if missing:
        raise KeyError(
            f"base artifact is missing: {missing}"
        )

    source_latent = torch.as_tensor(
        base["source_latent"]
    )

    target_latent = torch.as_tensor(
        base["target_latent"]
    )

    endpoints = torch.as_tensor(
        base["latent_frame_indices"],
        dtype=torch.long,
    )

    correct_mask = torch.as_tensor(
        base["transport_mask"]
    )

    correct_count = torch.as_tensor(
        base["contribution_count"]
    )

    if (
        source_latent.ndim != 5
        or tuple(source_latent.shape[:2]) != (1, 1)
    ):
        raise ValueError(
            "unexpected source_latent shape: "
            f"{tuple(source_latent.shape)}"
        )

    if (
        target_latent.ndim != 5
        or target_latent.shape[0] != 1
    ):
        raise ValueError(
            "unexpected target_latent shape: "
            f"{tuple(target_latent.shape)}"
        )

    if target_latent.shape[1] != endpoints.numel():
        raise ValueError(
            "target slots do not match "
            "latent_frame_indices"
        )

    if int(endpoints[0]) != 0:
        raise ValueError(
            "latent_frame_indices must start at 0"
        )

    if not bool(
        torch.all(
            endpoints[1:] > endpoints[:-1]
        )
    ):
        raise ValueError(
            "latent_frame_indices must be "
            "strictly increasing"
        )

    expected_mask_shape = (
        target_latent.shape[1],
        1,
        target_latent.shape[3],
        target_latent.shape[4],
    )

    if (
        tuple(correct_mask.shape)
        != expected_mask_shape
        or correct_mask.dtype != torch.bool
    ):
        raise ValueError(
            "invalid transport_mask"
        )

    if (
        tuple(correct_count.shape)
        != expected_mask_shape
    ):
        raise ValueError(
            "invalid contribution_count"
        )

    if not torch.equal(
        correct_mask,
        correct_count > 0,
    ):
        raise ValueError(
            "transport_mask must equal "
            "contribution_count > 0"
        )

    if not bool(
        torch.isfinite(source_latent).all()
        and torch.isfinite(target_latent).all()
    ):
        raise ValueError(
            "source or target latent "
            "contains NaN/Inf"
        )

    flows_np = np.load(
        flows_path,
        allow_pickle=False,
    )

    if (
        flows_np.ndim != 4
        or flows_np.shape[1] != 2
    ):
        raise ValueError(
            f"unexpected flows shape: {flows_np.shape}"
        )

    if int(endpoints[-1]) > flows_np.shape[0]:
        raise ValueError(
            "not enough consecutive flows "
            "for the final endpoint"
        )

    if not np.isfinite(flows_np).all():
        raise ValueError(
            "flows contain NaN/Inf"
        )

    source_grid = source_latent[
        0,
        0,
    ].to(
        device=device,
        dtype=torch.float32,
    )

    target = target_latent.to(
        device=device,
        dtype=torch.float32,
    )

    source_support = correct_mask[
        0
    ].to(
        device=device
    )

    flows = torch.from_numpy(
        flows_np
    ).to(
        device=device,
        dtype=torch.float32,
    )

    endpoints_device = endpoints.to(
        device=device
    )

    raw, flow_mask, flow_count, metadata = (
        build_transport(
            source_grid,
            source_support,
            flows,
            endpoints_device,
        )
    )

    raw = raw.unsqueeze(0)

    fused = torch.where(
        flow_mask.unsqueeze(0),
        raw,
        target,
    )

    if tuple(raw.shape) != tuple(target.shape):
        raise RuntimeError(
            "flow latent shape does not "
            "match target latent"
        )

    if not bool(
        torch.isfinite(raw).all()
        and torch.isfinite(fused).all()
    ):
        raise RuntimeError(
            "flow transport produced NaN/Inf"
        )

    if not torch.equal(
        flow_mask,
        flow_count > 0,
    ):
        raise RuntimeError(
            "flow mask/count contract failed"
        )

    correct_mask_device = correct_mask.to(
        device
    )

    overlap = (
        flow_mask
        & correct_mask_device
    )

    union = (
        flow_mask
        | correct_mask_device
    )

    support_iou = []

    for slot in range(flow_mask.shape[0]):
        union_count = int(
            union[slot].sum()
        )

        if union_count:
            value = float(
                overlap[slot].sum()
                / union[slot].sum()
            )
        else:
            value = 1.0

        support_iou.append(value)

    output_state = {
        "format_version": 1,
        "mode": (
            "raft_dense_flow_latent_transport"
        ),
        "latent_frame_indices": endpoints.cpu(),
        "flow_transported_latent": (
            raw.cpu().to(source_latent.dtype)
        ),
        "flow_fused_latent": (
            fused.cpu().to(target_latent.dtype)
        ),
        "transport_mask": flow_mask.cpu(),
        "contribution_count": flow_count.cpu(),
        "source_support_mask": (
            source_support.cpu()
        ),
        "metadata": {
            **metadata,
            "base_artifact": str(base_path),
            "base_artifact_sha256": (
                sha256(base_path)
            ),
            "flows": str(flows_path),
            "flows_sha256": (
                sha256(flows_path)
            ),
            "endpoint_indices": (
                endpoints.tolist()
            ),
            "correct_vs_flow_support_iou_per_slot": (
                support_iou
            ),
            "composition": (
                "flow latent inside flow mask; "
                "target coarse latent outside"
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output_state,
        output_path,
    )

    repo_root = Path.cwd()

    if str(repo_root) not in sys.path:
        sys.path.insert(
            0,
            str(repo_root),
        )

    from deform_transport.pipeline_integration import (
        load_precomputed_transport_latent,
    )

    loaded = load_precomputed_transport_latent(
        output_path,
        mode="flow",
        reference_latent=target_latent,
    )

    expected = output_state[
        "flow_fused_latent"
    ].to(
        loaded.dtype
    )

    if not torch.equal(
        loaded.cpu(),
        expected.cpu(),
    ):
        raise RuntimeError(
            "production loader returned "
            "a different flow latent"
        )

    report = {
        "status": "FLOW_ARTIFACT_READY",
        "output": str(output_path),
        "output_bytes": (
            output_path.stat().st_size
        ),
        "output_sha256": (
            sha256(output_path)
        ),
        "report": str(report_path),
        "device": str(device),
        "source_latent_shape": (
            list(source_latent.shape)
        ),
        "target_latent_shape": (
            list(target_latent.shape)
        ),
        "flow_input_shape": (
            list(flows_np.shape)
        ),
        "flow_transported_shape": (
            list(raw.shape)
        ),
        "flow_fused_shape": (
            list(fused.shape)
        ),
        "flow_finite": bool(
            torch.isfinite(fused).all()
        ),
        "flow_mask_cells_per_slot": [
            int(value)
            for value in flow_mask.sum(
                (1, 2, 3)
            ).tolist()
        ],
        "correct_mask_cells_per_slot": [
            int(value)
            for value in correct_mask.sum(
                (1, 2, 3)
            ).tolist()
        ],
        "correct_vs_flow_support_iou_per_slot": (
            support_iou
        ),
        "loader_validation": "passed",
        "metadata": metadata,
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


if __name__ == "__main__":
    main()