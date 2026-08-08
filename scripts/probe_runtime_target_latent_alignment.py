"""Compare artifact target latent with the fresh RealWonder runtime VAE encoding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from infer_sim import load_sim_frames
from vidgen.models import WanVAEWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sim-data-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def tensor_stats(tensor: torch.Tensor) -> dict:
    value = tensor.detach().to(torch.float32)

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
        "finite": bool(torch.isfinite(value).all()),
    }


def main() -> None:
    args = parse_args()

    sim_data_path = args.sim_data_path.resolve()
    artifact_path = args.artifact.resolve()
    output_report = args.output_report.resolve()

    if not sim_data_path.is_dir():
        raise FileNotFoundError(sim_data_path)

    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    if output_report.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_report}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    state = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "target_latent",
        "transport_mask",
        "contribution_count",
    }

    missing = sorted(required - set(state))

    if missing:
        raise ValueError(
            f"artifact is missing keys: {missing}"
        )

    target_latent = state["target_latent"]
    transport_mask = state["transport_mask"]
    contribution_count = state[
        "contribution_count"
    ]

    if target_latent.ndim != 5:
        raise ValueError(
            "target_latent must have shape [B,T,C,H,W]"
        )

    if not torch.equal(
        transport_mask,
        contribution_count > 0,
    ):
        raise ValueError(
            "artifact mask/count contract failed"
        )

    device = torch.device("cuda:0")

    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()

    frames_dir = sim_data_path / "frames"

    sim_frames = load_sim_frames(
        frames_dir,
        height=480,
        width=832,
    )

    started = time.perf_counter()

    vae = WanVAEWrapper()
    vae = vae.to(dtype=torch.bfloat16)
    vae = vae.to(device=device)
    vae.eval()

    load_seconds = time.perf_counter() - started

    encode_started = time.perf_counter()

    with torch.no_grad():
        fresh_latent = vae.encode_to_latent(
            sim_frames.to(
                device=device,
                dtype=torch.bfloat16,
            )
        )

    torch.cuda.synchronize()

    encode_seconds = (
        time.perf_counter() - encode_started
    )

    fresh_latent = fresh_latent.to(
        device=device,
        dtype=torch.bfloat16,
    )

    artifact_target = target_latent.to(
        device=device,
        dtype=torch.bfloat16,
    )

    if tuple(fresh_latent.shape) != tuple(
        artifact_target.shape
    ):
        raise RuntimeError(
            "fresh runtime latent shape differs from "
            f"artifact target: {tuple(fresh_latent.shape)} "
            f"vs {tuple(artifact_target.shape)}"
        )

    difference = (
        fresh_latent.to(torch.float32)
        - artifact_target.to(torch.float32)
    ).abs()

    mask_5d = (
        transport_mask
        .unsqueeze(0)
        .expand_as(fresh_latent)
        .to(device=device)
    )

    masked_difference = difference[mask_5d]

    report = {
        "sim_data_path": str(sim_data_path),
        "artifact": str(artifact_path),
        "fresh_runtime_latent": tensor_stats(
            fresh_latent
        ),
        "artifact_target_latent_after_bfloat16_cast":
            tensor_stats(artifact_target),
        "alignment": {
            "shape_equal": True,
            "exact_after_bfloat16_cast": bool(
                torch.equal(
                    fresh_latent,
                    artifact_target,
                )
            ),
            "allclose_atol_1e-4": bool(
                torch.allclose(
                    fresh_latent,
                    artifact_target,
                    atol=1e-4,
                    rtol=0.0,
                )
            ),
            "allclose_atol_1e-3": bool(
                torch.allclose(
                    fresh_latent,
                    artifact_target,
                    atol=1e-3,
                    rtol=0.0,
                )
            ),
            "allclose_atol_1e-2": bool(
                torch.allclose(
                    fresh_latent,
                    artifact_target,
                    atol=1e-2,
                    rtol=0.0,
                )
            ),
            "full_max_abs_difference": float(
                difference.max()
            ),
            "full_mean_abs_difference": float(
                difference.mean()
            ),
            "full_nonzero_values": int(
                torch.count_nonzero(difference)
            ),
            "transport_mask_values": int(
                masked_difference.numel()
            ),
            "transport_mask_max_abs_difference": (
                float(masked_difference.max())
                if masked_difference.numel()
                else 0.0
            ),
            "transport_mask_mean_abs_difference": (
                float(masked_difference.mean())
                if masked_difference.numel()
                else 0.0
            ),
        },
        "artifact_contract": {
            "mask_cells": int(
                transport_mask.sum()
            ),
            "mask_count_contract": True,
        },
        "runtime_seconds": {
            "vae_load": load_seconds,
            "vae_encode": encode_seconds,
            "total": time.perf_counter() - started,
        },
        "gpu_peak_memory_mib": {
            "allocated": float(
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            ),
            "reserved": float(
                torch.cuda.max_memory_reserved()
                / (1024 ** 2)
            ),
        },
    }

    output_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))

    if hasattr(vae, "model"):
        vae.model.clear_cache()


if __name__ == "__main__":
    main()
