"""Read-only audit for aligned RealWonder conditioning and Wan VAE contracts.

This script does not create an aligned final_sim, encode a VAE, or use CUDA.
It inventories:
- old final_sim frames/config/noises;
- source simulation RGB/flow/raster assets;
- old base Wan VAE artifact;
- aligned transport and visibility timelines;
- latent slot-0 alignment risks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--old-final-sim",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--old-base-vae",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--aligned-transport",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--aligned-visibility",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def tensor_info(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {
            "type": type(value).__name__,
            "repr": repr(value)[:1000],
        }

    result: dict[str, Any] = {
        "type": "Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }

    if value.numel() == 0:
        return result

    if value.dtype == torch.bool:
        result.update(
            {
                "true": int(value.sum()),
                "false": int(
                    value.numel() - value.sum()
                ),
            }
        )
        return result

    numeric = value.detach().to(
        torch.float32
    )

    finite = torch.isfinite(numeric)

    result["all_finite"] = bool(
        finite.all()
    )

    if bool(finite.any()):
        selected = numeric[finite]

        result.update(
            {
                "min": float(selected.min()),
                "max": float(selected.max()),
                "mean": float(selected.mean()),
                "std": (
                    float(selected.std())
                    if selected.numel() > 1
                    else 0.0
                ),
            }
        )

    if value.numel() <= 100:
        result["values"] = (
            value.detach().cpu().tolist()
        )

    return result


def numpy_info(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
        }

    value = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    result: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "sha256": sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": int(path.stat().st_size),
    }

    if value.size:
        result.update(
            {
                "minimum": float(
                    np.min(value)
                ),
                "maximum": float(
                    np.max(value)
                ),
                "mean": float(
                    np.mean(value)
                ),
                "std": float(
                    np.std(value)
                ),
                "all_finite": bool(
                    np.isfinite(value).all()
                ),
            }
        )

    return result


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "bytes": (
            int(path.stat().st_size)
            if path.is_file()
            else None
        ),
        "sha256": (
            sha256(path)
            if path.is_file()
            else None
        ),
    }


def main() -> None:
    args = parse_args()

    old_final_sim = (
        args.old_final_sim.resolve()
    )
    source_dir = (
        args.source_dir.resolve()
    )
    old_base_path = (
        args.old_base_vae.resolve()
    )
    aligned_transport_path = (
        args.aligned_transport.resolve()
    )
    aligned_visibility_path = (
        args.aligned_visibility.resolve()
    )
    output_report = (
        args.output_report.resolve()
    )

    required = [
        old_final_sim / "config.yaml",
        old_final_sim / "noises.npy",
        old_final_sim
        / "resized_input_image.png",
        source_dir / "flows.npy",
        source_dir
        / "flow_source_point_indices.npy",
        source_dir / "frame_initial.png",
        source_dir / "frame_0000.png",
        source_dir / "frame_0079.png",
        source_dir / "frame_0080.png",
        old_base_path,
        aligned_transport_path,
        aligned_visibility_path,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_report.exists():
        raise FileExistsError(
            f"refusing to overwrite: "
            f"{output_report}"
        )

    old_base = torch.load(
        old_base_path,
        map_location="cpu",
        weights_only=True,
    )

    aligned = torch.load(
        aligned_transport_path,
        map_location="cpu",
        weights_only=False,
    )

    visibility = torch.load(
        aligned_visibility_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(old_base, dict):
        raise ValueError(
            "old base VAE artifact is not "
            "a dictionary"
        )

    if not isinstance(aligned, dict):
        raise ValueError(
            "aligned transport artifact is "
            "not a dictionary"
        )

    if not isinstance(visibility, dict):
        raise ValueError(
            "visibility artifact is not "
            "a dictionary"
        )

    old_frame_paths = sorted(
        path
        for path in (
            old_final_sim / "frames"
        ).iterdir()
        if path.is_file()
    )

    source_frame_paths = sorted(
        source_dir.glob(
            "frame_[0-9][0-9][0-9][0-9].png"
        )
    )

    config_text = (
        old_final_sim
        / "config.yaml"
    ).read_text(
        encoding="utf-8",
        errors="replace",
    )

    source_latent = old_base.get(
        "source_latent"
    )
    target_latent = old_base.get(
        "target_latent"
    )
    latent_indices = old_base.get(
        "latent_frame_indices"
    )

    slot0_difference = None

    if (
        isinstance(source_latent, torch.Tensor)
        and isinstance(target_latent, torch.Tensor)
        and source_latent.ndim == 5
        and target_latent.ndim == 5
        and source_latent.shape[1] == 1
        and target_latent.shape[1] >= 1
        and tuple(source_latent.shape[2:])
        == tuple(target_latent.shape[2:])
    ):
        difference = torch.abs(
            source_latent.to(torch.float32)
            - target_latent[:, 0:1].to(
                torch.float32
            )
        )

        slot0_difference = {
            "mean_abs": float(
                difference.mean()
            ),
            "max_abs": float(
                difference.max()
            ),
            "exactly_equal": bool(
                torch.equal(
                    source_latent,
                    target_latent[:, 0:1],
                )
            ),
        }

    aligned_coarse_paths = aligned.get(
        "paths",
        {},
    ).get(
        "coarse_rgb_frames",
        [],
    )

    report = {
        "inputs": {
            "old_final_sim": str(
                old_final_sim
            ),
            "source_dir": str(
                source_dir
            ),
            "old_base_vae": {
                "path": str(
                    old_base_path
                ),
                "sha256": sha256(
                    old_base_path
                ),
            },
            "aligned_transport": {
                "path": str(
                    aligned_transport_path
                ),
                "sha256": sha256(
                    aligned_transport_path
                ),
            },
            "aligned_visibility": {
                "path": str(
                    aligned_visibility_path
                ),
                "sha256": sha256(
                    aligned_visibility_path
                ),
            },
        },
        "old_final_sim": {
            "frame_count": len(
                old_frame_paths
            ),
            "first_frame": (
                str(old_frame_paths[0])
                if old_frame_paths
                else None
            ),
            "last_frame": (
                str(old_frame_paths[-1])
                if old_frame_paths
                else None
            ),
            "config": file_info(
                old_final_sim
                / "config.yaml"
            ),
            "config_text": config_text,
            "input_image": file_info(
                old_final_sim
                / "resized_input_image.png"
            ),
            "noises": numpy_info(
                old_final_sim
                / "noises.npy"
            ),
        },
        "source_simulation": {
            "numbered_frame_count": len(
                source_frame_paths
            ),
            "frame_initial": file_info(
                source_dir
                / "frame_initial.png"
            ),
            "frame_0000": file_info(
                source_dir
                / "frame_0000.png"
            ),
            "frame_0079": file_info(
                source_dir
                / "frame_0079.png"
            ),
            "frame_0080": file_info(
                source_dir
                / "frame_0080.png"
            ),
            "flows": numpy_info(
                source_dir / "flows.npy"
            ),
            "raster": numpy_info(
                source_dir
                / "flow_source_point_indices.npy"
            ),
        },
        "old_base_vae": {
            "keys": sorted(
                old_base.keys()
            ),
            "latent_frame_indices":
                tensor_info(
                    latent_indices
                ),
            "source_latent":
                tensor_info(
                    source_latent
                ),
            "target_latent":
                tensor_info(
                    target_latent
                ),
            "transport_mask":
                tensor_info(
                    old_base.get(
                        "transport_mask"
                    )
                ),
            "contribution_count":
                tensor_info(
                    old_base.get(
                        "contribution_count"
                    )
                ),
            "source_files":
                old_base.get(
                    "source_files"
                ),
            "source_vs_target_slot0":
                slot0_difference,
        },
        "aligned_contract": {
            "frame_ids": tensor_info(
                aligned.get(
                    "frame_ids"
                )
            ),
            "simulation_steps":
                tensor_info(
                    aligned.get(
                        "simulation_steps"
                    )
                ),
            "coarse_rgb_count": len(
                aligned_coarse_paths
            ),
            "coarse_rgb_first": (
                aligned_coarse_paths[0]
                if aligned_coarse_paths
                else None
            ),
            "coarse_rgb_last": (
                aligned_coarse_paths[-1]
                if aligned_coarse_paths
                else None
            ),
            "all_coarse_rgb_exist":
                all(
                    Path(path).is_file()
                    for path
                    in aligned_coarse_paths
                ),
            "visibility_shape":
                tensor_info(
                    visibility.get(
                        "aligned_visible"
                    )
                ),
            "latent_anchor_steps": (
                aligned[
                    "simulation_steps"
                ][
                    torch.arange(
                        0,
                        81,
                        4,
                        dtype=torch.long,
                    )
                ].tolist()
            ),
        },
        "candidate_alignment": {
            "rgb_frames": (
                "frame_initial + old "
                "frame_0000..frame_0079"
            ),
            "point_states": (
                "source state + old states "
                "10..800"
            ),
            "raster_states": (
                "saved raster states 0..800"
            ),
            "flow_candidate_needing_code_confirmation":
                (
                    "either zero flow for aligned frame 0 "
                    "plus old flows[0:80], or another "
                    "mapping dictated by RealWonder's "
                    "flow-conditioning loader"
                ),
            "noise_mapping":
                (
                    "must be derived from code; shape "
                    "inventory alone is insufficient"
                ),
            "old_base_vae_is_not_safe_for_aligned_transport":
                True,
        },
        "interpretation_boundary": (
            "Read-only audit. No aligned final_sim "
            "or aligned VAE artifact was created."
        ),
    }

    output_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_report.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "report": str(
                    output_report
                ),
                "old_final_sim":
                    report[
                        "old_final_sim"
                    ],
                "source_simulation":
                    report[
                        "source_simulation"
                    ],
                "old_base_vae":
                    report[
                        "old_base_vae"
                    ],
                "aligned_contract":
                    report[
                        "aligned_contract"
                    ],
                "candidate_alignment":
                    report[
                        "candidate_alignment"
                    ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
