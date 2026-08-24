"""Build a temporally aligned Santa transport and raster-visibility contract.

New aligned pixel timeline:

    frame 0  = source state / simulation step 0
    frame 1  = old future frame 0 / simulation step 10
    ...
    frame 80 = old future frame 79 / simulation step 800

The old future frame 80 / simulation step 810 is not used.

This script is CPU-only. It does not modify the source artifact, RealWonder
generation code, transport implementation, or existing generated videos.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.transport_ready import validate_transport_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport-ready",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--raster-npy",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
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


def prepend_source_state(
    source: torch.Tensor,
    future: torch.Tensor,
    *,
    frame_count: int,
    name: str,
) -> torch.Tensor:
    if future.shape[0] != frame_count:
        raise ValueError(
            f"{name}: expected {frame_count} future frames, "
            f"got {future.shape[0]}"
        )

    if tuple(source.shape) != tuple(future.shape[1:]):
        raise ValueError(
            f"{name}: source/future shape mismatch: "
            f"{tuple(source.shape)} vs {tuple(future.shape)}"
        )

    return torch.cat(
        [
            source.unsqueeze(0),
            future[: frame_count - 1],
        ],
        dim=0,
    ).contiguous()


def visibility_from_raster(
    raster_frame: np.ndarray,
    *,
    point_count: int,
    valid: torch.Tensor,
) -> torch.Tensor:
    values = np.asarray(
        raster_frame
    ).reshape(-1)

    selected = values[
        values >= 0
    ]

    visible = torch.zeros(
        point_count,
        dtype=torch.bool,
    )

    if selected.size:
        ids = np.unique(
            selected.astype(
                np.int64,
                copy=False,
            )
        )

        if ids[0] < 0 or ids[-1] >= point_count:
            raise ValueError(
                "raster contains an out-of-range point ID"
            )

        visible[
            torch.from_numpy(ids)
        ] = True

    return visible & valid.to(
        device="cpu",
        dtype=torch.bool,
    )


def tensor_equal(
    first: torch.Tensor,
    second: torch.Tensor,
) -> bool:
    return bool(
        torch.equal(
            first.detach().cpu(),
            second.detach().cpu(),
        )
    )


def max_camera_difference(
    value: torch.Tensor,
) -> float:
    reference = value[0:1]

    return float(
        (
            value.to(torch.float64)
            - reference.to(torch.float64)
        )
        .abs()
        .max()
    )


def main() -> None:
    args = parse_args()

    old_path = (
        args.transport_ready.resolve()
    )
    source_dir = (
        args.source_dir.resolve()
    )
    raster_path = (
        args.raster_npy.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    aligned_path = (
        output_dir
        / "aligned_transport_ready.pt"
    )
    visibility_path = (
        output_dir
        / "aligned_visibility_contract.pt"
    )
    report_path = (
        output_dir
        / "report.json"
    )

    for path in (
        old_path,
        raster_path,
        source_dir / "frame_initial.png",
        source_dir / "frame_0000.png",
        source_dir / "frame_0079.png",
        source_dir / "frame_0080.png",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists() and any(
        output_dir.iterdir()
    ):
        raise FileExistsError(
            f"refusing to overwrite non-empty directory: "
            f"{output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    old = torch.load(
        old_path,
        map_location="cpu",
        weights_only=False,
    )

    validate_transport_ready(old)

    raster = np.load(
        raster_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    frame_count = int(
        old["frame_ids"].numel()
    )
    point_count = int(
        old["point_id"].numel()
    )

    if frame_count != 81:
        raise ValueError(
            f"expected 81 frames, got {frame_count}"
        )

    if tuple(raster.shape) != (
        frame_count,
        512,
        512,
    ):
        raise ValueError(
            f"unexpected raster shape: {raster.shape}"
        )

    if not np.issubdtype(
        raster.dtype,
        np.integer,
    ):
        raise TypeError(
            f"raster must be integer, got {raster.dtype}"
        )

    expected_old_steps = torch.arange(
        10,
        811,
        10,
        dtype=torch.long,
    )

    if not torch.equal(
        old["simulation_steps"].cpu(),
        expected_old_steps,
    ):
        raise ValueError(
            "old simulation steps are not [10,20,...,810]"
        )

    aligned = copy.deepcopy(old)

    aligned["case_name"] = (
        str(old["case_name"])
        + "_aligned_step0_to_800"
    )

    aligned["frame_ids"] = torch.arange(
        frame_count,
        dtype=torch.long,
    )

    aligned["simulation_steps"] = (
        torch.arange(
            0,
            801,
            10,
            dtype=torch.long,
        )
    )

    temporal_fields = {
        "points_3d":
            "source_points_3d",
        "points_2d_render":
            "source_points_2d_render",
        "points_2d_video":
            "source_points_2d_video",
        "points_2d_latent":
            "source_points_2d_latent",
        "depth":
            "source_depth",
        "render_projection_valid":
            "source_render_projection_valid",
        "projection_valid":
            "source_valid",
    }

    if (
        "points_2d_latent_continuous" in old
        and
        "source_points_2d_latent_continuous" in old
    ):
        temporal_fields[
            "points_2d_latent_continuous"
        ] = (
            "source_points_2d_latent_continuous"
        )

    for future_key, source_key in (
        temporal_fields.items()
    ):
        aligned[future_key] = (
            prepend_source_state(
                old[source_key],
                old[future_key],
                frame_count=frame_count,
                name=future_key,
            )
        )

    camera_differences: dict[
        str,
        float,
    ] = {}

    aligned_camera: dict[
        str,
        torch.Tensor,
    ] = {}

    for key in ("K", "R", "T"):
        value = old["camera"][key]

        if value.shape[0] != frame_count:
            raise ValueError(
                f"camera {key} frame mismatch: "
                f"{tuple(value.shape)}"
            )

        difference = (
            max_camera_difference(value)
        )

        camera_differences[key] = (
            difference
        )

        if difference > 1e-6:
            raise RuntimeError(
                "camera is not static, but the initial "
                "camera was not saved as a separate frame: "
                f"{key} max difference={difference}"
            )

        aligned_camera[key] = torch.cat(
            [
                value[0:1],
                value[: frame_count - 1],
            ],
            dim=0,
        ).contiguous()

    aligned["camera"] = aligned_camera

    paths = copy.deepcopy(
        old.get("paths", {})
    )

    old_coarse_paths = list(
        paths.get(
            "coarse_rgb_frames",
            [],
        )
    )

    initial_rgb_path = (
        source_dir / "frame_initial.png"
    )

    if len(old_coarse_paths) != frame_count:
        raise ValueError(
            "old coarse RGB path count does not match "
            f"frame count: {len(old_coarse_paths)}"
        )

    aligned_coarse_paths = [
        str(initial_rgb_path.resolve())
    ] + [
        str(Path(path).resolve())
        for path in old_coarse_paths[
            : frame_count - 1
        ]
    ]

    paths[
        "coarse_rgb_frames"
    ] = aligned_coarse_paths

    paths[
        "initial_rgb"
    ] = str(
        initial_rgb_path.resolve()
    )

    aligned["paths"] = paths

    aligned[
        "alignment_extension_version"
    ] = 1

    aligned["alignment_mapping"] = {
        "pixel_frame_count": frame_count,
        "old_contract": (
            "input=step0; future/coarse=steps10..810"
        ),
        "aligned_contract": (
            "frames=steps0,10,...,800"
        ),
        "frame_0": (
            "source state and frame_initial.png"
        ),
        "frames_1_to_80": (
            "old future/coarse frames 0..79"
        ),
        "discarded_old_future_frame": 80,
        "discarded_simulation_step": 810,
        "old_future_index_by_aligned_frame":
            [-1] + list(range(80)),
        "raster_index_by_aligned_frame":
            list(range(81)),
        "aligned_flow_slice": (
            "old flows.npy[0:80]"
        ),
        "source_transport_ready": str(
            old_path
        ),
        "source_transport_ready_sha256":
            sha256(old_path),
        "source_raster": str(
            raster_path
        ),
        "source_raster_sha256":
            sha256(raster_path),
    }

    # Existing validator checks the standard transport-ready contract.
    validate_transport_ready(aligned)

    aligned_visible_frames: list[
        torch.Tensor
    ] = []

    for frame_index in range(
        frame_count
    ):
        aligned_visible_frames.append(
            visibility_from_raster(
                raster[frame_index],
                point_count=point_count,
                valid=aligned[
                    "projection_valid"
                ][frame_index],
            )
        )

    aligned_visible = torch.stack(
        aligned_visible_frames,
        dim=0,
    )

    source_and_aligned_visible = (
        aligned_visible
        & aligned[
            "source_visible"
        ].unsqueeze(0)
    )

    visible_counts = (
        aligned_visible.sum(dim=1)
    )

    source_and_visible_counts = (
        source_and_aligned_visible.sum(
            dim=1
        )
    )

    visibility_contract = {
        "format_version": 1,
        "artifact_kind":
            "aligned_raster_visibility_contract",
        "case_name":
            aligned["case_name"],
        "frame_ids":
            aligned["frame_ids"].clone(),
        "simulation_steps":
            aligned[
                "simulation_steps"
            ].clone(),
        "point_id":
            aligned["point_id"].clone(),
        "source_visible":
            aligned[
                "source_visible"
            ].clone(),
        "aligned_projection_valid":
            aligned[
                "projection_valid"
            ].clone(),
        "aligned_visible":
            aligned_visible,
        "source_and_aligned_visible":
            source_and_aligned_visible,
        "visible_point_counts":
            visible_counts,
        "source_and_visible_point_counts":
            source_and_visible_counts,
        "raster_index_by_frame":
            torch.arange(
                frame_count,
                dtype=torch.long,
            ),
        "old_future_index_by_frame":
            torch.tensor(
                [-1] + list(range(80)),
                dtype=torch.long,
            ),
        "visibility_definition": {
            "raster":
                "unique non-negative frontmost "
                "point IDs from saved "
                "flow_source_point_indices.npy",
            "validity":
                "intersection with the aligned "
                "projection/crop validity mask",
            "not_projection_only": True,
            "not_depth_threshold": True,
        },
        "source_files": {
            "transport_ready": str(
                old_path
            ),
            "transport_ready_sha256":
                sha256(old_path),
            "raster_npy": str(
                raster_path
            ),
            "raster_npy_sha256":
                sha256(raster_path),
        },
    }

    checks: dict[str, bool] = {}

    checks[
        "aligned_frame_ids_are_0_to_80"
    ] = tensor_equal(
        aligned["frame_ids"],
        torch.arange(
            81,
            dtype=torch.long,
        ),
    )

    checks[
        "aligned_steps_are_0_to_800"
    ] = tensor_equal(
        aligned["simulation_steps"],
        torch.arange(
            0,
            801,
            10,
            dtype=torch.long,
        ),
    )

    for future_key, source_key in (
        temporal_fields.items()
    ):
        checks[
            f"{future_key}_frame0_equals_source"
        ] = tensor_equal(
            aligned[future_key][0],
            old[source_key],
        )

        checks[
            f"{future_key}_frames1_80_equal_old0_79"
        ] = tensor_equal(
            aligned[future_key][1:],
            old[future_key][:80],
        )

    checks[
        "visibility_frame0_equals_source_visible"
    ] = tensor_equal(
        aligned_visible[0],
        aligned["source_visible"],
    )

    checks[
        "visibility_is_subset_of_projection_valid"
    ] = not bool(
        (
            aligned_visible
            & ~aligned[
                "projection_valid"
            ]
        ).any()
    )

    checks[
        "source_and_visibility_is_subset_of_source_visible"
    ] = not bool(
        (
            source_and_aligned_visible
            & ~aligned[
                "source_visible"
            ].unsqueeze(0)
        ).any()
    )

    checks[
        "aligned_coarse_path_count_is_81"
    ] = (
        len(aligned_coarse_paths)
        == 81
    )

    checks[
        "aligned_coarse_frame0_is_initial"
    ] = (
        Path(
            aligned_coarse_paths[0]
        ).resolve()
        == initial_rgb_path.resolve()
    )

    checks[
        "all_aligned_coarse_paths_exist"
    ] = all(
        Path(path).is_file()
        for path in aligned_coarse_paths
    )

    checks[
        "camera_is_static"
    ] = all(
        value <= 1e-6
        for value in (
            camera_differences.values()
        )
    )

    if (
        "points_2d_latent_continuous"
        in aligned
    ):
        checks[
            "continuous_floor_recovers_discrete"
        ] = tensor_equal(
            torch.floor(
                aligned[
                    "points_2d_latent_continuous"
                ]
            ).to(torch.long),
            aligned[
                "points_2d_latent"
            ],
        )

        checks[
            "source_continuous_floor_recovers_discrete"
        ] = tensor_equal(
            torch.floor(
                aligned[
                    "source_points_2d_latent_continuous"
                ]
            ).to(torch.long),
            aligned[
                "source_points_2d_latent"
            ],
        )

    all_checks_pass = all(
        checks.values()
    )

    report: dict[str, Any] = {
        "inputs": {
            "old_transport_ready": {
                "path": str(old_path),
                "sha256": sha256(
                    old_path
                ),
            },
            "source_dir": str(
                source_dir
            ),
            "raster_npy": {
                "path": str(
                    raster_path
                ),
                "sha256": sha256(
                    raster_path
                ),
                "shape": list(
                    raster.shape
                ),
                "dtype": str(
                    raster.dtype
                ),
            },
        },
        "aligned_contract": {
            "case_name":
                aligned["case_name"],
            "frame_count":
                frame_count,
            "point_count":
                point_count,
            "simulation_steps":
                aligned[
                    "simulation_steps"
                ].tolist(),
            "coarse_frame_0":
                aligned_coarse_paths[0],
            "coarse_frame_80":
                aligned_coarse_paths[-1],
            "discarded_old_state":
                "old frame 80 / simulation step 810",
            "camera_static_max_abs":
                camera_differences,
        },
        "visibility": {
            "shape": list(
                aligned_visible.shape
            ),
            "minimum": int(
                visible_counts.min()
            ),
            "mean": float(
                visible_counts
                .to(torch.float32)
                .mean()
            ),
            "maximum": int(
                visible_counts.max()
            ),
            "first": int(
                visible_counts[0]
            ),
            "middle": int(
                visible_counts[
                    frame_count // 2
                ]
            ),
            "final": int(
                visible_counts[-1]
            ),
            "source_and_visible_first":
                int(
                    source_and_visible_counts[
                        0
                    ]
                ),
            "source_and_visible_final":
                int(
                    source_and_visible_counts[
                        -1
                    ]
                ),
        },
        "checks": checks,
        "all_checks_pass":
            all_checks_pass,
    }

    if not all_checks_pass:
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            "aligned contract validation failed"
        )

    torch.save(
        aligned,
        aligned_path,
    )

    torch.save(
        visibility_contract,
        visibility_path,
    )

    report["outputs"] = {
        "aligned_transport_ready": {
            "path": str(
                aligned_path
            ),
            "sha256": sha256(
                aligned_path
            ),
            "bytes": int(
                aligned_path.stat().st_size
            ),
        },
        "aligned_visibility_contract": {
            "path": str(
                visibility_path
            ),
            "sha256": sha256(
                visibility_path
            ),
            "bytes": int(
                visibility_path.stat().st_size
            ),
        },
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
            {
                "report": str(
                    report_path
                ),
                "outputs":
                    report["outputs"],
                "aligned_contract":
                    report[
                        "aligned_contract"
                    ],
                "visibility":
                    report["visibility"],
                "checks":
                    checks,
                "all_checks_pass":
                    all_checks_pass,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
