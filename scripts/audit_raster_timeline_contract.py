"""Audit saved RealWonder raster indices and temporal alignment.

CPU-only and read-only with respect to all source artifacts.

The saved raster stack is expected to contain the point-index raster that
served as the source of each optical-flow transition:

    raster[0]  -> initial state, simulation step 0
    raster[1]  -> old frame 0, simulation step 10
    ...
    raster[80] -> old frame 79, simulation step 800

No formal aligned artifact or visibility contract is written by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


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
        "--final-sim",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--raster-npy",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--flow-npy",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--crop-start",
        type=int,
        default=176,
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


def pair_metrics(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any]:
    if first.shape != second.shape:
        return {
            "shape_equal": False,
            "first_shape": list(first.shape),
            "second_shape": list(second.shape),
        }

    first_float = first.astype(np.float32)
    second_float = second.astype(np.float32)

    delta = first_float - second_float
    absolute = np.abs(delta)
    mse = float(np.mean(delta ** 2))

    return {
        "shape_equal": True,
        "exactly_equal": bool(
            np.array_equal(first, second)
        ),
        "mae_0_255": float(absolute.mean()),
        "mse_0_255": mse,
        "psnr_db": (
            "inf"
            if mse == 0
            else float(
                10.0
                * math.log10(
                    (255.0 ** 2) / mse
                )
            )
        ),
        "max_abs_difference": float(
            absolute.max()
        ),
        "different_values": int(
            np.count_nonzero(absolute)
        ),
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB")
        ).copy()


def resize_and_crop(
    path: Path,
    *,
    crop_start: int,
) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (832, 832),
            resample=Image.Resampling.BILINEAR,
        )
        image = image.crop(
            (
                0,
                crop_start,
                832,
                crop_start + 480,
            )
        )

        return np.asarray(image).copy()


def raster_ids(
    raster_frame: np.ndarray,
    *,
    point_count: int,
) -> np.ndarray:
    values = np.asarray(
        raster_frame
    ).reshape(-1)

    selected = values[
        values >= 0
    ]

    if selected.size == 0:
        return np.empty(
            (0,),
            dtype=np.int64,
        )

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

    return ids


def visibility_from_raster(
    raster_frame: np.ndarray,
    valid: torch.Tensor,
    *,
    point_count: int,
) -> torch.Tensor:
    ids = raster_ids(
        raster_frame,
        point_count=point_count,
    )

    visible = torch.zeros(
        point_count,
        dtype=torch.bool,
    )

    if ids.size:
        visible[
            torch.from_numpy(ids)
        ] = True

    return visible & valid.to(
        dtype=torch.bool,
        device="cpu",
    )


def main() -> None:
    args = parse_args()

    transport_path = (
        args.transport_ready.resolve()
    )
    source_dir = args.source_dir.resolve()
    final_sim = args.final_sim.resolve()
    raster_path = args.raster_npy.resolve()
    flow_path = args.flow_npy.resolve()
    output_dir = args.output_dir.resolve()

    required_files = [
        transport_path,
        raster_path,
        flow_path,
        source_dir / "frame_initial.png",
        source_dir / "frame_0000.png",
        source_dir / "frame_0079.png",
        source_dir / "frame_0080.png",
        final_sim / "resized_input_image.png",
        final_sim / "frames" / "frame_0000.png",
        final_sim / "frames" / "frame_0079.png",
        final_sim / "frames" / "frame_0080.png",
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    state = torch.load(
        transport_path,
        map_location="cpu",
        weights_only=False,
    )

    raster = np.load(
        raster_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    flows = np.load(
        flow_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    point_id = state["point_id"].to(
        torch.long
    )
    point_count = int(point_id.numel())

    if not torch.equal(
        point_id,
        torch.arange(
            point_count,
            dtype=torch.long,
        ),
    ):
        raise ValueError(
            "point IDs are not contiguous"
        )

    frame_ids = state["frame_ids"].to(
        torch.long
    )
    simulation_steps = state[
        "simulation_steps"
    ].to(torch.long)

    source_valid = state[
        "source_valid"
    ].to(torch.bool)

    source_visible = state[
        "source_visible"
    ].to(torch.bool)

    source_raster_ids_saved = state[
        "source_raster_visible_point_ids"
    ].to(torch.long)

    future_valid = state[
        "projection_valid"
    ].to(torch.bool)

    future_points = state[
        "points_3d"
    ]

    source_points = state[
        "source_points_3d"
    ]

    if raster.ndim != 3:
        raise ValueError(
            f"expected raster [T,H,W], got {raster.shape}"
        )

    if not np.issubdtype(
        raster.dtype,
        np.integer,
    ):
        raise TypeError(
            f"raster dtype must be integer, got {raster.dtype}"
        )

    raster_frame_count = int(
        raster.shape[0]
    )

    source_ids_numpy = raster_ids(
        raster[0],
        point_count=point_count,
    )

    source_ids_tensor = torch.from_numpy(
        source_ids_numpy
    ).to(torch.long)

    reconstructed_source_visible = (
        visibility_from_raster(
            raster[0],
            source_valid,
            point_count=point_count,
        )
    )

    aligned_frame_count = (
        min(
            raster_frame_count,
            int(future_points.shape[0]),
        )
    )

    # Candidate aligned temporal contract:
    # frame 0 is source state;
    # frames 1..80 are old future states 0..79.
    aligned_points = torch.cat(
        [
            source_points.unsqueeze(0),
            future_points[: aligned_frame_count - 1],
        ],
        dim=0,
    )

    aligned_valid = torch.cat(
        [
            source_valid.unsqueeze(0),
            future_valid[: aligned_frame_count - 1],
        ],
        dim=0,
    )

    aligned_steps = torch.cat(
        [
            torch.zeros(
                1,
                dtype=torch.long,
            ),
            simulation_steps[
                : aligned_frame_count - 1
            ],
        ],
        dim=0,
    )

    rows: list[dict[str, Any]] = []
    aligned_visible_masks: list[
        torch.Tensor
    ] = []

    for frame_index in range(
        aligned_frame_count
    ):
        visible = visibility_from_raster(
            raster[frame_index],
            aligned_valid[frame_index],
            point_count=point_count,
        )

        aligned_visible_masks.append(
            visible
        )

        source_and_future = int(
            (
                visible
                & source_visible
            ).sum()
        )

        rows.append(
            {
                "aligned_frame_index":
                    frame_index,
                "simulation_step": int(
                    aligned_steps[
                        frame_index
                    ]
                ),
                "raster_index":
                    frame_index,
                "old_future_index": (
                    -1
                    if frame_index == 0
                    else frame_index - 1
                ),
                "raster_unique_ids": int(
                    raster_ids(
                        raster[frame_index],
                        point_count=point_count,
                    ).size
                ),
                "visible_after_validity": int(
                    visible.sum()
                ),
                "source_and_current_visible":
                    source_and_future,
                "source_visible_but_not_current":
                    int(
                        source_visible.sum()
                    )
                    - source_and_future,
            }
        )

    aligned_visibility = torch.stack(
        aligned_visible_masks,
        dim=0,
    )

    counts = aligned_visibility.sum(
        dim=1
    ).to(torch.float32)

    # Verify the immutable assembly mapping.
    initial_processed = resize_and_crop(
        source_dir / "frame_initial.png",
        crop_start=args.crop_start,
    )

    raw_frame0_processed = resize_and_crop(
        source_dir / "frame_0000.png",
        crop_start=args.crop_start,
    )

    raw_frame79_processed = resize_and_crop(
        source_dir / "frame_0079.png",
        crop_start=args.crop_start,
    )

    raw_frame80_processed = resize_and_crop(
        source_dir / "frame_0080.png",
        crop_start=args.crop_start,
    )

    final_input = load_rgb(
        final_sim
        / "resized_input_image.png"
    )

    final_frame0 = load_rgb(
        final_sim
        / "frames"
        / "frame_0000.png"
    )

    final_frame79 = load_rgb(
        final_sim
        / "frames"
        / "frame_0079.png"
    )

    final_frame80 = load_rgb(
        final_sim
        / "frames"
        / "frame_0080.png"
    )

    initial_raw = load_rgb(
        source_dir / "frame_initial.png"
    )

    raw_frame0 = load_rgb(
        source_dir / "frame_0000.png"
    )

    checks = {
        "raster_shape_is_81x512x512":
            tuple(raster.shape)
            == (81, 512, 512),
        "raster_integer_dtype":
            bool(
                np.issubdtype(
                    raster.dtype,
                    np.integer,
                )
            ),
        "flow_frame_count_is_81":
            int(flows.shape[0]) == 81,
        "source_raster_ids_match_saved_ids":
            bool(
                torch.equal(
                    source_ids_tensor,
                    source_raster_ids_saved,
                )
            ),
        "source_visibility_exactly_reconstructed":
            bool(
                torch.equal(
                    reconstructed_source_visible,
                    source_visible,
                )
            ),
        "initial_preprocess_matches_final_input":
            bool(
                np.array_equal(
                    initial_processed,
                    final_input,
                )
            ),
        "raw_frame0_matches_final_frame0":
            bool(
                np.array_equal(
                    raw_frame0_processed,
                    final_frame0,
                )
            ),
        "raw_frame79_matches_final_frame79":
            bool(
                np.array_equal(
                    raw_frame79_processed,
                    final_frame79,
                )
            ),
        "raw_frame80_matches_final_frame80":
            bool(
                np.array_equal(
                    raw_frame80_processed,
                    final_frame80,
                )
            ),
        "aligned_contract_has_81_frames":
            aligned_frame_count == 81,
        "aligned_steps_are_0_to_800":
            bool(
                torch.equal(
                    aligned_steps,
                    torch.arange(
                        0,
                        801,
                        10,
                        dtype=torch.long,
                    ),
                )
            ),
        "aligned_points_shape":
            tuple(aligned_points.shape)
            == (
                81,
                point_count,
                3,
            ),
        "aligned_visibility_shape":
            tuple(
                aligned_visibility.shape
            )
            == (
                81,
                point_count,
            ),
        "current_step810_lacks_saved_current_raster":
            raster_frame_count
            == int(
                simulation_steps.numel()
            ),
    }

    all_checks_pass = all(
        checks.values()
    )

    csv_path = (
        output_dir
        / "aligned_visibility_counts.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "inputs": {
            "transport_ready": {
                "path": str(
                    transport_path
                ),
                "sha256": sha256(
                    transport_path
                ),
            },
            "raster_npy": {
                "path": str(raster_path),
                "sha256": sha256(
                    raster_path
                ),
            },
            "flow_npy": {
                "path": str(flow_path),
                "sha256": sha256(
                    flow_path
                ),
            },
            "source_dir": str(
                source_dir
            ),
            "final_sim": str(
                final_sim
            ),
        },
        "raster": {
            "shape": list(
                raster.shape
            ),
            "dtype": str(
                raster.dtype
            ),
            "minimum": int(
                np.min(raster)
            ),
            "maximum": int(
                np.max(raster)
            ),
            "background_value_is_negative":
                bool(
                    np.min(raster) < 0
                ),
            "source_unique_point_ids":
                int(
                    source_ids_numpy.size
                ),
        },
        "flow": {
            "shape": list(
                flows.shape
            ),
            "dtype": str(
                flows.dtype
            ),
        },
        "source_contract": {
            "saved_source_visible":
                int(
                    source_visible.sum()
                ),
            "saved_source_raster_ids":
                int(
                    source_raster_ids_saved.numel()
                ),
            "reconstructed_source_visible":
                int(
                    reconstructed_source_visible.sum()
                ),
            "raster_ids_exact_match":
                checks[
                    "source_raster_ids_match_saved_ids"
                ],
            "visibility_exact_match":
                checks[
                    "source_visibility_exactly_reconstructed"
                ],
        },
        "image_assembly_contract": {
            "initial_vs_old_frame0_raw":
                pair_metrics(
                    initial_raw,
                    raw_frame0,
                ),
            "processed_initial_vs_final_input":
                pair_metrics(
                    initial_processed,
                    final_input,
                ),
            "processed_raw_frame0_vs_final_frame0":
                pair_metrics(
                    raw_frame0_processed,
                    final_frame0,
                ),
            "processed_raw_frame79_vs_final_frame79":
                pair_metrics(
                    raw_frame79_processed,
                    final_frame79,
                ),
            "processed_raw_frame80_vs_final_frame80":
                pair_metrics(
                    raw_frame80_processed,
                    final_frame80,
                ),
        },
        "current_contract": {
            "input_state":
                "simulation step 0",
            "coarse_frames":
                "simulation steps 10..810",
            "future_point_states":
                "simulation steps 10..810",
            "available_previous_rasters":
                "simulation steps 0..800",
            "step_810_current_raster_saved":
                False,
        },
        "aligned_candidate_contract": {
            "frame_count":
                aligned_frame_count,
            "simulation_steps": (
                aligned_steps.tolist()
            ),
            "images": (
                "frame_initial + "
                "old frame_0000..frame_0079"
            ),
            "points": (
                "source_points + "
                "old future_points[0:80]"
            ),
            "raster": (
                "flow_source_point_indices[0:81]"
            ),
            "flow": (
                "old flows[0:80]"
            ),
            "discarded_old_state":
                "old frame_0080 / step 810",
            "visibility_count": {
                "minimum": int(
                    counts.min()
                ),
                "mean": float(
                    counts.mean()
                ),
                "maximum": int(
                    counts.max()
                ),
                "first": int(
                    counts[0]
                ),
                "middle": int(
                    counts[
                        aligned_frame_count
                        // 2
                    ]
                ),
                "final": int(
                    counts[-1]
                ),
            },
            "per_frame_csv": str(
                csv_path
            ),
        },
        "checks": checks,
        "all_checks_pass":
            all_checks_pass,
        "interpretation": {
            "raster_visibility": (
                "Frontmost point identity under "
                "the exact saved PyTorch3D raster "
                "contract, not projection validity "
                "and not a depth-threshold estimate."
            ),
            "scope": (
                "Read-only audit. No aligned "
                "transport artifact, final_sim, or "
                "future visibility artifact was "
                "written."
            ),
        },
    }

    report_path = (
        output_dir / "report.json"
    )

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
                "raster":
                    report["raster"],
                "flow":
                    report["flow"],
                "source_contract":
                    report[
                        "source_contract"
                    ],
                "image_assembly_contract":
                    report[
                        "image_assembly_contract"
                    ],
                "aligned_candidate_contract":
                    report[
                        "aligned_candidate_contract"
                    ],
                "checks":
                    report["checks"],
                "all_checks_pass":
                    all_checks_pass,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
