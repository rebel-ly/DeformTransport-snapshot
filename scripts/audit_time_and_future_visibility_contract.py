"""Read-only audit of temporal alignment and future visibility contracts.

This script:
1. compares the input image, coarse simulation frame 0, and generated frame 0;
2. measures source-to-target-frame-0 material-point displacement;
3. inventories depth, 3D point, camera, and temporal fields;
4. tests whether source visibility can be reproduced by a calibrated
   per-point z-buffer rule;
5. applies the best rule to future frames only as a diagnostic candidate.

It does not modify transport artifacts, core code, or generated videos.
It does not create a formal future-visibility contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport-ready",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--confidence-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--final-sim",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def inventory(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    if depth > 4:
        return {
            "type": type(value).__name__,
            "truncated": True,
        }

    if isinstance(value, torch.Tensor):
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
                    "true_values": int(value.sum()),
                    "false_values": int(
                        value.numel() - value.sum()
                    ),
                }
            )
            return result

        if value.dtype.is_floating_point:
            tensor = value.detach().to(
                torch.float32
            )

            finite = torch.isfinite(tensor)

            result["finite_values"] = int(
                finite.sum()
            )
            result["all_finite"] = bool(
                finite.all()
            )

            if bool(finite.any()):
                selected = tensor[finite]

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
                        "zero_values": int(
                            torch.count_nonzero(
                                selected == 0
                            )
                        ),
                        "negative_values": int(
                            torch.count_nonzero(
                                selected < 0
                            )
                        ),
                    }
                )
        else:
            result.update(
                {
                    "min": int(value.min()),
                    "max": int(value.max()),
                    "zero_values": int(
                        torch.count_nonzero(
                            value == 0
                        )
                    ),
                }
            )

        if value.numel() <= 32:
            result["values"] = (
                value.detach().cpu().tolist()
            )

        return result

    if isinstance(value, dict):
        return {
            str(key): inventory(
                item,
                depth=depth + 1,
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [
                inventory(
                    item,
                    depth=depth + 1,
                )
                for item in value[:20]
            ],
            "truncated": len(value) > 20,
        }

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    return {
        "type": type(value).__name__,
        "repr": repr(value)[:500],
    }


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB")
    )


def load_video_first_frame(
    path: Path,
) -> np.ndarray:
    reader = imageio.get_reader(path)

    try:
        return np.asarray(
            reader.get_data(0)
        )[..., :3]
    finally:
        reader.close()


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

    signed = first_float - second_float
    absolute = np.abs(signed)

    mse = float(
        np.mean(signed ** 2)
    )

    psnr: float | str

    if mse == 0:
        psnr = "inf"
    else:
        psnr = float(
            10.0
            * math.log10(
                (255.0 ** 2) / mse
            )
        )

    return {
        "shape_equal": True,
        "exactly_equal": bool(
            np.array_equal(first, second)
        ),
        "mae_0_255": float(
            absolute.mean()
        ),
        "mse_0_255": mse,
        "psnr_db": psnr,
        "max_abs_difference": float(
            absolute.max()
        ),
        "different_values": int(
            np.count_nonzero(absolute)
        ),
    }


def save_triptych(
    images: list[tuple[str, np.ndarray]],
    output_path: Path,
) -> None:
    height, width = images[0][1].shape[:2]
    title_height = 34

    canvas = Image.new(
        "RGB",
        (
            width * len(images),
            height + title_height,
        ),
        "black",
    )

    draw = ImageDraw.Draw(canvas)

    for column, (title, array) in enumerate(
        images
    ):
        image = Image.fromarray(array)

        canvas.paste(
            image,
            (column * width, title_height),
        )

        draw.text(
            (column * width + 8, 9),
            title,
            fill="white",
        )

    canvas.save(output_path)


def displacement_statistics(
    source: torch.Tensor,
    target: torch.Tensor,
    subset: torch.Tensor | None = None,
) -> dict[str, Any]:
    source_float = source.to(
        torch.float32
    )
    target_float = target.to(
        torch.float32
    )

    displacement = torch.linalg.vector_norm(
        target_float - source_float,
        dim=-1,
    )

    if subset is not None:
        displacement = displacement[
            subset.to(torch.bool)
        ]

    if displacement.numel() == 0:
        return {
            "values": 0,
        }

    probabilities = torch.tensor(
        [
            0.00,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        ],
        dtype=torch.float32,
    )

    quantile = torch.quantile(
        displacement,
        probabilities,
    )

    return {
        "values": int(displacement.numel()),
        "mean": float(displacement.mean()),
        "std": float(displacement.std()),
        "q00": float(quantile[0]),
        "q25": float(quantile[1]),
        "q50": float(quantile[2]),
        "q75": float(quantile[3]),
        "q90": float(quantile[4]),
        "q95": float(quantile[5]),
        "q99": float(quantile[6]),
        "q100": float(quantile[7]),
        "exactly_zero": int(
            torch.count_nonzero(
                displacement == 0
            )
        ),
    }


def reshape_source_per_point_depth(
    value: Any,
    point_count: int,
) -> np.ndarray | None:
    if not isinstance(value, torch.Tensor):
        return None

    if value.numel() != point_count:
        return None

    return (
        value.detach()
        .to(torch.float64)
        .cpu()
        .reshape(point_count)
        .numpy()
    )


def reshape_future_per_point_depth(
    value: Any,
    frame_count: int,
    point_count: int,
) -> np.ndarray | None:
    if not isinstance(value, torch.Tensor):
        return None

    if value.numel() != (
        frame_count * point_count
    ):
        return None

    return (
        value.detach()
        .to(torch.float64)
        .cpu()
        .reshape(
            frame_count,
            point_count,
        )
        .numpy()
    )


def quantize(
    values: np.ndarray,
    policy: str,
) -> np.ndarray:
    if policy == "floor":
        return np.floor(values).astype(
            np.int64
        )

    if policy == "round_half_up":
        return np.floor(
            values + 0.5
        ).astype(np.int64)

    if policy == "ceil":
        return np.ceil(values).astype(
            np.int64
        )

    raise ValueError(policy)


def compute_depth_gap(
    coordinates: np.ndarray,
    depths: np.ndarray,
    *,
    width: int,
    height: int,
    quantization: str,
    depth_order: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    pixel = quantize(
        coordinates,
        quantization,
    )

    x = pixel[:, 0]
    y = pixel[:, 1]

    valid = (
        np.isfinite(coordinates).all(axis=1)
        & np.isfinite(depths)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    gap = np.full(
        depths.shape[0],
        np.inf,
        dtype=np.float64,
    )

    valid_indices = np.flatnonzero(valid)

    if valid_indices.size == 0:
        return gap, valid

    pixel_id = (
        y[valid_indices] * width
        + x[valid_indices]
    )

    valid_depth = depths[
        valid_indices
    ]

    if depth_order == "minimum":
        front = np.full(
            width * height,
            np.inf,
            dtype=np.float64,
        )

        np.minimum.at(
            front,
            pixel_id,
            valid_depth,
        )

        local_gap = (
            valid_depth
            - front[pixel_id]
        )
    elif depth_order == "maximum":
        front = np.full(
            width * height,
            -np.inf,
            dtype=np.float64,
        )

        np.maximum.at(
            front,
            pixel_id,
            valid_depth,
        )

        local_gap = (
            front[pixel_id]
            - valid_depth
        )
    else:
        raise ValueError(depth_order)

    local_gap = np.maximum(
        local_gap,
        0.0,
    )

    gap[valid_indices] = local_gap

    return gap, valid


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    tp = int(
        np.logical_and(
            prediction,
            target,
        ).sum()
    )
    fp = int(
        np.logical_and(
            prediction,
            ~target,
        ).sum()
    )
    fn = int(
        np.logical_and(
            ~prediction,
            target,
        ).sum()
    )
    tn = int(
        np.logical_and(
            ~prediction,
            ~target,
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn
        else 0.0
    )

    return {
        "target_visible": int(target.sum()),
        "predicted_visible": int(
            prediction.sum()
        ),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": (
            (tp + tn) / target.size
        ),
        "exactly_equal": bool(
            np.array_equal(
                prediction,
                target,
            )
        ),
    }


def candidate_thresholds(
    gaps: np.ndarray,
    valid: np.ndarray,
) -> list[float]:
    values = gaps[
        valid & np.isfinite(gaps)
    ]

    if values.size == 0:
        return [0.0]

    probabilities = np.asarray(
        [
            0.000,
            0.001,
            0.005,
            0.010,
            0.025,
            0.050,
            0.100,
            0.250,
            0.500,
            0.750,
            0.900,
            0.950,
            0.990,
        ],
        dtype=np.float64,
    )

    quantiles = np.quantile(
        values,
        probabilities,
    )

    thresholds = sorted(
        {
            float(value)
            for value in quantiles
            if np.isfinite(value)
        }
    )

    return thresholds or [0.0]


def correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    valid = (
        np.isfinite(first)
        & np.isfinite(second)
    )

    first_selected = first[valid]
    second_selected = second[valid]

    if first_selected.size < 3:
        return None

    if (
        np.std(first_selected) == 0
        or np.std(second_selected) == 0
    ):
        return None

    return float(
        np.corrcoef(
            first_selected,
            second_selected,
        )[0, 1]
    )


def main() -> None:
    args = parse_args()

    transport_path = (
        args.transport_ready.resolve()
    )
    artifact_path = (
        args.confidence_artifact.resolve()
    )
    final_sim = args.final_sim.resolve()
    baseline_path = (
        args.baseline_video.resolve()
    )
    output_dir = args.output_dir.resolve()

    for path in (
        transport_path,
        artifact_path,
        baseline_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not final_sim.is_dir():
        raise FileNotFoundError(final_sim)

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    transport = torch.load(
        transport_path,
        map_location="cpu",
        weights_only=True,
    )

    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )

    source_image_path = (
        final_sim
        / "resized_input_image.png"
    )
    simulation_frame0_path = (
        final_sim
        / "frames"
        / "frame_0000.png"
    )

    source_image = load_rgb(
        source_image_path
    )
    simulation_frame0 = load_rgb(
        simulation_frame0_path
    )
    generated_frame0 = (
        load_video_first_frame(
            baseline_path
        )
    )

    if not (
        source_image.shape
        == simulation_frame0.shape
        == generated_frame0.shape
    ):
        raise ValueError(
            "input, simulation frame 0, and "
            "generated frame 0 have different shapes"
        )

    triptych_path = (
        output_dir
        / "time_alignment_triptych.png"
    )

    save_triptych(
        [
            (
                "Input resized image",
                source_image,
            ),
            (
                "Simulation frame 0",
                simulation_frame0,
            ),
            (
                "Baseline generated frame 0",
                generated_frame0,
            ),
        ],
        triptych_path,
    )

    source_points_video = transport[
        "source_points_2d_video"
    ]
    future_points_video = transport[
        "points_2d_video"
    ]
    source_points_latent = transport[
        "source_points_2d_latent_continuous"
    ]
    future_points_latent = transport[
        "points_2d_latent_continuous"
    ]
    source_visible_tensor = transport[
        "source_visible"
    ]

    point_count = int(
        source_points_video.shape[0]
    )
    frame_count = int(
        future_points_video.shape[0]
    )

    if tuple(
        future_points_video.shape
    ) != (
        frame_count,
        point_count,
        2,
    ):
        raise ValueError(
            "future video coordinate shape mismatch"
        )

    source_visible = (
        source_visible_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )

    latent_indices = artifact[
        "latent_frame_indices"
    ].to(torch.long)

    source_depth = (
        reshape_source_per_point_depth(
            transport.get("source_depth"),
            point_count,
        )
    )

    future_depth = (
        reshape_future_per_point_depth(
            transport.get("depth"),
            frame_count,
            point_count,
        )
    )

    render_width = int(
        transport["render_width"]
    )
    render_height = int(
        transport["render_height"]
    )
    video_width = int(
        transport["video_width"]
    )
    video_height = int(
        transport["video_height"]
    )

    coordinate_spaces: dict[
        str,
        tuple[np.ndarray, np.ndarray, int, int]
    ] = {
        "render": (
            transport[
                "source_points_2d_render"
            ].to(torch.float32).cpu().numpy(),
            transport[
                "points_2d_render"
            ].to(torch.float32).cpu().numpy(),
            render_width,
            render_height,
        ),
        "video": (
            source_points_video
            .to(torch.float32)
            .cpu()
            .numpy(),
            future_points_video
            .to(torch.float32)
            .cpu()
            .numpy(),
            video_width,
            video_height,
        ),
    }

    zbuffer_candidates: list[
        dict[str, Any]
    ] = []

    if source_depth is not None:
        for (
            coordinate_name,
            (
                source_coordinates,
                _,
                width,
                height,
            ),
        ) in coordinate_spaces.items():
            for quantization in (
                "floor",
                "round_half_up",
                "ceil",
            ):
                for depth_order in (
                    "minimum",
                    "maximum",
                ):
                    gaps, valid = (
                        compute_depth_gap(
                            source_coordinates,
                            source_depth,
                            width=width,
                            height=height,
                            quantization=quantization,
                            depth_order=depth_order,
                        )
                    )

                    for threshold in (
                        candidate_thresholds(
                            gaps,
                            valid,
                        )
                    ):
                        prediction = (
                            valid
                            & (gaps <= threshold)
                        )

                        metrics = binary_metrics(
                            prediction,
                            source_visible,
                        )

                        zbuffer_candidates.append(
                            {
                                "coordinate_space":
                                    coordinate_name,
                                "width": width,
                                "height": height,
                                "quantization":
                                    quantization,
                                "depth_order":
                                    depth_order,
                                "threshold":
                                    float(threshold),
                                "valid_projected_points":
                                    int(valid.sum()),
                                "metrics": metrics,
                            }
                        )

    best_candidate = (
        max(
            zbuffer_candidates,
            key=lambda item: (
                item["metrics"]["f1"],
                item["metrics"]["iou"],
                item["metrics"]["precision"],
                -item["threshold"],
            ),
        )
        if zbuffer_candidates
        else None
    )

    future_visibility_summary: dict[
        str,
        Any,
    ] = {
        "available": False,
    }

    if (
        best_candidate is not None
        and future_depth is not None
    ):
        source_coordinates, future_coordinates, width, height = (
            coordinate_spaces[
                best_candidate[
                    "coordinate_space"
                ]
            ]
        )

        del source_coordinates

        projection_valid = transport.get(
            "projection_valid"
        )

        projection_valid_numpy = None

        if (
            isinstance(
                projection_valid,
                torch.Tensor,
            )
            and tuple(
                projection_valid.shape
            )
            == (
                frame_count,
                point_count,
            )
        ):
            projection_valid_numpy = (
                projection_valid
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

        rows = []

        for frame_index in range(
            frame_count
        ):
            gaps, valid = compute_depth_gap(
                future_coordinates[
                    frame_index
                ],
                future_depth[
                    frame_index
                ],
                width=width,
                height=height,
                quantization=(
                    best_candidate[
                        "quantization"
                    ]
                ),
                depth_order=(
                    best_candidate[
                        "depth_order"
                    ]
                ),
            )

            if projection_valid_numpy is not None:
                valid = (
                    valid
                    & projection_valid_numpy[
                        frame_index
                    ]
                )

            prediction = (
                valid
                & (
                    gaps
                    <= best_candidate[
                        "threshold"
                    ]
                )
            )

            rows.append(
                {
                    "frame_index": frame_index,
                    "future_visible": int(
                        prediction.sum()
                    ),
                    "source_and_future_visible":
                        int(
                            np.logical_and(
                                source_visible,
                                prediction,
                            ).sum()
                        ),
                    "source_visible_but_occluded":
                        int(
                            np.logical_and(
                                source_visible,
                                valid & ~prediction,
                            ).sum()
                        ),
                    "source_visible_out_of_bounds":
                        int(
                            np.logical_and(
                                source_visible,
                                ~valid,
                            ).sum()
                        ),
                    "valid_projected": int(
                        valid.sum()
                    ),
                }
            )

        future_visibility_summary = {
            "available": True,
            "source_rule_quality": {
                "f1": best_candidate[
                    "metrics"
                ]["f1"],
                "iou": best_candidate[
                    "metrics"
                ]["iou"],
                "precision": best_candidate[
                    "metrics"
                ]["precision"],
                "recall": best_candidate[
                    "metrics"
                ]["recall"],
            },
            "confidence_tier": (
                "high"
                if (
                    best_candidate[
                        "metrics"
                    ]["f1"] >= 0.98
                    and best_candidate[
                        "metrics"
                    ]["iou"] >= 0.96
                )
                else "moderate"
                if (
                    best_candidate[
                        "metrics"
                    ]["f1"] >= 0.90
                    and best_candidate[
                        "metrics"
                    ]["iou"] >= 0.82
                )
                else "low"
            ),
            "rule": {
                key: best_candidate[key]
                for key in (
                    "coordinate_space",
                    "width",
                    "height",
                    "quantization",
                    "depth_order",
                    "threshold",
                )
            },
            "frames": rows,
        }

        table_path = (
            output_dir
            / "future_visibility_diagnostic.csv"
        )

        with table_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            columns = list(rows[0].keys())

            handle.write(
                ",".join(columns) + "\n"
            )

            for row in rows:
                handle.write(
                    ",".join(
                        str(row[column])
                        for column in columns
                    )
                    + "\n"
                )

    source_points_3d = transport.get(
        "source_points_3d"
    )

    depth_axis_correlations: dict[
        str,
        float | None,
    ] = {}

    if (
        source_depth is not None
        and isinstance(
            source_points_3d,
            torch.Tensor,
        )
        and tuple(
            source_points_3d.shape
        )
        == (
            point_count,
            3,
        )
    ):
        xyz = (
            source_points_3d
            .to(torch.float64)
            .cpu()
            .numpy()
        )

        depth_axis_correlations = {
            "depth_vs_x": correlation(
                source_depth,
                xyz[:, 0],
            ),
            "depth_vs_y": correlation(
                source_depth,
                xyz[:, 1],
            ),
            "depth_vs_z": correlation(
                source_depth,
                xyz[:, 2],
            ),
            "depth_vs_negative_z":
                correlation(
                    source_depth,
                    -xyz[:, 2],
                ),
            "depth_vs_euclidean_norm":
                correlation(
                    source_depth,
                    np.linalg.norm(
                        xyz,
                        axis=1,
                    ),
                ),
        }

    report = {
        "inputs": {
            "transport_ready": str(
                transport_path
            ),
            "confidence_artifact": str(
                artifact_path
            ),
            "final_sim": str(final_sim),
            "baseline_video": str(
                baseline_path
            ),
        },
        "time_alignment": {
            "input_image": str(
                source_image_path
            ),
            "simulation_frame_0": str(
                simulation_frame0_path
            ),
            "generated_frame_0": str(
                baseline_path
            ),
            "image_shape": list(
                source_image.shape
            ),
            "pairwise": {
                "input_vs_simulation_frame_0":
                    pair_metrics(
                        source_image,
                        simulation_frame0,
                    ),
                "input_vs_generated_frame_0":
                    pair_metrics(
                        source_image,
                        generated_frame0,
                    ),
                "simulation_frame_0_vs_generated_frame_0":
                    pair_metrics(
                        simulation_frame0,
                        generated_frame0,
                    ),
            },
            "triptych": str(
                triptych_path
            ),
            "frame_ids": inventory(
                transport.get("frame_ids")
            ),
            "simulation_steps": inventory(
                transport.get(
                    "simulation_steps"
                )
            ),
        },
        "source_to_target_frame_0": {
            "video_pixel_displacement_all":
                displacement_statistics(
                    source_points_video,
                    future_points_video[0],
                ),
            "video_pixel_displacement_source_visible":
                displacement_statistics(
                    source_points_video,
                    future_points_video[0],
                    source_visible_tensor,
                ),
            "latent_displacement_all":
                displacement_statistics(
                    source_points_latent,
                    future_points_latent[0],
                ),
            "latent_displacement_source_visible":
                displacement_statistics(
                    source_points_latent,
                    future_points_latent[0],
                    source_visible_tensor,
                ),
        },
        "artifact_slot_0": {
            "latent_frame_indices":
                inventory(latent_indices),
            "latent_frame_index_0":
                int(latent_indices[0]),
            "source_latent": inventory(
                artifact.get(
                    "source_latent"
                )
            ),
            "target_latent": inventory(
                artifact.get(
                    "target_latent"
                )
            ),
            "correct_transport_residual":
                inventory(
                    artifact.get(
                        "correct_transport_residual"
                    )
                ),
            "source_files": inventory(
                artifact.get(
                    "source_files"
                )
            ),
            "coordinate_candidate_for_slot_0":
                (
                    "points_2d_latent_continuous"
                    f"[{int(latent_indices[0])}]"
                ),
            "interpretation_requires_code_audit":
                True,
        },
        "depth_and_geometry": {
            "source_depth": inventory(
                transport.get(
                    "source_depth"
                )
            ),
            "future_depth": inventory(
                transport.get("depth")
            ),
            "source_depth_is_per_point":
                source_depth is not None,
            "future_depth_is_per_point":
                future_depth is not None,
            "source_points_3d": inventory(
                transport.get(
                    "source_points_3d"
                )
            ),
            "future_points_3d": inventory(
                transport.get(
                    "points_3d"
                )
            ),
            "camera": inventory(
                transport.get("camera")
            ),
            "coordinate_system":
                inventory(
                    transport.get(
                        "coordinate_system"
                    )
                ),
            "depth_axis_correlations":
                depth_axis_correlations,
        },
        "source_visibility": {
            "source_visible": inventory(
                source_visible_tensor
            ),
            "source_visible_point_ids":
                inventory(
                    transport.get(
                        "source_visible_point_ids"
                    )
                ),
            "source_raster_visible_point_ids":
                inventory(
                    transport.get(
                        "source_raster_visible_point_ids"
                    )
                ),
            "projection_valid":
                inventory(
                    transport.get(
                        "projection_valid"
                    )
                ),
        },
        "zbuffer_contract_audit": {
            "candidate_count": len(
                zbuffer_candidates
            ),
            "best_candidate":
                best_candidate,
            "top_10_candidates": sorted(
                zbuffer_candidates,
                key=lambda item: (
                    item["metrics"]["f1"],
                    item["metrics"]["iou"],
                    item["metrics"]["precision"],
                ),
                reverse=True,
            )[:10],
            "future_visibility_diagnostic":
                future_visibility_summary,
        },
        "decision_boundary": {
            "time": (
                "Do not define metric frame mapping "
                "until image comparisons and code hits "
                "agree on generated frame-0 semantics."
            ),
            "visibility": (
                "A calibrated z-buffer rule is only a "
                "candidate future-visibility contract. "
                "Low or moderate source agreement is not "
                "sufficient for formal metric filtering."
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

    summary = {
        "report": str(report_path),
        "time_alignment":
            report["time_alignment"],
        "source_to_target_frame_0":
            report[
                "source_to_target_frame_0"
            ],
        "artifact_slot_0":
            report["artifact_slot_0"],
        "depth_and_geometry":
            report["depth_and_geometry"],
        "best_zbuffer_candidate":
            best_candidate,
        "future_visibility_diagnostic":
            future_visibility_summary,
    }

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
