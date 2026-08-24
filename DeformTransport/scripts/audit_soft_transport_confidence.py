"""Read-only audit of Soft transport confidence-related fields.

The script:
1. inventories all top-level artifact fields;
2. compares the original Soft artifact with the derived residual artifact;
3. identifies spatial tensors matching the transport-mask resolution;
4. audits weight/count statistics on interior, boundary and Soft-only cells;
5. saves grayscale map sheets for first, middle and final latent frames.

No source artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


MAP_KEYWORDS = (
    "weight",
    "confidence",
    "count",
    "mask",
    "valid",
    "visibility",
    "support",
    "density",
    "coverage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--soft-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--residual-artifact",
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


def safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def quantiles(
    values: torch.Tensor,
) -> dict[str, float | int]:
    values = values.detach().to(torch.float32).flatten()

    finite = values[torch.isfinite(values)]

    if finite.numel() == 0:
        return {
            "values": 0,
        }

    probabilities = torch.tensor(
        [
            0.00,
            0.01,
            0.05,
            0.10,
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

    q = torch.quantile(
        finite,
        probabilities,
    )

    return {
        "values": int(finite.numel()),
        "mean": safe_float(finite.mean()),
        "std": safe_float(finite.std())
        if finite.numel() > 1
        else 0.0,
        "q00": safe_float(q[0]),
        "q01": safe_float(q[1]),
        "q05": safe_float(q[2]),
        "q10": safe_float(q[3]),
        "q25": safe_float(q[4]),
        "q50": safe_float(q[5]),
        "q75": safe_float(q[6]),
        "q90": safe_float(q[7]),
        "q95": safe_float(q[8]),
        "q99": safe_float(q[9]),
        "q100": safe_float(q[10]),
    }


def tensor_inventory(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {
            "type": type(value).__name__,
            "repr": repr(value)[:500],
        }

    result: dict[str, Any] = {
        "type": "Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
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
        float_value = value.to(torch.float32)

        result.update(
            {
                "finite": bool(
                    torch.isfinite(float_value).all()
                ),
                "min": safe_float(float_value.min()),
                "max": safe_float(float_value.max()),
                "mean": safe_float(float_value.mean()),
                "std": safe_float(float_value.std()),
                "nonzero_values": int(
                    torch.count_nonzero(float_value)
                ),
            }
        )
    else:
        result.update(
            {
                "min": int(value.min()),
                "max": int(value.max()),
                "nonzero_values": int(
                    torch.count_nonzero(value)
                ),
            }
        )

    return result


def canonical_map(
    tensor: torch.Tensor,
    *,
    mask_shape: tuple[int, int, int, int],
) -> torch.Tensor | None:
    """Convert map-like tensors to [T,1,H,W] when unambiguous."""

    if tuple(tensor.shape) == mask_shape:
        return tensor

    t, _, h, w = mask_shape

    if tuple(tensor.shape) == (1, t, 1, h, w):
        return tensor.squeeze(0)

    if tuple(tensor.shape) == (t, h, w):
        return tensor.unsqueeze(1)

    if tuple(tensor.shape) == (1, t, h, w):
        return tensor.squeeze(0).unsqueeze(1)

    return None


def build_regions(
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask_bool = mask.to(torch.bool)

    neighbor_count = F.conv2d(
        mask_bool.to(torch.float32),
        torch.ones(
            1,
            1,
            3,
            3,
            dtype=torch.float32,
        ),
        padding=1,
    )

    interior = mask_bool & (neighbor_count == 9)
    boundary = mask_bool & ~interior

    return {
        "support": mask_bool,
        "interior": interior,
        "boundary": boundary,
    }


def region_stats(
    spatial_map: torch.Tensor,
    regions: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result = {}

    value = spatial_map.to(torch.float32)

    for name, region in regions.items():
        selected = value[region]

        result[name] = {
            "cells": int(region.sum()),
            "statistics": quantiles(selected),
        }

    return result


def normalized_uint8(
    value: torch.Tensor,
    *,
    logarithmic: bool,
) -> np.ndarray:
    tensor = value.detach().to(torch.float32).cpu()

    if logarithmic:
        tensor = torch.log1p(
            torch.clamp(tensor, min=0)
        )

    finite = tensor[torch.isfinite(tensor)]

    if finite.numel() == 0:
        return np.zeros(
            tensor.shape,
            dtype=np.uint8,
        )

    low = torch.quantile(finite, 0.01)
    high = torch.quantile(finite, 0.99)

    if float(high - low) <= 1e-12:
        scaled = torch.zeros_like(tensor)
    else:
        scaled = (
            (tensor - low)
            / (high - low)
        ).clamp(0, 1)

    return (
        scaled.numpy() * 255.0
    ).round().astype(np.uint8)


def save_map_sheet(
    *,
    output_path: Path,
    frame_index: int,
    maps: list[tuple[str, torch.Tensor, bool]],
) -> None:
    tile_scale = 4
    title_height = 28

    tile_width = maps[0][1].shape[-1] * tile_scale
    tile_height = maps[0][1].shape[-2] * tile_scale

    canvas = Image.new(
        "RGB",
        (
            tile_width * len(maps),
            tile_height + title_height,
        ),
        "black",
    )

    draw = ImageDraw.Draw(canvas)

    for column, (
        title,
        spatial_map,
        logarithmic,
    ) in enumerate(maps):
        frame = spatial_map[
            frame_index,
            0,
        ]

        array = normalized_uint8(
            frame,
            logarithmic=logarithmic,
        )

        image = Image.fromarray(
            array,
            mode="L",
        ).resize(
            (tile_width, tile_height),
            Image.Resampling.NEAREST,
        ).convert("RGB")

        x = column * tile_width

        canvas.paste(
            image,
            (x, title_height),
        )

        draw.text(
            (x + 6, 7),
            title,
            fill="white",
        )

    canvas.save(output_path)


def select_weight_key(
    maps: dict[str, torch.Tensor],
) -> str | None:
    exact_priority = (
        "transport_weight",
        "soft_transport_weight",
        "gated_transport_weight",
        "correct_transport_weight",
        "weight",
    )

    for key in exact_priority:
        if key in maps:
            return key

    candidates = sorted(
        key
        for key in maps
        if "weight" in key.lower()
    )

    return candidates[0] if candidates else None


def compare_common_tensor(
    first: dict[str, Any],
    second: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    a = first.get(key)
    b = second.get(key)

    if not isinstance(a, torch.Tensor):
        return {
            "available_in_soft": False,
            "available_in_residual": isinstance(
                b,
                torch.Tensor,
            ),
        }

    if not isinstance(b, torch.Tensor):
        return {
            "available_in_soft": True,
            "available_in_residual": False,
        }

    result = {
        "available_in_soft": True,
        "available_in_residual": True,
        "shape_equal": tuple(a.shape)
        == tuple(b.shape),
        "dtype_equal": a.dtype == b.dtype,
    }

    if tuple(a.shape) == tuple(b.shape):
        result["exactly_equal"] = bool(
            torch.equal(a, b)
        )

        if (
            a.dtype.is_floating_point
            and b.dtype.is_floating_point
        ):
            difference = (
                a.to(torch.float32)
                - b.to(torch.float32)
            ).abs()

            result.update(
                {
                    "max_abs_difference": safe_float(
                        difference.max()
                    ),
                    "mean_abs_difference": safe_float(
                        difference.mean()
                    ),
                }
            )

    return result


def main() -> None:
    args = parse_args()

    soft_path = args.soft_artifact.resolve()
    residual_path = args.residual_artifact.resolve()
    output_dir = args.output_dir.resolve()

    if not soft_path.is_file():
        raise FileNotFoundError(soft_path)

    if not residual_path.is_file():
        raise FileNotFoundError(residual_path)

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    soft = torch.load(
        soft_path,
        map_location="cpu",
        weights_only=True,
    )

    residual = torch.load(
        residual_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(soft, dict):
        raise ValueError(
            "Soft artifact is not a dictionary"
        )

    if not isinstance(residual, dict):
        raise ValueError(
            "Residual artifact is not a dictionary"
        )

    for state, label in (
        (soft, "Soft"),
        (residual, "Residual"),
    ):
        if "transport_mask" not in state:
            raise ValueError(
                f"{label} artifact has no transport_mask"
            )

        if "contribution_count" not in state:
            raise ValueError(
                f"{label} artifact has no contribution_count"
            )

    mask = soft["transport_mask"]

    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 4
        or mask.shape[1] != 1
    ):
        raise ValueError(
            "Soft transport_mask must have shape [T,1,H,W]"
        )

    mask_shape = tuple(mask.shape)

    if not torch.equal(
        mask,
        soft["contribution_count"] > 0,
    ):
        raise ValueError(
            "Soft mask/count contract failed"
        )

    spatial_maps: dict[str, torch.Tensor] = {}

    for key, value in soft.items():
        if not isinstance(value, torch.Tensor):
            continue

        canonical = canonical_map(
            value,
            mask_shape=mask_shape,
        )

        if canonical is None:
            continue

        if (
            any(
                word in key.lower()
                for word in MAP_KEYWORDS
            )
            or canonical.dtype == torch.bool
        ):
            spatial_maps[key] = canonical

    regions = build_regions(mask)

    soft_only = soft.get("soft_only_mask")

    if isinstance(soft_only, torch.Tensor):
        soft_only_map = canonical_map(
            soft_only,
            mask_shape=mask_shape,
        )

        if soft_only_map is not None:
            regions["soft_only"] = (
                soft_only_map.to(torch.bool)
                & mask
            )

    map_reports = {}

    for key, spatial_map in spatial_maps.items():
        map_reports[key] = {
            "inventory": tensor_inventory(
                spatial_map
            ),
            "region_statistics": region_stats(
                spatial_map,
                regions,
            ),
        }

    weight_key = select_weight_key(
        spatial_maps
    )

    derived: dict[str, Any] = {
        "selected_weight_key": weight_key,
    }

    visualization_maps: list[
        tuple[str, torch.Tensor, bool]
    ] = [
        (
            "Transport mask",
            mask.to(torch.float32),
            False,
        ),
        (
            "Contribution count",
            soft["contribution_count"].to(
                torch.float32
            ),
            True,
        ),
        (
            "Boundary",
            regions["boundary"].to(
                torch.float32
            ),
            False,
        ),
    ]

    if "soft_only" in regions:
        visualization_maps.append(
            (
                "Soft-only",
                regions["soft_only"].to(
                    torch.float32
                ),
                False,
            )
        )

    if weight_key is not None:
        weight = spatial_maps[
            weight_key
        ].to(torch.float32)

        count = soft[
            "contribution_count"
        ].to(torch.float32)

        weight_per_count = torch.where(
            count > 0,
            weight / count.clamp_min(1),
            torch.zeros_like(weight),
        )

        derived.update(
            {
                "weight_statistics":
                    region_stats(
                        weight,
                        regions,
                    ),
                "weight_per_count_statistics":
                    region_stats(
                        weight_per_count,
                        regions,
                    ),
                "weight_per_count_global":
                    quantiles(
                        weight_per_count[
                            mask
                        ]
                    ),
            }
        )

        visualization_maps.extend(
            [
                (
                    f"Weight: {weight_key}",
                    weight,
                    True,
                ),
                (
                    "Weight / count",
                    weight_per_count,
                    False,
                ),
            ]
        )

    frame_indices = {
        "first": 0,
        "middle": mask.shape[0] // 2,
        "final": mask.shape[0] - 1,
    }

    snapshots = {}

    for label, frame_index in frame_indices.items():
        path = (
            output_dir
            / f"confidence_maps_{label}.png"
        )

        save_map_sheet(
            output_path=path,
            frame_index=frame_index,
            maps=visualization_maps,
        )

        snapshots[label] = str(path)

    report = {
        "inputs": {
            "soft_artifact": {
                "path": str(soft_path),
                "sha256": sha256(soft_path),
            },
            "residual_artifact": {
                "path": str(residual_path),
                "sha256": sha256(
                    residual_path
                ),
            },
        },
        "soft_artifact_inventory": {
            key: tensor_inventory(value)
            for key, value in soft.items()
        },
        "residual_artifact_inventory": {
            key: tensor_inventory(value)
            for key, value in residual.items()
        },
        "common_tensor_comparisons": {
            key: compare_common_tensor(
                soft,
                residual,
                key,
            )
            for key in (
                "latent_frame_indices",
                "source_latent",
                "target_latent",
                "transport_mask",
                "contribution_count",
                "soft_only_mask",
                "transport_weight",
            )
        },
        "transport_geometry": {
            "mask_shape": list(mask.shape),
            "support_cells": int(
                regions["support"].sum()
            ),
            "interior_cells": int(
                regions["interior"].sum()
            ),
            "boundary_cells": int(
                regions["boundary"].sum()
            ),
            "soft_only_cells": int(
                regions.get(
                    "soft_only",
                    torch.zeros_like(mask),
                ).sum()
            ),
            "mask_count_contract": True,
        },
        "candidate_spatial_maps": map_reports,
        "derived_diagnostics": derived,
        "snapshots": snapshots,
        "interpretation_boundary": (
            "This is a read-only engineering audit. "
            "No confidence formula is selected solely "
            "from these statistics."
        ),
    }

    report_path = output_dir / "report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "report": str(report_path),
                "soft_keys": sorted(soft.keys()),
                "residual_keys": sorted(
                    residual.keys()
                ),
                "candidate_spatial_maps": sorted(
                    spatial_maps.keys()
                ),
                "selected_weight_key": weight_key,
                "transport_geometry":
                    report["transport_geometry"],
                "derived_diagnostics": derived,
                "snapshots": snapshots,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
