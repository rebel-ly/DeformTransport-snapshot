"""Joint latent-level scan of transport blend alpha and soft gate threshold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--alphas",
        default=(
            "0,0.125,0.25,0.375,0.5,"
            "0.625,0.75,0.875,1.0"
        ),
    )
    parser.add_argument(
        "--thresholds",
        default="0,0.1,0.25,0.5,0.75",
    )
    parser.add_argument(
        "--reference-alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--reference-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
    )

    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    result = []

    for item in value.split(","):
        number = float(item.strip())

        if not np.isfinite(number):
            raise ValueError(
                f"non-finite parameter: {item}"
            )

        result.append(number)

    if not result:
        raise ValueError(
            "parameter list cannot be empty"
        )

    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parameter_token(value: float) -> str:
    return (
        f"{value:.3f}"
        .replace("-", "m")
        .replace(".", "p")
    )


def candidate_id(
    alpha: float,
    threshold: float,
) -> str:
    return (
        f"a{parameter_token(alpha)}"
        f"_t{parameter_token(threshold)}"
    )


def temporal_statistics(
    latent: torch.Tensor,
) -> dict[str, float]:
    first = (
        latent[:, 1:]
        - latent[:, :-1]
    ).abs()

    second = (
        latent[:, 2:]
        - 2.0 * latent[:, 1:-1]
        + latent[:, :-2]
    ).abs()

    return {
        "first_order": (
            float(first.mean())
            if first.numel()
            else 0.0
        ),
        "second_order": (
            float(second.mean())
            if second.numel()
            else 0.0
        ),
    }


def masked_mean_abs(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    expanded = (
        mask.unsqueeze(0)
        .expand_as(first)
    )

    if not bool(expanded.any()):
        return 0.0

    return float(
        torch.abs(
            first - second
        )[expanded].mean()
    )


def evaluate_candidate(
    *,
    alpha: float,
    threshold: float,
    target: torch.Tensor,
    raw_correct: torch.Tensor,
    raw_shuffled: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_only_mask: torch.Tensor,
    transport_weight: torch.Tensor,
    target_temporal: dict[str, float],
) -> dict[str, Any]:
    gated_mask = (
        hard_mask
        | (
            soft_only_mask
            & (
                transport_weight
                >= threshold
            )
        )
    )

    mask_5d = gated_mask.unsqueeze(0)

    correct_inside = (
        target
        + alpha
        * (
            raw_correct
            - target
        )
    )

    shuffled_inside = (
        target
        + alpha
        * (
            raw_shuffled
            - target
        )
    )

    correct = torch.where(
        mask_5d,
        correct_inside,
        target,
    )

    shuffled = torch.where(
        mask_5d,
        shuffled_inside,
        target,
    )

    correct_temporal = (
        temporal_statistics(correct)
    )

    temporal_penalty = (
        abs(
            correct_temporal[
                "first_order"
            ]
            - target_temporal[
                "first_order"
            ]
        )
        / max(
            target_temporal[
                "first_order"
            ],
            1e-12,
        )
        + abs(
            correct_temporal[
                "second_order"
            ]
            - target_temporal[
                "second_order"
            ]
        )
        / max(
            target_temporal[
                "second_order"
            ],
            1e-12,
        )
    )

    return {
        "candidate_id": candidate_id(
            alpha,
            threshold,
        ),
        "alpha": float(alpha),
        "threshold": float(threshold),
        "support_cells": int(
            gated_mask.sum()
        ),
        "full_correct_target_l1": float(
            torch.abs(
                correct - target
            ).mean()
        ),
        "masked_correct_target_l1": (
            masked_mean_abs(
                correct,
                target,
                gated_mask,
            )
        ),
        "full_shuffled_target_l1": float(
            torch.abs(
                shuffled - target
            ).mean()
        ),
        "masked_shuffled_target_l1": (
            masked_mean_abs(
                shuffled,
                target,
                gated_mask,
            )
        ),
        "full_correct_shuffled_l1": float(
            torch.abs(
                correct - shuffled
            ).mean()
        ),
        "masked_correct_shuffled_l1": (
            masked_mean_abs(
                correct,
                shuffled,
                gated_mask,
            )
        ),
        "correct_first_order": (
            correct_temporal[
                "first_order"
            ]
        ),
        "correct_second_order": (
            correct_temporal[
                "second_order"
            ]
        ),
        "temporal_penalty": float(
            temporal_penalty
        ),
    }


def dominates(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    eps: float = 1e-12,
) -> bool:
    no_worse = (
        first[
            "full_correct_target_l1"
        ]
        <= second[
            "full_correct_target_l1"
        ]
        + eps
        and first[
            "temporal_penalty"
        ]
        <= second[
            "temporal_penalty"
        ]
        + eps
        and first[
            "identity_retention"
        ]
        >= second[
            "identity_retention"
        ]
        - eps
    )

    strictly_better = (
        first[
            "full_correct_target_l1"
        ]
        < second[
            "full_correct_target_l1"
        ]
        - eps
        or first[
            "temporal_penalty"
        ]
        < second[
            "temporal_penalty"
        ]
        - eps
        or first[
            "identity_retention"
        ]
        > second[
            "identity_retention"
        ]
        + eps
    )

    return no_worse and strictly_better


def pareto_frontier(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []

    for candidate in rows:
        if any(
            dominates(other, candidate)
            for other in rows
            if other is not candidate
        ):
            continue

        result.append(candidate)

    return sorted(
        result,
        key=lambda item: (
            item["identity_retention"],
            item["full_correct_target_l1"],
        ),
    )


def select_representatives(
    *,
    rows: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    reference: dict[str, Any],
    max_candidates: int,
) -> list[dict[str, Any]]:
    if max_candidates < 3:
        raise ValueError(
            "max_candidates must be at least 3"
        )

    nonzero = [
        row
        for row in rows
        if row["alpha"] > 0
    ]

    quality_pool = [
        row
        for row in nonzero
        if row[
            "identity_retention"
        ] >= 0.20
    ]

    quality_anchor = min(
        quality_pool,
        key=lambda item: (
            item[
                "full_correct_target_l1"
            ],
            item["temporal_penalty"],
        ),
    )

    control_anchor = max(
        nonzero,
        key=lambda item: (
            item[
                "identity_retention"
            ],
            -item[
                "full_correct_target_l1"
            ],
        ),
    )

    selected: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in (
        quality_anchor,
        reference,
        control_anchor,
    ):
        selected[
            item["candidate_id"]
        ] = item

    levels = np.linspace(
        0.25,
        1.0,
        max_candidates,
    )

    source = (
        frontier
        if frontier
        else nonzero
    )

    for level in levels:
        item = min(
            source,
            key=lambda row: (
                abs(
                    row[
                        "identity_retention"
                    ]
                    - float(level)
                )
                + 0.15
                * row[
                    "target_distortion_ratio"
                ]
                + 0.05
                * row[
                    "temporal_penalty_ratio"
                ]
            ),
        )

        selected[
            item["candidate_id"]
        ] = item

    ordered = sorted(
        selected.values(),
        key=lambda item: (
            item["identity_retention"],
            item["threshold"],
        ),
    )

    if len(ordered) <= max_candidates:
        return ordered

    mandatory_ids = {
        quality_anchor["candidate_id"],
        reference["candidate_id"],
        control_anchor["candidate_id"],
    }

    mandatory = [
        item
        for item in ordered
        if item["candidate_id"]
        in mandatory_ids
    ]

    optional = [
        item
        for item in ordered
        if item["candidate_id"]
        not in mandatory_ids
    ]

    remaining = (
        max_candidates
        - len(mandatory)
    )

    if remaining > 0 and optional:
        indices = np.linspace(
            0,
            len(optional) - 1,
            remaining,
        )

        sampled = [
            optional[
                int(round(index))
            ]
            for index in indices
        ]
    else:
        sampled = []

    deduplicated = {
        item["candidate_id"]: item
        for item in (
            mandatory + sampled
        )
    }

    return sorted(
        deduplicated.values(),
        key=lambda item: (
            item["identity_retention"],
            item["threshold"],
        ),
    )


def main() -> None:
    args = parse_args()

    artifact_path = (
        args.artifact.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    if not artifact_path.is_file():
        raise FileNotFoundError(
            artifact_path
        )

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                "refusing to use non-empty output directory: "
                f"{output_dir}"
            )
    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    candidate_dir = (
        output_dir / "candidates"
    )

    candidate_dir.mkdir()

    alphas = parse_float_list(
        args.alphas
    )

    thresholds = parse_float_list(
        args.thresholds
    )

    if any(
        alpha < 0 or alpha > 1
        for alpha in alphas
    ):
        raise ValueError(
            "alphas must be in [0,1]"
        )

    if any(
        threshold < 0
        for threshold in thresholds
    ):
        raise ValueError(
            "thresholds must be nonnegative"
        )

    state = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "target_latent",
        "correct_transported_latent",
        "shuffled_transported_latent",
        "hard_reference_mask",
        "soft_only_mask",
        "transport_weight",
        "latent_frame_indices",
    }

    missing = sorted(
        required - set(state)
    )

    if missing:
        raise ValueError(
            f"artifact missing keys: {missing}"
        )

    target = state[
        "target_latent"
    ].float().contiguous()

    raw_correct = state[
        "correct_transported_latent"
    ].float().contiguous()

    raw_shuffled = state[
        "shuffled_transported_latent"
    ].float().contiguous()

    hard_mask = state[
        "hard_reference_mask"
    ].bool().contiguous()

    soft_only_mask = state[
        "soft_only_mask"
    ].bool().contiguous()

    transport_weight = state[
        "transport_weight"
    ].float().contiguous()

    expected_latent_shape = (
        1,
        21,
        16,
        60,
        104,
    )

    if tuple(target.shape) != (
        expected_latent_shape
    ):
        raise ValueError(
            f"unexpected target shape: "
            f"{tuple(target.shape)}"
        )

    for name, value in (
        ("raw_correct", raw_correct),
        ("raw_shuffled", raw_shuffled),
    ):
        if value.shape != target.shape:
            raise ValueError(
                f"{name} shape mismatch"
            )

        if not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(
                f"{name} is non-finite"
            )

    expected_mask_shape = (
        21,
        1,
        60,
        104,
    )

    for name, value in (
        ("hard_mask", hard_mask),
        (
            "soft_only_mask",
            soft_only_mask,
        ),
        (
            "transport_weight",
            transport_weight,
        ),
    ):
        if tuple(value.shape) != (
            expected_mask_shape
        ):
            raise ValueError(
                f"{name} shape mismatch: "
                f"{tuple(value.shape)}"
            )

    if bool(
        (
            hard_mask
            & soft_only_mask
        ).any()
    ):
        raise ValueError(
            "hard and soft-only masks overlap"
        )

    target_temporal = (
        temporal_statistics(target)
    )

    rows = []

    for threshold in thresholds:
        for alpha in alphas:
            rows.append(
                evaluate_candidate(
                    alpha=alpha,
                    threshold=threshold,
                    target=target,
                    raw_correct=raw_correct,
                    raw_shuffled=raw_shuffled,
                    hard_mask=hard_mask,
                    soft_only_mask=soft_only_mask,
                    transport_weight=transport_weight,
                    target_temporal=target_temporal,
                )
            )

    reference_matches = [
        row
        for row in rows
        if abs(
            row["alpha"]
            - args.reference_alpha
        ) < 1e-9
        and abs(
            row["threshold"]
            - args.reference_threshold
        ) < 1e-9
    ]

    if len(reference_matches) != 1:
        raise ValueError(
            "reference alpha/threshold pair "
            "must occur exactly once"
        )

    reference = reference_matches[0]

    reference_identity = max(
        reference[
            "full_correct_shuffled_l1"
        ],
        1e-12,
    )

    reference_distortion = max(
        reference[
            "full_correct_target_l1"
        ],
        1e-12,
    )

    reference_temporal = max(
        reference[
            "temporal_penalty"
        ],
        1e-12,
    )

    reference_support = max(
        reference[
            "support_cells"
        ],
        1,
    )

    for row in rows:
        row["identity_retention"] = (
            row[
                "full_correct_shuffled_l1"
            ]
            / reference_identity
        )

        row[
            "target_distortion_ratio"
        ] = (
            row[
                "full_correct_target_l1"
            ]
            / reference_distortion
        )

        row[
            "temporal_penalty_ratio"
        ] = (
            row[
                "temporal_penalty"
            ]
            / reference_temporal
        )

        row["support_ratio"] = (
            row["support_cells"]
            / reference_support
        )

    nonzero_rows = [
        row
        for row in rows
        if row["alpha"] > 0
    ]

    frontier = pareto_frontier(
        nonzero_rows
    )

    selected = select_representatives(
        rows=rows,
        frontier=frontier,
        reference=reference,
        max_candidates=args.max_candidates,
    )

    artifact_hash = sha256(
        artifact_path
    )

    candidate_records = []

    for row in selected:
        threshold = row[
            "threshold"
        ]

        alpha = row["alpha"]

        gated_mask = (
            hard_mask
            | (
                soft_only_mask
                & (
                    transport_weight
                    >= threshold
                )
            )
        )

        mask_5d = (
            gated_mask.unsqueeze(0)
        )

        correct = torch.where(
            mask_5d,
            target
            + alpha
            * (
                raw_correct
                - target
            ),
            target,
        )

        shuffled = torch.where(
            mask_5d,
            target
            + alpha
            * (
                raw_shuffled
                - target
            ),
            target,
        )

        path = (
            candidate_dir
            / (
                row["candidate_id"]
                + ".pt"
            )
        )

        candidate_state = {
            "format_version": 1,
            "artifact_kind": (
                "wan_vae_alpha_tau_candidate"
            ),
            "candidate_id": row[
                "candidate_id"
            ],
            "alpha": float(alpha),
            "threshold": float(
                threshold
            ),
            "target_latent": (
                target.contiguous()
            ),
            "correct_fused_latent": (
                correct.contiguous()
            ),
            "shuffled_fused_latent": (
                shuffled.contiguous()
            ),
            "transport_mask": (
                gated_mask.contiguous()
            ),
            "latent_frame_indices": (
                state[
                    "latent_frame_indices"
                ]
                .detach()
                .cpu()
                .contiguous()
            ),
            "transport_validity_mode": (
                state.get(
                    "transport_validity_mode"
                )
            ),
            "base_artifact": {
                "path": str(
                    artifact_path
                ),
                "sha256": artifact_hash,
            },
            "latent_scan_metrics": row,
        }

        torch.save(
            candidate_state,
            path,
        )

        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if not torch.equal(
            loaded[
                "transport_mask"
            ],
            gated_mask,
        ):
            raise RuntimeError(
                "saved candidate mask mismatch"
            )

        candidate_records.append(
            {
                "candidate_id": row[
                    "candidate_id"
                ],
                "path": str(
                    path.resolve()
                ),
                "sha256": sha256(path),
                "alpha": float(alpha),
                "threshold": float(
                    threshold
                ),
                "latent_scan_metrics": row,
            }
        )

    fieldnames = list(
        rows[0].keys()
    )

    with (
        output_dir
        / "all_parameter_results.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    scan_report = {
        "stage": (
            "wan_transport_alpha_tau_latent_scan"
        ),
        "base_artifact": {
            "path": str(
                artifact_path
            ),
            "sha256": artifact_hash,
            "transport_validity_mode": (
                state.get(
                    "transport_validity_mode"
                )
            ),
        },
        "grid": {
            "alphas": alphas,
            "thresholds": thresholds,
            "combination_count": (
                len(alphas)
                * len(thresholds)
            ),
        },
        "target_temporal": (
            target_temporal
        ),
        "reference": reference,
        "pareto_frontier": frontier,
        "selected_candidates": (
            candidate_records
        ),
        "selection_note": (
            "Candidates span target fidelity and identity retention. "
            "Alpha zero is retained in the full CSV as the no-transport "
            "baseline but is not sent to VAE decoding."
        ),
    }

    (
        output_dir
        / "latent_scan_report.json"
    ).write_text(
        json.dumps(
            scan_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    base_manifest = {
        "base_artifact": str(
            artifact_path
        ),
        "base_artifact_sha256": (
            artifact_hash
        ),
        "evaluation_mask_artifact": (
            str(artifact_path)
        ),
        "evaluation_mask_key": (
            "raw_soft_transport_mask"
        ),
    }

    sorted_candidates = sorted(
        candidate_records,
        key=lambda item: (
            item[
                "latent_scan_metrics"
            ][
                "identity_retention"
            ],
            item["threshold"],
        ),
    )

    gpu2_candidates = (
        sorted_candidates[::2]
    )

    gpu3_candidates = (
        sorted_candidates[1::2]
    )

    for name, candidates in (
        (
            "gpu2_manifest.json",
            gpu2_candidates,
        ),
        (
            "gpu3_manifest.json",
            gpu3_candidates,
        ),
    ):
        manifest = {
            **base_manifest,
            "candidates": candidates,
        }

        (
            output_dir / name
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output_dir": str(
                    output_dir
                ),
                "grid_combinations": len(
                    rows
                ),
                "pareto_count": len(
                    frontier
                ),
                "selected_count": len(
                    candidate_records
                ),
                "selected": [
                    {
                        "candidate_id": item[
                            "candidate_id"
                        ],
                        "alpha": item[
                            "alpha"
                        ],
                        "threshold": item[
                            "threshold"
                        ],
                        "identity_retention": item[
                            "latent_scan_metrics"
                        ][
                            "identity_retention"
                        ],
                        "target_distortion_ratio": item[
                            "latent_scan_metrics"
                        ][
                            "target_distortion_ratio"
                        ],
                    }
                    for item in candidate_records
                ],
                "gpu2_count": len(
                    gpu2_candidates
                ),
                "gpu3_count": len(
                    gpu3_candidates
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
