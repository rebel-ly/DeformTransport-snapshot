"""Build temporal alpha schedules for selected Wan transport candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schedule_values(
    *,
    alpha: float,
    schedule_name: str,
    slots: int,
) -> torch.Tensor:
    if schedule_name == "freeze0":
        values = [0.0] + [alpha] * (slots - 1)

    elif schedule_name == "ramp2":
        values = [0.0, alpha / 2.0] + [alpha] * (slots - 2)

    elif schedule_name == "ramp4":
        values = [
            0.0,
            alpha / 3.0,
            2.0 * alpha / 3.0,
        ] + [alpha] * (slots - 3)

    else:
        raise ValueError(f"unknown schedule: {schedule_name}")

    return torch.tensor(
        values,
        dtype=torch.float32,
    ).view(1, slots, 1, 1, 1)


def candidate_id(
    group: str,
    alpha: float,
    threshold: float,
    schedule_name: str,
) -> str:
    alpha_token = f"{alpha:.3f}".replace(".", "p")
    threshold_token = f"{threshold:.3f}".replace(".", "p")
    return (
        f"{group}_a{alpha_token}"
        f"_t{threshold_token}"
        f"_{schedule_name}"
    )


def main() -> None:
    args = parse_args()

    artifact_path = args.artifact.resolve()
    output_dir = args.output_dir.resolve()

    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir()

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

    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"missing artifact keys: {missing}")

    target = state["target_latent"].float().contiguous()
    raw_correct = state["correct_transported_latent"].float().contiguous()
    raw_shuffled = state["shuffled_transported_latent"].float().contiguous()

    hard_mask = state["hard_reference_mask"].bool().contiguous()
    soft_only_mask = state["soft_only_mask"].bool().contiguous()
    transport_weight = state["transport_weight"].float().contiguous()

    if tuple(target.shape) != (1, 21, 16, 60, 104):
        raise ValueError(f"unexpected target shape: {tuple(target.shape)}")

    if raw_correct.shape != target.shape:
        raise ValueError("correct transported latent shape mismatch")

    if raw_shuffled.shape != target.shape:
        raise ValueError("shuffled transported latent shape mismatch")

    expected_mask_shape = (21, 1, 60, 104)

    for name, value in (
        ("hard_mask", hard_mask),
        ("soft_only_mask", soft_only_mask),
        ("transport_weight", transport_weight),
    ):
        if tuple(value.shape) != expected_mask_shape:
            raise ValueError(f"{name} shape mismatch: {tuple(value.shape)}")

    threshold = float(args.threshold)

    gated_mask = (
        hard_mask
        | (
            soft_only_mask
            & (transport_weight >= threshold)
        )
    )

    mask_5d = gated_mask.unsqueeze(0)

    specifications = [
        ("quality", 0.25, "freeze0"),
        ("quality", 0.25, "ramp2"),
        ("quality", 0.25, "ramp4"),
        ("balanced", 0.50, "freeze0"),
        ("balanced", 0.50, "ramp2"),
        ("balanced", 0.50, "ramp4"),
    ]

    records = []
    paths_by_group: dict[str, list[dict]] = {
        "quality": [],
        "balanced": [],
    }

    for group, alpha, schedule_name in specifications:
        schedule = schedule_values(
            alpha=alpha,
            schedule_name=schedule_name,
            slots=target.shape[1],
        )

        correct = torch.where(
            mask_5d,
            target + schedule * (raw_correct - target),
            target,
        ).contiguous()

        shuffled = torch.where(
            mask_5d,
            target + schedule * (raw_shuffled - target),
            target,
        ).contiguous()

        identifier = candidate_id(
            group,
            alpha,
            threshold,
            schedule_name,
        )

        path = candidate_dir / f"{identifier}.pt"

        candidate_state = {
            "format_version": 1,
            "artifact_kind": "wan_temporal_alpha_schedule_candidate",
            "candidate_id": identifier,
            "group": group,
            "alpha": alpha,
            "threshold": threshold,
            "schedule_name": schedule_name,
            "alpha_schedule": schedule.flatten().tolist(),
            "target_latent": target,
            "correct_fused_latent": correct,
            "shuffled_fused_latent": shuffled,
            "transport_mask": gated_mask,
            "latent_frame_indices": (
                state["latent_frame_indices"]
                .detach()
                .cpu()
                .contiguous()
            ),
            "transport_validity_mode": state.get(
                "transport_validity_mode"
            ),
            "base_artifact": {
                "path": str(artifact_path),
                "sha256": sha256(artifact_path),
            },
        }

        torch.save(candidate_state, path)

        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        checks = {
            "target_slot0_exact": torch.equal(
                loaded["target_latent"][:, 0],
                target[:, 0],
            ),
            "correct_slot0_exact": torch.equal(
                loaded["correct_fused_latent"][:, 0],
                target[:, 0],
            ),
            "shuffled_slot0_exact": torch.equal(
                loaded["shuffled_fused_latent"][:, 0],
                target[:, 0],
            ),
            "correct_finite": bool(
                torch.isfinite(
                    loaded["correct_fused_latent"]
                ).all()
            ),
            "shuffled_finite": bool(
                torch.isfinite(
                    loaded["shuffled_fused_latent"]
                ).all()
            ),
            "mask_equal": torch.equal(
                loaded["transport_mask"],
                gated_mask,
            ),
        }

        if not all(checks.values()):
            raise RuntimeError(
                f"candidate checks failed for {identifier}: {checks}"
            )

        record = {
            "candidate_id": identifier,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "group": group,
            "alpha": alpha,
            "threshold": threshold,
            "schedule_name": schedule_name,
            "alpha_schedule": schedule.flatten().tolist(),
            "checks": checks,
        }

        records.append(record)
        paths_by_group[group].append(record)

    evaluation_artifact = records[0]["path"]

    for filename, group in (
        ("gpu2_quality_manifest.json", "quality"),
        ("gpu3_balanced_manifest.json", "balanced"),
    ):
        manifest = {
            "base_artifact": str(artifact_path),
            "base_artifact_sha256": sha256(artifact_path),
            "evaluation_mask_artifact": evaluation_artifact,
            "evaluation_mask_key": "transport_mask",
            "candidates": paths_by_group[group],
        }

        (output_dir / filename).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    report = {
        "stage": "temporal_alpha_schedule_candidate_build",
        "base_artifact": {
            "path": str(artifact_path),
            "sha256": sha256(artifact_path),
        },
        "threshold": threshold,
        "candidate_count": len(records),
        "candidates": records,
        "all_checks_pass": all(
            all(record["checks"].values())
            for record in records
        ),
        "boundary_contract": (
            "Latent slot zero is exactly equal to the target "
            "latent for both Correct and Shuffled."
        ),
    }

    report_path = output_dir / "build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_count": len(records),
                "gpu2_count": len(paths_by_group["quality"]),
                "gpu3_count": len(paths_by_group["balanced"]),
                "all_checks_pass": report["all_checks_pass"],
                "candidates": [
                    {
                        "candidate_id": item["candidate_id"],
                        "schedule": item["alpha_schedule"][:5],
                        "checks": item["checks"],
                    }
                    for item in records
                ],
            },
            indent=2,
        )
    )

    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
