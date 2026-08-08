"""Decode a manifest of Wan alpha/tau candidates with one VAE model load."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.wan_vae_codec import RealWonderWanVAECodec  # noqa: E402


EXPECTED_VAE_SHA256 = (
    "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
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


def tensor_sha256(
    tensor: torch.Tensor,
) -> str:
    value = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
    )

    digest = hashlib.sha256()
    digest.update(value.tobytes())

    return digest.hexdigest()


def timed_cuda(function):
    torch.cuda.synchronize()
    started = time.perf_counter()

    output = function()

    torch.cuda.synchronize()

    return (
        output,
        time.perf_counter()
        - started,
    )


def pixels_01(
    decoded: torch.Tensor,
) -> torch.Tensor:
    return (
        decoded[0]
        .mul(0.5)
        .add(0.5)
        .clamp(0.0, 1.0)
        .contiguous()
    )


def uint8_frames(
    frames: torch.Tensor,
) -> np.ndarray:
    return (
        frames.detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )


def write_video(
    path: Path,
    frames: np.ndarray,
    fps: int,
) -> None:
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
    )

    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def expand_latent_mask(
    mask: torch.Tensor,
    pixel_frames: int = 81,
) -> torch.Tensor:
    mask = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device="cpu",
    )

    expanded = [mask[:1]]

    for index in range(
        1,
        mask.shape[0],
    ):
        expanded.append(
            mask[
                index : index + 1
            ].repeat(
                4,
                1,
                1,
                1,
            )
        )

    result = torch.cat(
        expanded,
        dim=0,
    )

    if result.shape[0] < pixel_frames:
        raise RuntimeError(
            "expanded mask has too few frames"
        )

    return result[:pixel_frames]


def pixel_mask(
    latent_mask: torch.Tensor,
) -> torch.Tensor:
    return F.interpolate(
        expand_latent_mask(
            latent_mask
        ).float(),
        size=(480, 832),
        mode="nearest",
    ).bool()


def aggregate_error(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int | None]:
    l1_values = []
    mse_values = []
    psnr_values = []

    for frame_index in range(
        first.shape[0]
    ):
        expanded = mask[
            frame_index
        ].expand(
            first.shape[1],
            -1,
            -1,
        )

        values = (
            first[frame_index]
            - second[frame_index]
        )[expanded]

        if values.numel() == 0:
            continue

        l1 = float(
            values.abs().mean()
        )

        mse = float(
            values.square().mean()
        )

        l1_values.append(l1)
        mse_values.append(mse)

        if mse > 0:
            psnr_values.append(
                float(
                    10.0
                    * np.log10(
                        1.0 / mse
                    )
                )
            )

    return {
        "valid_frames": len(
            l1_values
        ),
        "mean_masked_l1": float(
            np.mean(l1_values)
        ),
        "mean_masked_mse": float(
            np.mean(mse_values)
        ),
        "mean_masked_psnr_db": (
            float(
                np.mean(
                    psnr_values
                )
            )
            if psnr_values
            else None
        ),
    }


def sharpness(
    frames: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    gray = (
        frames[:, 0:1]
        * 0.2989
        + frames[:, 1:2]
        * 0.5870
        + frames[:, 2:3]
        * 0.1140
    ).float()

    lap_kernel = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ).view(1, 1, 3, 3)

    sx_kernel = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]
    ).view(1, 1, 3, 3)

    sy_kernel = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ]
    ).view(1, 1, 3, 3)

    lap = F.conv2d(
        gray,
        lap_kernel,
        padding=1,
    )

    sx = F.conv2d(
        gray,
        sx_kernel,
        padding=1,
    )

    sy = F.conv2d(
        gray,
        sy_kernel,
        padding=1,
    )

    sobel = torch.sqrt(
        sx.square()
        + sy.square()
        + 1e-12
    )

    low = F.avg_pool2d(
        gray,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    high = (
        gray - low
    ).abs()

    lap_values = []
    sobel_values = []
    high_values = []

    for frame_index in range(
        frames.shape[0]
    ):
        selected = mask[
            frame_index,
            0,
        ]

        lap_values.append(
            float(
                lap[
                    frame_index,
                    0,
                ][selected].var(
                    unbiased=False
                )
            )
        )

        sobel_values.append(
            float(
                sobel[
                    frame_index,
                    0,
                ][selected].mean()
            )
        )

        high_values.append(
            float(
                high[
                    frame_index,
                    0,
                ][selected].mean()
            )
        )

    return {
        "mean_laplacian_variance": float(
            np.mean(lap_values)
        ),
        "mean_sobel": float(
            np.mean(sobel_values)
        ),
        "mean_high_frequency": float(
            np.mean(high_values)
        ),
    }


def temporal(
    frames: torch.Tensor,
) -> dict[str, float]:
    first = (
        frames[1:]
        - frames[:-1]
    ).abs()

    second = (
        frames[2:]
        - 2.0 * frames[1:-1]
        + frames[:-2]
    ).abs()

    return {
        "mean_first_order_abs_difference": float(
            first.mean()
        ),
        "mean_second_order_abs_difference": float(
            second.mean()
        ),
    }


def comparison_frames(
    *,
    target: np.ndarray,
    correct: np.ndarray,
    shuffled: np.ndarray,
    label: str,
) -> np.ndarray:
    result = []

    for index in range(len(target)):
        panels = []

        for panel_label, frames in (
            (
                "Target VAE reconstruction",
                target,
            ),
            (
                f"{label} Correct",
                correct,
            ),
            (
                f"{label} Shuffled",
                shuffled,
            ),
        ):
            image = Image.fromarray(
                frames[index]
            ).resize(
                (416, 240),
                resample=Image.Resampling.BILINEAR,
            )

            panel = Image.new(
                "RGB",
                (416, 272),
                "black",
            )

            panel.paste(
                image,
                (0, 32),
            )

            ImageDraw.Draw(
                panel
            ).text(
                (8, 9),
                panel_label,
                fill="white",
            )

            panels.append(
                np.asarray(panel)
            )

        result.append(
            np.concatenate(
                panels,
                axis=1,
            )
        )

    return np.stack(result)


def main() -> None:
    args = parse_args()

    manifest_path = (
        args.manifest.resolve()
    )

    checkpoint_path = (
        args.checkpoint.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    for path in (
        manifest_path,
        checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                "output directory is not empty: "
                f"{output_dir}"
            )
    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    candidates = manifest[
        "candidates"
    ]

    if not candidates:
        raise ValueError(
            "manifest contains no candidates"
        )

    evaluation_path = Path(
        manifest[
            "evaluation_mask_artifact"
        ]
    )

    evaluation_key = manifest[
        "evaluation_mask_key"
    ]

    evaluation_state = torch.load(
        evaluation_path,
        map_location="cpu",
        weights_only=True,
    )

    if evaluation_key not in (
        evaluation_state
    ):
        raise ValueError(
            f"evaluation artifact missing "
            f"{evaluation_key}"
        )

    common_mask = pixel_mask(
        evaluation_state[
            evaluation_key
        ]
    )

    full_mask = torch.ones(
        (81, 1, 480, 832),
        dtype=torch.bool,
    )

    checkpoint_hash = sha256(
        checkpoint_path
    )

    if checkpoint_hash != (
        EXPECTED_VAE_SHA256
    ):
        raise ValueError(
            "Wan VAE checkpoint hash mismatch"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    first_state = torch.load(
        Path(
            candidates[0]["path"]
        ),
        map_location="cpu",
        weights_only=True,
    )

    target_latent = first_state[
        "target_latent"
    ].float().contiguous()

    torch.cuda.reset_peak_memory_stats()

    overall_started = (
        time.perf_counter()
    )

    codec, model_load_seconds = (
        timed_cuda(
            lambda: RealWonderWanVAECodec(
                checkpoint_path,
                device="cuda",
                dtype=torch.bfloat16,
            )
        )
    )

    def decode_cpu(
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        decoded, seconds = (
            timed_cuda(
                lambda: codec.decode_latents(
                    latent
                )
            )
        )

        value = (
            decoded.detach()
            .cpu()
            .contiguous()
        )

        del decoded

        codec.clear_cache()
        torch.cuda.empty_cache()

        return value, seconds

    target_decoded, target_seconds = (
        decode_cpu(target_latent)
    )

    if tuple(
        target_decoded.shape
    ) != (
        1,
        81,
        3,
        480,
        832,
    ):
        raise RuntimeError(
            "unexpected target decode shape"
        )

    target_01 = pixels_01(
        target_decoded
    )

    target_uint8 = uint8_frames(
        target_01
    )

    target_hash = tensor_sha256(
        torch.from_numpy(
            target_uint8
        )
    )

    write_video(
        output_dir
        / "target_reconstruction.mp4",
        target_uint8,
        args.fps,
    )

    target_sharpness = sharpness(
        target_01,
        common_mask,
    )

    results = []

    for candidate in candidates:
        candidate_path = Path(
            candidate["path"]
        )

        state = torch.load(
            candidate_path,
            map_location="cpu",
            weights_only=True,
        )

        if not torch.equal(
            state[
                "target_latent"
            ],
            target_latent,
        ):
            raise RuntimeError(
                "candidate target latent differs"
            )

        native_mask = pixel_mask(
            state[
                "transport_mask"
            ]
        )

        correct_decoded, correct_seconds = (
            decode_cpu(
                state[
                    "correct_fused_latent"
                ].float()
            )
        )

        shuffled_decoded, shuffled_seconds = (
            decode_cpu(
                state[
                    "shuffled_fused_latent"
                ].float()
            )
        )

        correct_01 = pixels_01(
            correct_decoded
        )

        shuffled_01 = pixels_01(
            shuffled_decoded
        )

        correct_uint8 = uint8_frames(
            correct_01
        )

        shuffled_uint8 = uint8_frames(
            shuffled_01
        )

        candidate_output = (
            output_dir
            / candidate[
                "candidate_id"
            ]
        )

        candidate_output.mkdir()

        write_video(
            candidate_output
            / "correct.mp4",
            correct_uint8,
            args.fps,
        )

        write_video(
            candidate_output
            / "shuffled.mp4",
            shuffled_uint8,
            args.fps,
        )

        panels = comparison_frames(
            target=target_uint8,
            correct=correct_uint8,
            shuffled=shuffled_uint8,
            label=candidate[
                "candidate_id"
            ],
        )

        write_video(
            candidate_output
            / "comparison.mp4",
            panels,
            args.fps,
        )

        selected_images = {}

        for name, index in (
            ("first", 0),
            ("quarter", 20),
            ("mid", 40),
            ("three_quarter", 60),
            ("final", 80),
        ):
            path = (
                candidate_output
                / f"{name}.png"
            )

            Image.fromarray(
                panels[index]
            ).save(path)

            selected_images[
                name
            ] = str(path)

        result = {
            "candidate_id": (
                candidate[
                    "candidate_id"
                ]
            ),
            "path": str(
                candidate_path
            ),
            "alpha": float(
                candidate["alpha"]
            ),
            "threshold": float(
                candidate[
                    "threshold"
                ]
            ),
            "latent_scan_metrics": candidate.get(
                "latent_scan_metrics",
                {
                    "group": candidate.get("group"),
                    "schedule_name": candidate.get("schedule_name"),
                    "alpha_schedule": candidate.get("alpha_schedule"),
                    "checks": candidate.get("checks"),
                },
            ),
            "common_raw_support": {
                "correct_vs_target": (
                    aggregate_error(
                        correct_01,
                        target_01,
                        common_mask,
                    )
                ),
                "shuffled_vs_target": (
                    aggregate_error(
                        shuffled_01,
                        target_01,
                        common_mask,
                    )
                ),
                "correct_vs_shuffled": (
                    aggregate_error(
                        correct_01,
                        shuffled_01,
                        common_mask,
                    )
                ),
            },
            "native_support": {
                "correct_vs_target": (
                    aggregate_error(
                        correct_01,
                        target_01,
                        native_mask,
                    )
                ),
            },
            "full_frame": {
                "correct_vs_target": (
                    aggregate_error(
                        correct_01,
                        target_01,
                        full_mask,
                    )
                ),
            },
            "sharpness_common": {
                "target": (
                    target_sharpness
                ),
                "correct": sharpness(
                    correct_01,
                    common_mask,
                ),
                "shuffled": sharpness(
                    shuffled_01,
                    common_mask,
                ),
            },
            "temporal": {
                "target": temporal(
                    target_01
                ),
                "correct": temporal(
                    correct_01
                ),
                "shuffled": temporal(
                    shuffled_01
                ),
            },
            "runtime_seconds": {
                "correct_decode": (
                    correct_seconds
                ),
                "shuffled_decode": (
                    shuffled_seconds
                ),
            },
            "artifacts": {
                "correct_video": str(
                    candidate_output
                    / "correct.mp4"
                ),
                "shuffled_video": str(
                    candidate_output
                    / "shuffled.mp4"
                ),
                "comparison_video": str(
                    candidate_output
                    / "comparison.mp4"
                ),
                "selected_images": (
                    selected_images
                ),
            },
        }

        (
            candidate_output
            / "report.json"
        ).write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

        results.append(result)

        del (
            correct_decoded,
            shuffled_decoded,
            correct_01,
            shuffled_01,
        )

    report = {
        "stage": (
            "wan_alpha_tau_candidate_batch_decode"
        ),
        "manifest": str(
            manifest_path
        ),
        "checkpoint": {
            "path": str(
                checkpoint_path
            ),
            "sha256": (
                checkpoint_hash
            ),
        },
        "evaluation_mask": {
            "path": str(
                evaluation_path
            ),
            "key": evaluation_key,
        },
        "target_uint8_sha256": (
            target_hash
        ),
        "candidate_count": len(
            results
        ),
        "target_sharpness_common": (
            target_sharpness
        ),
        "results": results,
        "runtime_seconds": {
            "model_load": (
                model_load_seconds
            ),
            "target_decode": (
                target_seconds
            ),
            "total": (
                time.perf_counter()
                - overall_started
            ),
        },
        "all_checks_pass": True,
        "interpretation_boundary": (
            "VAE-only proxy scan. Full RealWonder diffusion "
            "generation has not yet been run."
        ),
    }

    (
        output_dir
        / "batch_decode_report.json"
    ).write_text(
        json.dumps(
            report,
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
                "candidate_count": len(
                    results
                ),
                "target_uint8_sha256": (
                    target_hash
                ),
                "candidates": [
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
                        "correct_target_l1": item[
                            "common_raw_support"
                        ][
                            "correct_vs_target"
                        ][
                            "mean_masked_l1"
                        ],
                        "identity_l1": item[
                            "common_raw_support"
                        ][
                            "correct_vs_shuffled"
                        ][
                            "mean_masked_l1"
                        ],
                    }
                    for item in results
                ],
                "runtime_seconds": (
                    report[
                        "runtime_seconds"
                    ]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
