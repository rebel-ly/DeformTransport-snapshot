from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO = Path("/workspace/DeformTransport")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# noise_warp.py imports two optional modules at import time.
# They are irrelevant here:
# - fire is CLI-only
# - BackgroundRemover is unused because remove_background=False
import types

sys.modules.setdefault(
    "fire",
    types.ModuleType("fire"),
)

bg = types.ModuleType(
    "simulation.image23D.noise_warp.background_remover"
)
bg.BackgroundRemover = object
sys.modules[
    "simulation.image23D.noise_warp.background_remover"
] = bg

from simulation.image23D.noise_warp import noise_warp as nw


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    frames_dir = root / "frames"

    frame_paths = sorted(
        frames_dir.glob("frame_*.png")
    )

    if len(frame_paths) != 165:
        raise RuntimeError(
            f"expected 165 frames, got {len(frame_paths)}"
        )

    expected_names = [
        f"frame_{i:04d}.png"
        for i in range(165)
    ]

    if [
        p.name
        for p in frame_paths
    ] != expected_names:
        raise RuntimeError(
            "frame filenames are not exactly "
            "frame_0000.png ... frame_0164.png"
        )

    # --------------------------------------------------
    # Fixed seed.
    # --------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------
    # Load the exact 165 RealWonder-preprocessed frames.
    # They are already 480 x 832.
    # --------------------------------------------------
    frames = []

    for path in frame_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")

            if image.size != (832, 480):
                raise ValueError(
                    f"{path}: expected 832x480, "
                    f"got {image.size}"
                )

            frames.append(
                np.asarray(
                    image,
                    dtype=np.uint8,
                ).copy()
            )

    video_raw = np.stack(
        frames,
        axis=0,
    )

    if video_raw.shape != (
        165,
        480,
        832,
        3,
    ):
        raise RuntimeError(
            f"unexpected raw video shape "
            f"{video_raw.shape}"
        )

    # --------------------------------------------------
    # EXACT successful Santa preprocessing:
    #
    # [T,H,W,C]
    # -> [T,C,H,W]
    # -> area resize to 240x416
    # -> uint8 [T,H,W,C]
    #
    # This replaces the broken resize_frames=0.5
    # code path which depended on cv2.
    # --------------------------------------------------
    video_t = torch.from_numpy(
        video_raw.copy()
    )

    video_t = (
        video_t
        .permute(0, 3, 1, 2)
        .float()
    )

    video_t = F.interpolate(
        video_t,
        size=(240, 416),
        mode="area",
    )

    video = (
        video_t
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )

    if video.shape != (
        165,
        240,
        416,
        3,
    ):
        raise RuntimeError(
            f"unexpected RAFT input shape "
            f"{video.shape}"
        )

    print(
        "RAW_VIDEO_SHAPE =",
        video_raw.shape,
    )

    print(
        "RAFT_INPUT_SHAPE =",
        video.shape,
    )

    start = time.perf_counter()

    # --------------------------------------------------
    # EXACT successful Santa noise-warp parameters.
    # --------------------------------------------------
    nw.get_noise_from_video(
        video,
        noise_channels=32,
        input_flow=False,
        output_folder=str(root),
        visualize=False,
        resize_frames=None,
        resize_flow=8,
        downscale_factor=32,
        device="cuda",
        save_files=True,
        remove_background=False,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    noise_path = (
        root
        / "noises.npy"
    )

    flow_path = (
        root
        / "flows.npy"
    )

    if not noise_path.is_file():
        raise FileNotFoundError(
            noise_path
        )

    if not flow_path.is_file():
        raise FileNotFoundError(
            flow_path
        )

    noises = np.load(
        noise_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    flows = np.load(
        flow_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    checks = {
        "noise_shape":
            list(noises.shape),

        "noise_dtype":
            str(noises.dtype),

        "noise_finite":
            bool(
                np.isfinite(noises).all()
            ),

        "flow_shape":
            list(flows.shape),

        "flow_dtype":
            str(flows.dtype),

        "flow_finite":
            bool(
                np.isfinite(flows).all()
            ),
    }

    hard_checks = {
        "noise_shape_exact":
            tuple(noises.shape)
            == (
                165,
                60,
                104,
                32,
            ),

        "noise_dtype_float16":
            noises.dtype
            == np.float16,

        "noise_finite":
            bool(
                np.isfinite(noises).all()
            ),

        "flow_shape_exact":
            tuple(flows.shape)
            == (
                164,
                2,
                240,
                416,
            ),

        "flow_dtype_float16":
            flows.dtype
            == np.float16,

        "flow_finite":
            bool(
                np.isfinite(flows).all()
            ),
    }

    all_checks_pass = all(
        hard_checks.values()
    )

    report = {
        "生成时间":
            datetime.datetime.now(
                datetime.timezone(
                    datetime.timedelta(
                        hours=8
                    )
                )
            ).isoformat(),

        "case":
            "sand_house",

        "experiment_definition":
            (
                "SandHouse cached simulation geometry "
                "+ SandHouse robot-action dynamics"
            ),

        "seed":
            args.seed,

        "算法":
            (
                "项目simulation/image23D/"
                "noise_warp/noise_warp.py"
                "::get_noise_from_video"
            ),

        "光流模型":
            (
                "torchvision RAFT-Large "
                "C_T_SKHT_V2"
            ),

        "权重来源":
            (
                "https://download.pytorch.org/"
                "models/"
                "raft_large_C_T_SKHT_V2-"
                "ff5fadd5.pth"
            ),

        "权重SHA256":
            (
                "ff5fadd56d26b406"
                "47388883af154735"
                "1ea17868b765c05b"
                "27231e72dd16a322"
            ),

        "输入帧":
            len(frame_paths),

        "原始输入shape":
            list(video_raw.shape),

        "RAFT输入shape":
            list(video.shape),

        "预处理":
            (
                "torch.nn.functional.interpolate("
                "mode=area,size=(240,416)); "
                "equivalent replacement for "
                "original resize_frames=0.5 "
                "while bypassing unavailable cv2"
            ),

        "参数": {
            "noise_channels":
                32,

            "resize_frames":
                None,

            "等效预缩放":
                0.5,

            "resize_flow":
                8,

            "downscale_factor":
                32,

            "remove_background":
                False,
        },

        "检查":
            checks,

        "hard_checks":
            hard_checks,

        "all_checks_pass":
            all_checks_pass,

        "耗时秒":
            elapsed,

        "torch峰值allocated_MiB":
            (
                torch.cuda
                .max_memory_allocated()
                / 1048576
            ),

        "torch峰值reserved_MiB":
            (
                torch.cuda
                .max_memory_reserved()
                / 1048576
            ),

        "noises_SHA256":
            sha256(
                noise_path
            ),

        "flows_SHA256":
            sha256(
                flow_path
            ),
    }

    report_path = (
        root
        / "noise_generation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not all_checks_pass:
        raise RuntimeError(
            "Tree RAFT/noise validation failed"
        )

    print()
    print(
        "TREE_RAFT_NOISE_GENERATION_OK"
    )


if __name__ == "__main__":
    main()
