#!/usr/bin/env python
"""用 RealWonder 预计算 demo 资产运行可复现的离线物理仿真。

该入口复用 ``demo_web/simulation_engine.py`` 的 ``_MinimalSVR``，因此不会
加载 SAM3D、SAM2、MoGe 或 Flux 重建前端。输出遵循 ``infer_sim.py`` 所需
的 final_sim 物理帧、mask、flow、首帧和配置契约；结构化 noise 由后续独立
RAFT/noise 步骤生成。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "demo_web"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from case_handlers.base import get_demo_case_handler  # noqa: E402
import case_handlers  # noqa: E402,F401 触发官方 demo handler 注册
from config import TEMPORAL_FACTOR  # noqa: E402
from simulation.utils import resize_and_crop_pil, save_video_from_pil  # noqa: E402
from simulation_engine import InteractiveSimulator  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def downsample_masks(
    masks: list[torch.Tensor | None], *, crop_start: int
) -> torch.Tensor:
    """镜像 RealWonder offline ``preprocess_masks_downsample``。"""
    if not masks:
        raise ValueError("mask 列表为空")
    prepared = []
    for mask in masks:
        if mask is None:
            prepared.append(torch.zeros((512, 512), dtype=torch.float32))
            continue
        value = torch.as_tensor(mask).detach().cpu().float()
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.shape != (512, 512):
            raise ValueError(f"mask 必须为 512x512，实际为 {tuple(value.shape)}")
        prepared.append(value)

    stacked = torch.stack(prepared, dim=0).unsqueeze(1)
    resized = F.interpolate(
        stacked, size=(832, 832), mode="bilinear", align_corners=False
    )
    cropped = resized[:, :, crop_start : crop_start + 480, :]
    latent = F.interpolate(
        cropped, size=(60, 104), mode="bilinear", align_corners=False
    ).squeeze(1)
    groups = [
        latent[i : i + TEMPORAL_FACTOR].mean(dim=0, keepdim=True)
        for i in range(0, latent.shape[0], TEMPORAL_FACTOR)
    ]
    return torch.cat(groups, dim=0).gt(0.5).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction", choices=("left", "none", "right"), default="right")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-pixel-frames", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo_data = args.demo_data.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir()

    config = OmegaConf.to_container(
        OmegaConf.load(demo_data / "config.yaml"), resolve=True
    )
    latent_frames = int(config["num_output_frames"])
    expected_pixel_frames = (latent_frames - 1) * TEMPORAL_FACTOR + 1
    num_pixel_frames = args.num_pixel_frames or expected_pixel_frames
    if num_pixel_frames != expected_pixel_frames:
        raise ValueError(
            f"原生契约要求 {expected_pixel_frames} 个 pixel 帧，实际请求 {num_pixel_frames}"
        )

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    overrides = {
        "output_folder": str(output / "sim_tmp"),
        "seed": args.seed,
        "debug": False,
        "export_point_trajectories": True,
        "disable_genesis_visualizer": True,
    }
    if demo_data.name == "santa_cloth":
        overrides["skip_force_fields"] = True

    simulator = InteractiveSimulator(
        str(demo_data), device="cuda", config_overrides=overrides
    )
    handler = get_demo_case_handler(demo_data.name, simulator.config)
    handler.set_object_masks(simulator.object_masks_b64)
    simulator.set_demo_case_handler(handler)
    force_config = handler.get_force_config_from_ui(
        [{"obj_idx": 0, "direction": args.direction, "strength": args.strength}]
    )
    handler.set_forces(force_config)
    handler.configure_simulation(simulator)

    frames: list[Image.Image] = []
    point_masks: list[torch.Tensor | None] = []
    mesh_masks: list[torch.Tensor | None] = []
    for frame_index in range(num_pixel_frames):
        updated_points = None
        for substep_index in range(simulator.frame_steps):
            updated_points = simulator.step(
                extract_points=substep_index == simulator.frame_steps - 1
            )
        if updated_points is None:
            raise RuntimeError("物理步未返回点云")
        frame, _, point_mask, mesh_mask = simulator.render_and_flow(
            updated_points, frame_id=frame_index
        )
        frame = resize_and_crop_pil(frame, start_y=simulator.crop_start)
        if frame.size != (832, 480):
            raise RuntimeError(f"渲染帧尺寸错误：{frame.size}")
        frame.save(frames_dir / f"frame_{frame_index:04d}.png")
        frames.append(frame)
        point_masks.append(point_mask)
        mesh_masks.append(mesh_mask)

    flow = np.asarray(simulator.svr.optical_flow, dtype=np.float32)[..., :2]
    if flow.shape != (num_pixel_frames - 1, 512, 512, 2):
        raise RuntimeError(f"几何 flow 形状错误：{flow.shape}")
    np.save(output / "flows_geometry.npy", flow.transpose(0, 3, 1, 2))
    torch.save(
        downsample_masks(point_masks, crop_start=simulator.crop_start),
        output / "points_masks_downsampled.pt",
    )
    torch.save(
        downsample_masks(mesh_masks, crop_start=simulator.crop_start),
        output / "mesh_masks_downsampled.pt",
    )
    simulator.save_point_trajectories(output / "point_trajectories.pt")

    first_frame = Image.open(demo_data / "first_frame.png").convert("RGB")
    if first_frame.size != (832, 480):
        first_frame = resize_and_crop_pil(first_frame, start_y=simulator.crop_start)
    first_frame.save(output / "resized_input_image.png")
    (output / "prompt.txt").write_text(
        str(simulator.config["vgen_prompt"]).strip() + "\n", encoding="utf-8"
    )
    config_out = dict(simulator.config)
    config_out["output_folder"] = str(output)
    config_out["seed"] = args.seed
    OmegaConf.save(OmegaConf.create(config_out), output / "config.yaml")
    save_video_from_pil(frames, str(output / "simulation.mp4"), fps=10)

    report = {
        "任务": "RealWonder预计算官方case离线物理仿真",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "case": demo_data.name,
        "demo_data": str(demo_data),
        "输出": str(output),
        "seed": args.seed,
        "动作": force_config,
        "pixel帧数": len(frames),
        "latent帧数": latent_frames,
        "frame_steps": simulator.frame_steps,
        "flow形状": list(flow.shape),
        "points_mask形状": list(torch.load(output / "points_masks_downsampled.pt", weights_only=True).shape),
        "mesh_mask形状": list(torch.load(output / "mesh_masks_downsampled.pt", weights_only=True).shape),
        "重建前端": "未加载；复用demo_web预计算资产与_MinimalSVR",
    }
    (output / "simulation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.rmtree(output / "sim_tmp", ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
