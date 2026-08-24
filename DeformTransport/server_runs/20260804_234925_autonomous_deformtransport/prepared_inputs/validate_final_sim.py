#!/usr/bin/env python
"""不初始化CUDA的RealWonder final_sim完整性验证器。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def tensor_info(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    return {"形状": list(value.shape), "类型": str(value.dtype), "非空": bool(value.numel())}


def main() -> int:
    parser = argparse.ArgumentParser(description="验证final_sim与transport artifact的CPU静态兼容性")
    parser.add_argument("final_sim", type=Path)
    parser.add_argument("--transport-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.final_sim
    report = {"目录": str(root), "通过": True, "错误": [], "警告": [], "检查": {}}
    required = ["config.yaml", "noises.npy", "resized_input_image.png", "prompt.txt"]
    for name in required:
        path = root / name
        report["检查"][name] = {"存在": path.is_file(), "大小字节": path.stat().st_size if path.is_file() else 0}
        if not path.is_file():
            report["错误"].append("缺失必需文件：" + name)

    frames = sorted((root / "frames").glob("frame_*.png")) if (root / "frames").is_dir() else []
    report["检查"]["frames"] = {"数量": len(frames), "目录存在": (root / "frames").is_dir()}
    if not frames:
        report["错误"].append("缺失frames/frame_*.png")

    config = None
    if (root / "config.yaml").is_file():
        with (root / "config.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        latent_frames = int(config["num_output_frames"])
        expected_pixel_frames = latent_frames * 4 - 3
        report["检查"]["时序"] = {
            "配置latent帧数": latent_frames,
            "期望pixel帧数": expected_pixel_frames,
            "实际coarse帧数": len(frames),
            "每块latent帧数": 3,
        }
        if latent_frames % 3:
            report["错误"].append("num_output_frames必须能被当前num_frame_per_block=3整除")
        if frames and len(frames) != expected_pixel_frames:
            report["错误"].append("coarse RGB帧数不等于4*T-3")
        if not config.get("denoising_step_list"):
            report["错误"].append("denoising_step_list为空")

    if (root / "noises.npy").is_file():
        noise = np.load(root / "noises.npy", mmap_mode="r", allow_pickle=False)
        report["检查"]["noise"] = {"形状": list(noise.shape), "类型": str(noise.dtype)}
        if noise.ndim != 4:
            report["错误"].append("noises.npy必须是4维")
        else:
            if noise.shape[1:3] != (60, 104):
                report["错误"].append("noise空间尺寸必须是60x104")
            if noise.shape[3] < 16:
                report["错误"].append("noise通道数少于16")
            if frames and noise.shape[0] != len(frames):
                report["错误"].append("noise原始时间长度与coarse RGB帧数不一致")

    if (root / "resized_input_image.png").is_file():
        with Image.open(root / "resized_input_image.png") as image:
            report["检查"]["初始图像"] = {"尺寸": list(image.size), "模式": image.mode}
            if image.size != (832, 480):
                report["警告"].append("初始图像不是832x480，infer_sim会再次resize")

    if (root / "prompt.txt").is_file():
        prompt = (root / "prompt.txt").read_text(encoding="utf-8").strip()
        report["检查"]["prompt"] = {"非空": bool(prompt), "字符数": len(prompt)}
        if not prompt:
            report["错误"].append("prompt为空")

    for name in ["points_masks_downsampled.pt", "mesh_masks_downsampled.pt"]:
        path = root / name
        if path.is_file():
            report["检查"][name] = tensor_info(path)
        elif config and (name.startswith("points_") and int(config.get("mask_dropin_step", -1)) > 0):
            report["错误"].append("mask_dropin_step>0但缺失" + name)
        else:
            report["警告"].append("未提供可选文件：" + name)

    if args.transport_artifact:
        if not args.transport_artifact.is_file():
            report["错误"].append("transport artifact不存在")
        else:
            artifact = torch.load(args.transport_artifact, map_location="cpu", weights_only=True)
            correct = artifact.get("correct_fused_latent")
            shuffled = artifact.get("shuffled_fused_latent")
            report["检查"]["transport"] = {
                "Correct形状": list(correct.shape) if isinstance(correct, torch.Tensor) else None,
                "Shuffled形状": list(shuffled.shape) if isinstance(shuffled, torch.Tensor) else None,
                "二者不同": bool(not torch.equal(correct, shuffled)) if isinstance(correct, torch.Tensor) and isinstance(shuffled, torch.Tensor) else False,
            }
            if config and isinstance(correct, torch.Tensor) and correct.shape[1] != int(config["num_output_frames"]):
                report["错误"].append("transport latent时间长度与config.num_output_frames不一致")
            if frames and isinstance(correct, torch.Tensor):
                expected_latent = 1 + (len(frames) - 1) // 4
                if (len(frames) - 1) % 4 or correct.shape[1] != expected_latent:
                    report["错误"].append("transport latent与coarse RGB因果VAE时序不一致")

    report["通过"] = not report["错误"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["通过"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
