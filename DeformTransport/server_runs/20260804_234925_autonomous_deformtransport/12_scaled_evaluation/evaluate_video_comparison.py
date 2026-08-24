#!/usr/bin/env python
"""CPU视频指标：逐帧PSNR、Gaussian-window SSIM与时序差异。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


def read_video(path: Path) -> torch.Tensor:
    frames = np.stack([np.asarray(frame) for frame in iio.imiter(path, plugin="FFMPEG")])
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("视频必须是[T,H,W,C]")
    return torch.from_numpy(frames[..., :3].copy()).float().div_(255.0)


def ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma = 1.5
    coords = torch.arange(11, dtype=torch.float32) - 5
    kernel = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    kernel = (kernel / kernel.sum()).outer(kernel / kernel.sum())
    weight = kernel.expand(3, 1, 11, 11)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    mu_x = F.conv2d(x, weight, padding=5, groups=3)
    mu_y = F.conv2d(y, weight, padding=5, groups=3)
    var_x = F.conv2d(x * x, weight, padding=5, groups=3) - mu_x.square()
    var_y = F.conv2d(y * y, weight, padding=5, groups=3) - mu_y.square()
    cov = F.conv2d(x * y, weight, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--mask-video", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    ref, pred = read_video(args.reference), read_video(args.prediction)
    if ref.shape != pred.shape:
        raise ValueError("参考与预测视频shape不同：" + str((tuple(ref.shape), tuple(pred.shape))))

    mask = None
    if args.mask_video:
        raw = read_video(args.mask_video).mean(dim=-1, keepdim=True)
        if raw.shape[:3] != ref.shape[:3]:
            raise ValueError("mask视频shape不匹配")
        mask = raw > 0.5

    smap = ssim_map(ref, pred).permute(0, 2, 3, 1)
    rows = []
    for index in range(ref.shape[0]):
        diff = ref[index] - pred[index]
        if mask is None:
            values = diff.reshape(-1)
            ssim = float(smap[index].mean())
        else:
            expanded = mask[index].expand_as(diff)
            values = diff[expanded]
            expanded_ssim = mask[index].expand_as(smap[index])
            ssim = float(smap[index][expanded_ssim].mean()) if expanded_ssim.any() else float("nan")
        mse = float(values.square().mean()) if values.numel() else float("nan")
        psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
        rows.append({"帧": index, "MSE": mse, "PSNR_dB": psnr, "SSIM": ssim})

    temporal_error = 0.0
    if ref.shape[0] > 1:
        temporal_error = float(((pred[1:] - pred[:-1]) - (ref[1:] - ref[:-1])).abs().mean())

    finite_psnr = [row["PSNR_dB"] for row in rows if math.isfinite(row["PSNR_dB"])]
    summary = {
        "说明": "GT只用于离线评测；本脚本不生成或修改模型输入",
        "参考视频": str(args.reference),
        "预测视频": str(args.prediction),
        "帧数": int(ref.shape[0]),
        "分辨率": [int(ref.shape[1]), int(ref.shape[2])],
        "使用mask": mask is not None,
        "平均PSNR_dB": float(np.mean(finite_psnr)) if finite_psnr else float("inf"),
        "平均SSIM": float(np.mean([row["SSIM"] for row in rows])),
        "时序差分L1": temporal_error,
        "逐帧": rows,
        "未计算": ["LPIPS", "FVD"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["帧", "MSE", "PSNR_dB", "SSIM"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "逐帧"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
