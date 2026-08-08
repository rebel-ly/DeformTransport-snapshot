#!/usr/bin/env python
"""把无损 RealWonder trajectory probe 输出组装成 infer_sim final_sim。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
# import torch
from omegaconf import OmegaConf
from PIL import Image
# from torchvision.io import write_video
import imageio.v2 as imageio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_and_crop(image: Image.Image, crop_start: int) -> Image.Image:
    resized = image.convert("RGB").resize((832, 832), Image.Resampling.BILINEAR)
    return resized.crop((0, crop_start, 832, crop_start + 480))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--demo-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    demo_data = args.demo_data.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    output_frames = output / "frames"
    output_frames.mkdir()

    config = OmegaConf.to_container(
        OmegaConf.load(demo_data / "config.yaml"), resolve=True
    )
    crop_start = int(config.get("crop_start", 176))
    expected = (int(config["num_output_frames"]) - 1) * 4 + 1
    raw_frames = sorted(source.glob("frame_[0-9][0-9][0-9][0-9].png"))
    if len(raw_frames) != expected:
        raise ValueError(f"物理帧数应为 {expected}，实际为 {len(raw_frames)}")

    video_frames = []
    frame_hashes = {}
    for index, raw_path in enumerate(raw_frames):
        with Image.open(raw_path) as raw:
            frame = resize_and_crop(raw, crop_start)
        target = output_frames / f"frame_{index:04d}.png"
        frame.save(target)
        # video_frames.append(torch.from_numpy(np.asarray(frame).copy()))
        video_frames.append(np.asarray(frame).copy())
        frame_hashes[target.name] = sha256(target)

    initial_source = source / "frame_initial.png"
    with Image.open(initial_source) as raw_initial:
        initial = resize_and_crop(raw_initial, crop_start)
    initial_target = output / "resized_input_image.png"
    initial.save(initial_target)

    config["output_folder"] = str(output)
    config["seed"] = args.seed
    OmegaConf.save(OmegaConf.create(config), output / "config.yaml")
    (output / "prompt.txt").write_text(
        str(config["vgen_prompt"]).strip() + "\n", encoding="utf-8"
    )
    for name in ("point_trajectories.pt", "flows.npy", "flow_source_point_indices.npy"):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        target_name = "flows_geometry.npy" if name == "flows.npy" else name
        shutil.copy2(path, output / target_name)
    # video = torch.stack(video_frames, dim=0).to(torch.uint8)
    # write_video(str(output / "simulation.mp4"), video, fps=10, video_codec="libx264")
    video_path = output / "simulation.mp4"

    with imageio.get_writer(
        str(video_path),
        fps=10,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", "18", "-preset", "medium"],
    ) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    reader = imageio.get_reader(str(video_path))
    decoded_frames = sum(1 for _ in reader)
    reader.close()

    if decoded_frames != len(video_frames):
        raise RuntimeError(
            f"simulation.mp4帧数校验失败："
            f"期望{len(video_frames)}，实际{decoded_frames}"
        )

    report = {
        "任务": "从RealWonder无损trajectory probe组装final_sim",
        "生成时间UTC": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source),
        "demo_data": str(demo_data),
        "output_dir": str(output),
        "pixel帧数": len(raw_frames),
        "latent帧数": int(config["num_output_frames"]),
        "crop契约": f"512x512 PIL bilinear resize至832x832，垂直裁剪[{crop_start}:{crop_start + 480}]",
        "initial_sha256": sha256(initial_target),
        "config_sha256": sha256(output / "config.yaml"),
        "prompt_sha256": sha256(output / "prompt.txt"),
        "trajectory_sha256": sha256(output / "point_trajectories.pt"),
        "geometry_flow_sha256": sha256(output / "flows_geometry.npy"),
        "frame_sha256": frame_hashes,
        "noise状态": "待独立RAFT/noise步骤生成noises.npy",
        "simulation_mp4_sha256": sha256(video_path),
        "simulation_mp4_decoded_frames": decoded_frames,
        "simulation_mp4_backend": "imageio-ffmpeg/libx264",
    }
    (output / "assembly_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
