"""使用真实 Wan VAE 对现有 transport latent 进行 GPU 端到端解码验证。"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.wan_vae_codec import RealWonderWanVAECodec

EXPECTED_CHECKPOINT_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
EXPECTED_ARTIFACT_SHA256 = "f2ad921d3548d1d22acbee4d36896e0c16f9ff576f165dbef3d27b387613345c"
LATENT_KEYS = ("target_latent", "correct_fused_latent", "shuffled_fused_latent")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "mean": float(value.mean()), "std": float(value.std()),
            "min": float(value.min()), "max": float(value.max()),
            "finite": bool(torch.isfinite(value).all())}


def write_video(path: Path, decoded: torch.Tensor) -> None:
    frames = (decoded[0].mul(0.5).add(0.5).clamp(0, 1)
              .permute(0, 2, 3, 1).mul(255).round().to(torch.uint8).numpy())
    writer = imageio.get_writer(path, fps=8, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = sha256(args.checkpoint)
    artifact_hash = sha256(args.artifact)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA256 不一致：{checkpoint_hash}")
    if artifact_hash != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError(f"artifact SHA256 不一致：{artifact_hash}")

    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    for key in LATENT_KEYS:
        value = payload[key]
        if list(value.shape) != [1, 6, 16, 60, 104]:
            raise RuntimeError(f"{key} shape 异常：{list(value.shape)}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{key} 含非有限值")
    if torch.equal(payload["correct_fused_latent"], payload["shuffled_fused_latent"]):
        raise RuntimeError("Correct 与 Shuffled latent 完全相同")
    mask = payload["transport_mask"]
    count = payload["contribution_count"]
    if not torch.equal(mask, count > 0):
        raise RuntimeError("transport_mask 与 contribution_count 不一致")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device_name = torch.cuda.get_device_name(0)
    device_uuid = str(torch.cuda.get_device_properties(0).uuid)
    started = time.perf_counter()
    load_started = time.perf_counter()
    codec = RealWonderWanVAECodec(args.checkpoint, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    decoded = {}
    decode_seconds = {}
    for key in LATENT_KEYS:
        stage_started = time.perf_counter()
        value = codec.decode_latents(payload[key])
        torch.cuda.synchronize()
        decode_seconds[key] = time.perf_counter() - stage_started
        value = value.cpu()
        if list(value.shape) != [1, 21, 3, 480, 832]:
            raise RuntimeError(f"{key} 解码 shape 异常：{list(value.shape)}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{key} 解码结果含非有限值")
        decoded[key] = value
        write_video(args.output_dir / f"{key}.mp4", value)
        codec.clear_cache()
        torch.cuda.empty_cache()

    correct_delta = (decoded["correct_fused_latent"] - decoded["target_latent"]).abs()
    shuffled_delta = (decoded["shuffled_fused_latent"] - decoded["target_latent"]).abs()
    report = {
        "任务": "真实 Wan VAE + transport artifact GPU 端到端 smoke",
        "状态": "通过", "seed": args.seed,
        "GPU": {"名称": device_name, "UUID": device_uuid},
        "checkpoint": {"路径": str(args.checkpoint.resolve()), "SHA256": checkpoint_hash},
        "artifact": {"路径": str(args.artifact.resolve()), "SHA256": artifact_hash},
        "latent统计": {key: stats(payload[key]) for key in LATENT_KEYS},
        "解码统计": {key: stats(decoded[key]) for key in LATENT_KEYS},
        "差异": {
            "Correct相对Target平均绝对差": float(correct_delta.mean()),
            "Shuffled相对Target平均绝对差": float(shuffled_delta.mean()),
            "Correct与Shuffled平均绝对差": float((decoded["correct_fused_latent"] - decoded["shuffled_fused_latent"]).abs().mean())},
        "transport覆盖": {"latent单元数": int(mask.sum()), "总单元数": int(mask.numel()), "覆盖率": float(mask.float().mean())},
        "运行秒数": {"模型加载": model_load_seconds, "逐项解码": decode_seconds, "总计": time.perf_counter() - started},
        "显存MiB": {"峰值已分配": float(torch.cuda.max_memory_allocated() / 1024**2), "峰值已保留": float(torch.cuda.max_memory_reserved() / 1024**2)}}
    (args.output_dir / "结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
