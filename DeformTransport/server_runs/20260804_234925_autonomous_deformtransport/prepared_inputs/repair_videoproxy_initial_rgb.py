"""生成保留 v1/v2 失败现场的 video-proxy v3 transport_ready 资产。"""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT = Path("/workspace/DeformTransport")
ASSET_DIR = ROOT / "server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/santa_21f_videoproxy_transport_ready"
SOURCE_STATE = ASSET_DIR / "transport_ready_videoproxy.pt"
OUTPUT_STATE = ASSET_DIR / "transport_ready_videoproxy_v3.pt"
SOURCE_INITIAL = ROOT / "artifacts/stage1_dynamic/santa_cloth_2f_wsl_20260802_retry3/frame_initial.png"
OUTPUT_INITIAL = ASSET_DIR / "initial_rgb_512.png"
OUTPUT_FRAMES = ASSET_DIR / "frames_512"
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def to_square_proxy(cropped: Image.Image) -> Image.Image:
    if cropped.size != (832, 480):
        raise ValueError(f"视频代理帧应为 832x480，实际为 {cropped.size}")
    canvas = Image.new("RGB", (832, 832))
    canvas.paste(cropped, (0, 176))
    top = cropped.crop((0, 0, 832, 1)).resize((832, 176))
    bottom = cropped.crop((0, 479, 832, 480)).resize((832, 176))
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, 656))
    return canvas.resize((512, 512), resample=Image.BILINEAR)
def official_crop(square: Image.Image) -> Image.Image:
    return square.resize((832, 832), resample=Image.BILINEAR).crop((0, 176, 832, 656))
def main() -> None:
    if Image.open(SOURCE_INITIAL).size != (512, 512):
        raise ValueError("历史初始帧不是 RealWonder 所需的 512x512")
    shutil.copyfile(SOURCE_INITIAL, OUTPUT_INITIAL)
    state = torch.load(SOURCE_STATE, map_location="cpu")
    OUTPUT_FRAMES.mkdir(parents=True, exist_ok=True)
    output_paths = []
    maes = []
    maxima = []
    for index, source_path in enumerate(state["paths"]["coarse_rgb_frames"]):
        source = Image.open(source_path).convert("RGB")
        square = to_square_proxy(source)
        output = OUTPUT_FRAMES / f"frame_{index:04d}.png"
        square.save(output)
        recovered = np.asarray(official_crop(square), dtype=np.int16)
        original = np.asarray(source, dtype=np.int16)
        delta = np.abs(recovered - original)
        maes.append(float(delta.mean()))
        maxima.append(int(delta.max()))
        output_paths.append(str(output))
    state["paths"] = dict(state["paths"])
    state["paths"]["initial_rgb"] = str(OUTPUT_INITIAL)
    state["paths"]["coarse_rgb_frames"] = output_paths
    torch.save(state, OUTPUT_STATE)
    provenance = {
        "用途": "真实 Wan VAE 编码→transport→解码 GPU smoke；不作为公平生成或 GT 评估",
        "版本": "v3",
        "修复原因": "v1 的 initial_rgb 和 coarse_rgb_frames 都已是 832x480，官方加载器要求 512x512；v2 仅修复 initial_rgb，仍不完整",
        "初始帧来源": str(SOURCE_INITIAL),
        "初始帧SHA256": sha256(OUTPUT_INITIAL),
        "历史源图像素核验": "官方预处理后与历史 source_original_reconstruction 左面板逐像素一致（MAE=0，最大绝对差=0）",
        "未来帧转换": "832x480 视频代理帧嵌入 832x832（上下边缘复制），双线性缩放至 512x512；运行时再走官方 resize+crop",
        "未来帧往返MAE均值": float(np.mean(maes)),
        "未来帧往返MAE最大": float(np.max(maes)),
        "未来帧往返最大绝对差": int(np.max(maxima)),
        "未来帧数": len(output_paths),
        "v1_transport_ready": str(SOURCE_STATE),
        "v1_SHA256": sha256(SOURCE_STATE),
        "v3_transport_ready": str(OUTPUT_STATE),
        "v3_SHA256": sha256(OUTPUT_STATE),
        "重要限制": "未来帧源自历史 target_input.mp4，且增加一次尺寸往返；属于有损 proxy，仅验证真实 GPU 闭环工程可行性",
    }
    (ASSET_DIR / "provenance_v3.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
