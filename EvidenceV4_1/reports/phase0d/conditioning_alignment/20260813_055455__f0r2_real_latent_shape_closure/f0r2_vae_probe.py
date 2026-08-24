import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image
from wan.modules.vae import WanVAE

source = "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"
checkpoint = "/workspace/Wan-Move/Wan-Move-14B-480P/Wan2.1_VAE.pth"
image = Image.open(source).convert("RGB")
rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
source_tensor = torch.cat([rgb[:, None], torch.zeros((3, 80, 480, 832), dtype=rgb.dtype)], dim=1)
print(json.dumps({
    "event": "pre_encode",
    "source_tensor_shape": list(source_tensor.shape),
    "source_tensor_dtype": str(source_tensor.dtype),
    "source_sha256": hashlib.sha256(open(source, "rb").read()).hexdigest(),
    "checkpoint_sha256": hashlib.sha256(open(checkpoint, "rb").read()).hexdigest(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}), flush=True)
vae = WanVAE(vae_pth=checkpoint, device="cuda")
print(json.dumps({"event": "vae_instantiated"}), flush=True)
y = vae.encode([source_tensor.cuda()])[0]
print(json.dumps({
    "event": "encoded",
    "raw_vae_encode_return_type": type(y).__name__,
    "raw_vae_latent_shape": list(y.shape),
    "raw_vae_latent_dtype": str(y.dtype),
    "raw_vae_latent_device": str(y.device),
}), flush=True)
print(json.dumps({
    "event": "transport_input",
    "real_transport_y_shape": [1] + list(y.shape),
    "real_transport_y_dtype": str(y.dtype),
    "real_transport_latent_hw": list(y.shape[-2:]),
}), flush=True)
decoded = vae.decode([torch.zeros_like(y)])[0]
print(json.dumps({"event": "zero_latent_decoded", "vae_decode_raw_shape": list(decoded.shape)}), flush=True)
