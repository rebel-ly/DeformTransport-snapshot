"""Minimal RealWonder Wan VAE-only codec without text or diffusion models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn as nn


# Importing ``wan.modules.vae`` normally executes ``wan/__init__.py``, which
# eagerly imports the diffusion model and requires unrelated packages. Loading
# this standalone module file preserves RealWonder's exact VAE implementation
# while keeping the VAE-only stage independent of diffusers, PEFT, T5, and CLIP.
_VAE_SOURCE = Path(__file__).resolve().parents[1] / "wan" / "modules" / "vae.py"
_VAE_SPEC = importlib.util.spec_from_file_location(
    "deform_transport_realwonder_wan_vae", _VAE_SOURCE
)
if _VAE_SPEC is None or _VAE_SPEC.loader is None:
    raise ImportError(f"cannot load RealWonder Wan VAE source from {_VAE_SOURCE}")
_VAE_MODULE = importlib.util.module_from_spec(_VAE_SPEC)
sys.modules[_VAE_SPEC.name] = _VAE_MODULE
_VAE_SPEC.loader.exec_module(_VAE_MODULE)
_video_vae = _VAE_MODULE._video_vae


WAN_VAE_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN_VAE_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


def causal_latent_frame_end_indices(pixel_frame_count: int) -> torch.Tensor:
    """Map each causal VAE latent slot to the end of its pixel-frame chunk.

    Wan encodes frame zero alone, then consecutive four-frame chunks. For 21
    pixel frames this produces six latent slots aligned to `[0,4,8,12,16,20]`.
    """

    if pixel_frame_count < 1:
        raise ValueError("pixel_frame_count must be at least one")
    return torch.tensor(
        [0, *range(4, pixel_frame_count, 4)], dtype=torch.long
    )


class RealWonderWanVAECodec(nn.Module):
    """The exact 16-channel Wan2.1 VAE and normalization used by RealWonder."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.compute_device = torch.device(device)
        self.compute_dtype = dtype
        self.model = _video_vae(
            pretrained_path=str(self.checkpoint_path), z_dim=16, device="cpu"
        ).eval().requires_grad_(False)
        self.model.to(device=self.compute_device, dtype=self.compute_dtype)
        self.register_buffer(
            "latent_mean", torch.tensor(WAN_VAE_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "latent_inverse_std",
            1.0 / torch.tensor(WAN_VAE_STD, dtype=torch.float32),
        )

    def _scale(self, dtype: torch.dtype) -> list[torch.Tensor]:
        return [
            self.latent_mean.to(device=self.compute_device, dtype=dtype),
            self.latent_inverse_std.to(device=self.compute_device, dtype=dtype),
        ]

    @torch.inference_mode()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode `[B,C,T,H,W]` pixels in `[-1,1]` to `[B,Tz,16,h,w]`."""

        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 5 or pixels.shape[1] != 3:
            raise ValueError("pixels must have shape [B, 3, T, H, W]")
        pixels = pixels.to(device=self.compute_device, dtype=self.compute_dtype)
        scale = self._scale(pixels.dtype)
        outputs = [
            self.model.encode(sample.unsqueeze(0), scale).float().squeeze(0)
            for sample in pixels
        ]
        latent = torch.stack(outputs).permute(0, 2, 1, 3, 4).contiguous()
        expected_slots = (pixels.shape[2] + 3) // 4
        if latent.shape[1] != expected_slots:
            raise RuntimeError(
                f"unexpected temporal compression: got {latent.shape[1]}, "
                f"expected {expected_slots}"
            )
        return latent

    @torch.inference_mode()
    def decode_latents(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode `[B,Tz,16,h,w]` to `[B,T,C,H,W]` pixels in `[-1,1]`."""

        latent = torch.as_tensor(latent)
        if latent.ndim != 5 or latent.shape[2] != 16:
            raise ValueError("latent must have shape [B, Tz, 16, H, W]")
        latent = latent.to(device=self.compute_device, dtype=self.compute_dtype)
        latent_channel_first = latent.permute(0, 2, 1, 3, 4).contiguous()
        scale = self._scale(latent.dtype)
        outputs = [
            self.model.decode(sample.unsqueeze(0), scale)
            .float()
            .clamp_(-1, 1)
            .squeeze(0)
            for sample in latent_channel_first
        ]
        return torch.stack(outputs).permute(0, 2, 1, 3, 4).contiguous()

    def clear_cache(self) -> None:
        self.model.clear_cache()
