from pathlib import Path
import os
import torch

from deform_transport.hard_transport import hard_point_transport
from deform_transport.transport_payloads import load_realwonder_rgb_crop
from deform_transport.transport_ready import validate_transport_ready
from deform_transport.wan_vae_codec import (
    RealWonderWanVAECodec,
    causal_latent_frame_end_indices,
)

tr_path = Path(os.environ["TR"])
ckpt = Path(os.environ["CKPT"])
out_path = Path(os.environ["OUT"]) / "vae_latent_outputs.pt"

torch.manual_seed(0)

state = torch.load(
    tr_path,
    map_location="cpu",
    weights_only=False,
)
validate_transport_ready(state)

assert state["case_name"] == "sand_house"
assert state["points_2d_latent"].shape[0] == 165
assert len(state["paths"]["coarse_rgb_frames"]) == 165

print("case =", state["case_name"])
print("pixel states =", state["points_2d_latent"].shape[0])
print("points =", state["points_2d_latent"].shape[1])
print("source visible =", int(state["source_visible"].sum()))

codec = RealWonderWanVAECodec(
    ckpt,
    device="cuda",
    dtype=torch.bfloat16,
)

source = load_realwonder_rgb_crop(
    state["paths"]["initial_rgb"]
)
source_pixels = (
    source.mul(2.0).sub(1.0)
    .unsqueeze(0)
    .unsqueeze(2)
)

source_latent = codec.encode_pixels(source_pixels)

frames = [
    load_realwonder_rgb_crop(p)
    for p in state["paths"]["coarse_rgb_frames"]
]
future_pixels = (
    torch.stack(frames)
    .permute(1, 0, 2, 3)
    .unsqueeze(0)
    .mul(2.0)
    .sub(1.0)
)

assert future_pixels.shape[2] == 165

target_latent = codec.encode_pixels(future_pixels)

latent_indices = causal_latent_frame_end_indices(
    future_pixels.shape[2]
).cpu()

assert latent_indices.tolist() == list(range(0, 165, 4))
assert tuple(source_latent.shape) == (1, 1, 16, 60, 104)
assert tuple(target_latent.shape) == (1, 42, 16, 60, 104)

print("source latent =", tuple(source_latent.shape))
print("target latent =", tuple(target_latent.shape))
print("latent indices =", latent_indices.tolist())

def run_hard(mode):
    return hard_point_transport(
        source_grid=source_latent[0, 0],
        source_uv=state["source_points_2d_latent"].cuda(),
        target_uv=state["points_2d_latent"][latent_indices].cuda(),
        source_visible=state["source_visible"].cuda(),
        source_valid=state["source_valid"].cuda(),
        target_valid=state["projection_valid"][latent_indices].cuda(),
        point_id=state["point_id"].cuda(),
        object_id=state["object_id"].cuda(),
        mode=mode,
        seed=0,
    )

correct = run_hard("correct")
shuffled = run_hard("shuffled")

assert torch.equal(
    correct["transport_mask"],
    shuffled["transport_mask"],
)
assert torch.equal(
    correct["contribution_count"],
    shuffled["contribution_count"],
)
assert torch.equal(
    correct["valid_point_mask"],
    shuffled["valid_point_mask"],
)
assert not torch.equal(
    correct["permutation"],
    shuffled["permutation"],
)

raw_correct = correct["transported_grid"].unsqueeze(0)
raw_shuffled = shuffled["transported_grid"].unsqueeze(0)
mask = correct["transport_mask"].unsqueeze(0)

fused_correct = torch.where(
    mask,
    raw_correct,
    target_latent,
)
fused_shuffled = torch.where(
    mask,
    raw_shuffled,
    target_latent,
)

artifact = {
    "format_version": 1,
    "artifact_kind": "sand_house_wan_vae_base_cache",
    "latent_frame_indices": latent_indices,
    "source_latent": source_latent.detach().cpu(),
    "target_latent": target_latent.detach().cpu(),
    "correct_transported_latent": raw_correct.detach().cpu(),
    "shuffled_transported_latent": raw_shuffled.detach().cpu(),
    "correct_fused_latent": fused_correct.detach().cpu(),
    "shuffled_fused_latent": fused_shuffled.detach().cpu(),
    "transport_mask": correct["transport_mask"].detach().cpu(),
    "contribution_count": correct["contribution_count"].detach().cpu(),
    "shuffled_permutation": shuffled["permutation"].detach().cpu(),
    "source_transport_ready": str(tr_path),
}

torch.save(artifact, out_path)

check = torch.load(
    out_path,
    map_location="cpu",
    weights_only=False,
)

assert tuple(check["target_latent"].shape) == (
    1, 42, 16, 60, 104
)
assert torch.equal(
    check["transport_mask"],
    check["contribution_count"] > 0,
)

print("support cells =", int(check["transport_mask"].sum()))
print("artifact =", out_path)
print("bytes =", out_path.stat().st_size)
print("SANDHOUSE_WAN_VAE_BASE_CACHE_OK")
