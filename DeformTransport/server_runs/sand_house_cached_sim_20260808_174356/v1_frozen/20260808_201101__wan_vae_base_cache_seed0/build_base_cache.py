from pathlib import Path
import os
import time

import torch

from deform_transport.hard_transport import hard_point_transport
from deform_transport.transport_payloads import load_realwonder_rgb_crop
from deform_transport.transport_ready import validate_transport_ready
from deform_transport.wan_vae_codec import (
    RealWonderWanVAECodec,
    causal_latent_frame_end_indices,
)


transport_path = Path(os.environ["TR"])
checkpoint = Path(os.environ["CKPT"])
output_dir = Path(os.environ["OUT"])
output_path = output_dir / "vae_latent_outputs.pt"

torch.manual_seed(0)

state = torch.load(
    transport_path,
    map_location="cpu",
    weights_only=False,
)

validate_transport_ready(state)

print("===== INPUT CONTRACT =====")
print("case =", state["case_name"])
print("trajectory frames =", state["points_2d_latent"].shape[0])
print("points =", state["points_2d_latent"].shape[1])
print("source_visible =", int(state["source_visible"].sum()))
print("source_valid =", int(state["source_valid"].sum()))
print("coarse frames =", len(state["paths"]["coarse_rgb_frames"]))

assert state["points_2d_latent"].shape[0] == 165
assert len(state["paths"]["coarse_rgb_frames"]) == 165


def pixel_video_from_paths(paths):
    frames = [
        load_realwonder_rgb_crop(path)
        for path in paths
    ]

    return (
        torch.stack(frames)
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
        .mul(2.0)
        .sub(1.0)
    )


print("\n===== LOAD WAN VAE =====")

started = time.perf_counter()

codec = RealWonderWanVAECodec(
    checkpoint,
    device="cuda",
    dtype=torch.bfloat16,
)

print(
    "model load seconds =",
    time.perf_counter() - started,
)


print("\n===== SOURCE ENCODE =====")

source_crop = load_realwonder_rgb_crop(
    state["paths"]["initial_rgb"]
)

source_pixels = (
    source_crop
    .mul(2.0)
    .sub(1.0)
    .unsqueeze(0)
    .unsqueeze(2)
)

started = time.perf_counter()

source_latent = codec.encode_pixels(
    source_pixels
)

torch.cuda.synchronize()

print("source latent =", tuple(source_latent.shape))
print(
    "source encode seconds =",
    time.perf_counter() - started,
)

assert tuple(source_latent.shape) == (
    1, 1, 16, 60, 104
)


print("\n===== FUTURE ENCODE =====")

future_pixels = pixel_video_from_paths(
    state["paths"]["coarse_rgb_frames"]
)

print(
    "future pixels =",
    tuple(future_pixels.shape)
)

assert future_pixels.shape[2] == 165

started = time.perf_counter()

target_latent = codec.encode_pixels(
    future_pixels
)

torch.cuda.synchronize()

latent_indices = causal_latent_frame_end_indices(
    future_pixels.shape[2]
)

print(
    "latent indices =",
    latent_indices.tolist()
)

print(
    "target latent =",
    tuple(target_latent.shape)
)

print(
    "future encode seconds =",
    time.perf_counter() - started,
)

assert latent_indices.tolist() == list(
    range(0, 165, 4)
)

assert tuple(target_latent.shape) == (
    1, 42, 16, 60, 104
)


print("\n===== HARD REFERENCE TRANSPORT =====")


def run_transport(mode):
    return hard_point_transport(
        source_latent[0, 0],
        state[
            "source_points_2d_latent"
        ].cuda(),
        state[
            "points_2d_latent"
        ][latent_indices].cuda(),
        state[
            "source_visible"
        ].cuda(),
        state[
            "source_valid"
        ].cuda(),
        state[
            "projection_valid"
        ][latent_indices].cuda(),
        state[
            "point_id"
        ].cuda(),
        object_id=state[
            "object_id"
        ].cuda(),
        mode=mode,
        seed=0,
    )


started = time.perf_counter()

correct = run_transport("correct")
shuffled = run_transport("shuffled")

torch.cuda.synchronize()

print(
    "transport seconds =",
    time.perf_counter() - started,
)


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

assert not torc
    correct["permutation
    shuff
)


raw_cor

    .unsquee


raw_shuffled = (
    shuffled["transported_grid"]
    .unsqueez
)

mask = correct[
    "transport_mask"
].unsqueeze(0)

fused_correct = 
    mask,
    raw_correct,
    target_latent,
)

fused_shuffled = tor
    mask,
    raw_shuffled,
    target_laten
)


output = {
    "format
    "artifact_kind":
        "sand_h




    "source_latent":
        source_lat

    "target_latent":
        target_

    "correct_transported_latent":
 

    "shuffled_transported_latent":
        raw_sh

    "correct_fused_l


    "shuffled_fuse
        fused_shuff

    "transport_mask":

            "transport_
 

    "contribution_count":
     

        ].detach().cpu(),

    "shu
        shuffled[
            "perm
        ].detach().cpu(),

    "source_transport_ready":
       
}


print("\n===== SAVE =====")

torch.save(
    outp
    output_path,
)

loaded = torch.load(
    output_path,
   
    weights_only=False,
)

assert tuple(
    loaded["targ
) == (1, 42, 16, 60, 104)

assert loaded[
    "latent_frame_indice


assert torch.equal(
    loaded["transport_mask"
    loaded["contribution_count"] > 0,
)

print(
    "s


            "transport_mask"
     
    ),
)

prin
    "artifact =",
    output_path,
)


    "bytes =",
    output
)

pr

)
