from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path("/workspace/DeformTransport")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deform_transport.transport_payloads import (
    load_realwonder_rgb_crop,
)


tree_long = Path(sys.argv[1]).resolve()
runtime_config_path = Path(sys.argv[2]).resolve()
aligned_contract_dir = Path(sys.argv[3]).resolve()
output = Path(sys.argv[4]).resolve()

frames_dir = output / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def preprocess_to_uint8(path: Path) -> np.ndarray:
    tensor = load_realwonder_rgb_crop(path)

    if tuple(tensor.shape) != (3, 480, 832):
        raise ValueError(
            f"unexpected processed shape for {path}: "
            f"{tuple(tensor.shape)}"
        )

    array = (
        tensor
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

    return array


print("========== INPUT CONTRACT ==========")

initial_source = tree_long / "frame_initial.png"

future_sources = [
    tree_long / f"frame_future_{i:04d}.png"
    for i in range(80)
]

source_paths = [
    initial_source
] + future_sources

if len(source_paths) != 81:
    raise RuntimeError("internal source frame count error")

missing = [
    str(p)
    for p in source_paths
    if not p.is_file()
]

if missing:
    raise FileNotFoundError(
        f"missing source frames: {missing[:10]}"
    )

aligned_transport_path = (
    aligned_contract_dir
    / "aligned_transport_ready.pt"
)

aligned_visibility_path = (
    aligned_contract_dir
    / "aligned_visibility_contract.pt"
)

for p in (
    runtime_config_path,
    aligned_transport_path,
    aligned_visibility_path,
):
    if not p.is_file():
        raise FileNotFoundError(p)

print("source_count =", len(source_paths))
print("source_0 =", source_paths[0])
print("source_1 =", source_paths[1])
print("source_80 =", source_paths[-1])


print()
print("========== CONFIG ==========")

runtime_config = OmegaConf.load(
    runtime_config_path
)

runtime_dict = OmegaConf.to_container(
    runtime_config,
    resolve=True,
)

config = OmegaConf.create(
    runtime_dict
)

config.output_folder = str(output)

OmegaConf.save(
    config,
    output / "config.yaml",
)

saved_config = OmegaConf.load(
    output / "config.yaml"
)

# Frozen official Tree dynamics / RealWonder contract.
assert saved_config.example_name == "tree"
assert float(saved_config.dt) == 0.01
assert int(saved_config.substeps) == 20
assert int(saved_config.frame_steps) == 2
assert int(saved_config.MPM_grid_density) == 32
assert float(saved_config.particle_size) == 0.02
assert list(saved_config.material_type) == ["mpm_elastic"]
assert int(saved_config.simulated_frames_num) == 81
assert int(saved_config.num_output_frames) == 21
assert list(saved_config.denoising_step_list) == [
    750,
    500,
    250,
]
assert int(saved_config.mask_dropin_step) == -1
assert int(saved_config.crop_start) == 176

prompt = str(
    saved_config.vgen_prompt
).strip()

(output / "prompt.txt").write_text(
    prompt + "\n",
    encoding="utf-8",
)

print(
    "denoising_step_list =",
    list(saved_config.denoising_step_list),
)
print(
    "frame_steps =",
    int(saved_config.frame_steps),
)
print(
    "substeps =",
    int(saved_config.substeps),
)
print(
    "MPM_grid_density =",
    int(saved_config.MPM_grid_density),
)


print()
print("========== EXACT RGB PREPROCESSING ==========")

frame_records = []
processed_arrays = []

for index, source_path in enumerate(
    source_paths
):
    with Image.open(source_path) as raw:
        raw_size = list(raw.size)

    if raw_size != [512, 512]:
        raise ValueError(
            f"{source_path} is not 512x512: "
            f"{raw_size}"
        )

    array = preprocess_to_uint8(
        source_path
    )

    target = (
        frames_dir
        / f"frame_{index:04d}.png"
    )

    Image.fromarray(
        array,
        mode="RGB",
    ).save(target)

    processed_arrays.append(array)

    frame_records.append(
        {
            "index": index,
            "source_path":
                str(source_path),
            "raw_size":
                raw_size,
            "processed_shape":
                list(array.shape),
            "target_path":
                str(target),
            "target_sha256":
                sha256(target),
        }
    )

    if index in (
        0,
        1,
        40,
        80,
    ):
        print(
            f"frame={index:04d}",
            f"source={source_path.name}",
            f"shape={array.shape}",
        )


print()
print("========== FIRST FRAME ==========")

frame0 = (
    frames_dir
    / "frame_0000.png"
)

resized_input = (
    output
    / "resized_input_image.png"
)

# Exact byte identity, not merely pixel identity.
shutil.copy2(
    frame0,
    resized_input,
)

assert (
    sha256(frame0)
    ==
    sha256(resized_input)
)

print(
    "frame0_sha256 =",
    sha256(frame0),
)

print(
    "resized_input_sha256 =",
    sha256(resized_input),
)


print()
print("========== SIMULATION MP4 ==========")

video_path = (
    output
    / "simulation.mp4"
)

with imageio.get_writer(
    str(video_path),
    fps=10,
    codec="libx264",
    pixelformat="yuv420p",
    output_params=[
        "-crf",
        "18",
        "-preset",
        "medium",
    ],
) as writer:

    for array in processed_arrays:
        writer.append_data(array)

reader = imageio.get_reader(
    str(video_path)
)

decoded_frames = sum(
    1 for _ in reader
)

reader.close()

if decoded_frames != 81:
    raise RuntimeError(
        "simulation.mp4 frame count mismatch: "
        f"{decoded_frames}"
    )

print(
    "simulation_mp4_decoded_frames =",
    decoded_frames,
)


print()
print("========== STRONG CHECKS ==========")

checks = {}

checks[
    "frame_count_is_81"
] = (
    len(frame_records) == 81
)

checks[
    "all_processed_480x832"
] = all(
    record["processed_shape"]
    == [480, 832, 3]
    for record in frame_records
)

checks[
    "frame0_source_is_initial"
] = (
    Path(
        frame_records[0][
            "source_path"
        ]
    ).resolve()
    == initial_source.resolve()
)

checks[
    "frame1_source_is_S2"
] = (
    Path(
        frame_records[1][
            "source_path"
        ]
    ).resolve()
    == (
        tree_long
        / "frame_future_0000.png"
    ).resolve()
)

checks[
    "frame80_source_is_S160"
] = (
    Path(
        frame_records[-1][
            "source_path"
        ]
    ).resolve()
    == (
        tree_long
        / "frame_future_0079.png"
    ).resolve()
)

checks[
    "resized_input_equals_frame0"
] = (
    sha256(resized_input)
    ==
    sha256(frame0)
)

checks[
    "config_exists"
] = (
    output
    / "config.yaml"
).is_file()

checks[
    "prompt_exists"
] = (
    output
    / "prompt.txt"
).is_file()

checks[
    "simulation_video_exists"
] = (
    video_path.is_file()
    and
    video_path.stat().st_size > 0
)

checks[
    "simulation_video_has_81_frames"
] = (
    decoded_frames == 81
)

# Must remain absent until the dedicated
# RealWonder RAFT/noise stage.
checks[
    "no_stale_noise"
] = not (
    output
    / "noises.npy"
).exists()

checks[
    "no_stale_raft_flow"
] = not (
    output
    / "flows.npy"
).exists()

checks[
    "official_tree_dynamics_preserved"
] = (
    int(saved_config.substeps) == 20
    and
    int(saved_config.frame_steps) == 2
    and
    int(saved_config.MPM_grid_density)
    == 32
    and
    list(saved_config.material_type)
    == ["mpm_elastic"]
)

checks[
    "tree_realwonder_schedule_preserved"
] = (
    int(saved_config.num_output_frames)
    == 21
    and
    list(
        saved_config.denoising_step_list
    )
    == [750, 500, 250]
)

all_checks_pass = all(
    checks.values()
)

for key, value in checks.items():
    print(
        key,
        "=",
        value,
    )

print(
    "all_checks_pass =",
    all_checks_pass,
)

if not all_checks_pass:
    raise RuntimeError(
        "Tree aligned final_sim "
        "validation failed"
    )


print()
print("========== REPORT ==========")

report = {
    "stage":
        "tree_aligned_final_sim_exact_realwonder_preprocessing_build",

    "experiment_definition":
        (
            "Official precomputed Tree geometry "
            "+ official Tree dynamics"
        ),

    "output":
        str(output),

    "aligned_contract": {
        "transport_path":
            str(aligned_transport_path),

        "transport_sha256":
            sha256(
                aligned_transport_path
            ),

        "visibility_path":
            str(aligned_visibility_path),

        "visibility_sha256":
            sha256(
                aligned_visibility_path
            ),
    },

    "runtime_config": {
        "source":
            str(runtime_config_path),

        "source_sha256":
            sha256(
                runtime_config_path
            ),

        "saved":
            str(
                output
                / "config.yaml"
            ),

        "saved_sha256":
            sha256(
                output
                / "config.yaml"
            ),

        "only_intended_change":
            "output_folder",
    },

    "preprocessing_function":
        (
            "deform_transport.transport_payloads."
            "load_realwonder_rgb_crop"
        ),

    "mapping_contract":
        (
            "frame_0000=S0; "
            "frame_0001..0080="
            "S2,S4,...,S160"
        ),

    "frame_count":
        81,

    "latent_pixel_indices":
        list(range(0, 81, 4)),

    "latent_physical_states":
        list(range(0, 161, 8)),

    "frames":
        frame_records,

    "simulation_mp4": {
        "path":
            str(video_path),

        "sha256":
            sha256(video_path),

        "decoded_frames":
            decoded_frames,

        "fps":
            10,
    },

    "checks":
        checks,

    "all_checks_pass":
        all_checks_pass,
}

report_path = (
    output
    / "aligned_build_report.json"
)

report_path.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print(
    json.dumps(
        {
            "output":
                str(output),

            "frame_count":
                81,

            "raw_frame0_size":
                frame_records[0][
                    "raw_size"
                ],

            "processed_frame0_shape":
                frame_records[0][
                    "processed_shape"
                ],

            "frame0_sha256":
                frame_records[0][
                    "target_sha256"
                ],

            "resized_input_sha256":
                sha256(
                    resized_input
                ),

            "simulation_mp4_frames":
                decoded_frames,

            "checks":
                checks,

            "all_checks_pass":
                all_checks_pass,
        },
        ensure_ascii=False,
        indent=2,
    )
)

print()
print(
    "TREE_ALIGNED_FINAL_SIM_BUILD_OK"
)
