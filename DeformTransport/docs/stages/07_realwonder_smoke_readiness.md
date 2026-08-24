# Stage 7 readiness audit: full RealWonder smoke not executed

## Status

The checkpoint-free and Wan VAE-only gates pass, but the third mandatory gate
for full generation does not: the complete RealWonder and Wan conditioning
weights are not installed, and no separate authorization to download them was
given. Therefore no baseline, Correct, or Shuffled diffusion video was run.

This document is a readiness and resource report. It must not be cited as an
end-to-end experiment.

## Local model inventory

The only installed video-model weight is:

```text
wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth
507,609,880 bytes
SHA-256 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981
```

There is no local RealWonder distilled generator checkpoint, T5 weight, CLIP
weight, or complete Wan model repository.

## Minimum missing downloads

The exact RealWonder code paths and the official repository file listings imply
the following minimum missing set:

| File group | Exact bytes | Approx. GiB |
| --- | ---: | ---: |
| RealWonder `step=000800.pt` | 18,774,247,788 | 17.485 |
| Wan UMT5-XXL BF16 weight | 11,361,920,418 | 10.582 |
| Wan CLIP image-encoder weight | 4,772,359,047 | 4.445 |
| UMT5 tokenizer files | 21,454,081 | 0.020 |
| **Total still missing** | **34,929,981,334** | **32.531** |

The official Wan repository is 19,814,506,583 bytes in total. Its
3,128,957,992-byte `diffusion_pytorch_model.safetensors` is not part of the
minimum list above because RealWonder constructs its generator in code and
loads the separate distilled checkpoint. The repository README downloads the
entire Wan directory for convenience; a future deployment should confirm the
minimal include list on the target server before downloading.

At least 50 GB of free model/cache space should be reserved in addition to
experiment outputs. The local WSL filesystem currently has ample disk space,
but disk is not the limiting resource.

## Local and server feasibility

The 8 GB RTX 5060 is not a credible target for the unmodified RealWonder
pipeline:

- the distilled checkpoint file alone is 17.485 GiB;
- `infer_sim.py` moves the generator to CUDA;
- it also loads text, image, VAE, processor, cache, noise, and activation state;
- its low-memory path dynamically swaps the text encoder but still places the
  generator on the GPU;
- the official README reports testing the interactive system on an H200.

Recommended first server target:

- one 80 GB or larger CUDA GPU (A100 80GB, H100 80GB, or H200); H200 is the
  closest match to the public implementation;
- at least 64 GB system RAM, with 128 GB preferred because `torch.load` first
  materializes the large checkpoint on CPU;
- at least 60 GB free disk after environment installation;
- CUDA/PyTorch compatibility matching the server GPU rather than forcing the
  local Blackwell build.

A 48 GB GPU may be explored only as a low-memory engineering attempt; current
code evidence is insufficient to promise it will fit. Repeated 8 GB OOM trials
are not recommended.

## Prepared integration boundary

`deform_transport/pipeline_integration.py` now implements a validated loader for
the `correct_fused_latent` and `shuffled_fused_latent` tensors already produced
by the VAE-only probe. It requires an exact shape match with the freshly encoded
RealWonder `sim_latent`, checks finite floating-point values, validates that the
saved boolean mask equals `contribution_count > 0`, and converts to the existing
device and dtype.

`infer_sim.py` has two optional arguments:

```text
--transport_latent_path PATH
--transport_mode {correct,shuffled}
```

The optional replacement occurs immediately after RealWonder encodes the
coarse simulation frames and before the existing temporal trim/pad logic. With
no `--transport_latent_path`, the original baseline path is unchanged. This
adds no model channel, adapter, loss, denoising intervention, or training.

Three no-model integration tests pass:

1. Correct and Shuffled keys are selected exactly and converted to the
   reference dtype/device;
2. a different frame count, case, or resolution is rejected by exact shape
   validation;
3. NaN and inconsistent mask/count support are rejected before generation.

The real Santa VAE artifact also passes the loader contract for both modes with
shape `[1,6,16,60,104]`, BF16 output, finite values, and distinct tensors.

## Remaining input-bundle gap

The current saved Santa 21-frame probe contains trajectories, rendered frames,
flow, source raster IDs, camera data, and transport latent outputs. It does not
contain all files expected by offline `infer_sim.py`, notably:

- `noises.npy` produced by RealWonder's `NoiseWarper`;
- `points_masks_downsampled.pt` and `mesh_masks_downsampled.pt`;
- the exact offline `final_sim/config.yaml`, `prompt.txt`, and directory layout.

These files must be prepared from the same saved simulation frames/flow without
rerunning three separate Genesis simulations. The current transport and VAE
results must not be overwritten. This is an independent input-packaging step on
the server or after model-download authorization.

## Intended fixed-seed smoke matrix

Once the missing weights and a complete `final_sim` bundle exist, run exactly
one shared simulation condition and one seed:

1. baseline: original encoded `sim_latent`, no transport arguments;
2. Correct: the same command plus the VAE artifact and
   `--transport_mode correct`;
3. Shuffled: the same command plus the same artifact and
   `--transport_mode shuffled`.

The checkpoint, flow, frames, camera, masks, prompt, seed, denoising steps,
resolution, and output length must remain identical. Each version should be
run separately to control peak memory. The integration hook is prepared, but
no claim is made that the generator will accept, use, or improve from the
transported latent until these three executions actually finish.

## Explicitly deferred

- complete model download;
- dependency installation for full RealWonder generation;
- baseline/Correct/Shuffled video generation;
- any 8 GB OOM experiment;
- CamProbe-style denoising intervention;
- soft splatting, refiner, QWM, adapter, or training;
- robot-action or robot-cloth experiment.
