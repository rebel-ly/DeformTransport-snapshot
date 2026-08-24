# Stage 6: Wan VAE-only point-latent transport

## Verdict

The gated Wan VAE-only probe passes its engineering, quantitative-proxy, and
visual checks on the saved 21-frame Santa cloth sequence. Correct persistent
point identity is better than an object-internal shuffled identity assignment
at all six causal latent slots and all 21 decoded frames under identical
transport support and contribution counts.

This is the first direct positive result using RealWonder's actual 16-channel
Wan VAE latent, rather than RGB or coordinate payloads. It does not show that a
diffusion video generator will use the condition successfully, and the decoded
transport is visibly too blurred and semi-transparent to be a final video.

## Gate and scope

This stage was entered only after the checkpoint-free operator passed. It uses:

- the existing standardized Santa trajectory artifact;
- RealWonder's exact `wan/modules/vae.py` implementation;
- the exact Wan normalization constants copied from `vidgen/models.py`;
- one official 507,609,880-byte VAE checkpoint;
- BF16 CUDA inference on the RTX 5060;
- the already-tested nearest-cell read and `scatter_add_` plus count-average
  transport operator.

No diffusion transformer, RealWonder distilled video checkpoint, T5, CLIP,
SAM, FLUX, FlashAttention, QWM, training, or robot asset was downloaded or
loaded. Importing `wan.modules.vae` through the package would eagerly import
unrelated diffusion dependencies, so `deform_transport/wan_vae_codec.py` loads
the same VAE module file directly. This changes only package import behavior,
not the VAE architecture or weights.

## Checkpoint provenance

The single checkpoint is:

```text
wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth
```

- repository path: `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth`;
- file size: 507,609,880 bytes;
- SHA-256:
  `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`;
- VAE parameter count: 126,892,531.

The probe verifies the checksum before loading.

## Verified codec contract

The source baseline uses the real 480x832 Santa input in RealWonder's `[-1,1]`
pixel range.

```text
input:   [B,C,T,H,W] = [1,3,1,480,832]
latent:  [B,Tz,C,H,W] = [1,1,16,60,104]
decoded: [B,T,C,H,W] = [1,1,3,480,832]
```

All tensors are finite. The source encode/decode reconstruction has mean
absolute error 0.00588 in the `[0,1]` pixel range. Visual inspection confirms
that the collar, belt, buttons, trim, garment boundary, and brick background
are reconstructed correctly.

For the 21-frame sequence, the causal VAE produces six latent slots. The
verified trajectory samples corresponding to those slots are:

```text
pixel frame indices: [0, 4, 8, 12, 16, 20]
```

Frame zero is encoded alone; every subsequent latent slot represents a causal
four-frame chunk. The complete future latent has shape `[1,6,16,60,104]`, and
decoding it returns exactly 21 pixel frames.

## Transport and matched control

The initial RGB frame is encoded once. Every raster-visible source point reads
one fixed 16-channel source feature from its nearest 60x104 cell. Correct mode
carries that feature along the same persistent point ID to each of the six
future latent positions. Shuffled mode applies a seeded permutation only among
eligible points in the same object before carrying the features.

For both modes, the following are exactly equal:

- point trajectories and future positions;
- source-visible and target-valid point sets;
- transport masks;
- contribution counts;
- valid-point masks;
- source feature multiset.

Only the point-to-source-feature identity differs. Cells outside the transport
mask retain the encoded RealWonder coarse-RGB latent; cells inside the mask are
replaced by the transported latent with fixed alpha 1.0. This is a minimal
masked composition test, not a learned fusion module.

## Quantitative proxy results

The comparison target is the saved RealWonder coarse RGB sequence or its VAE
encoding. Metrics are measured only on the shared transport and target-point
support. This target is a geometry-aligned proxy, not real future-video ground
truth.

### Six latent slots

| Metric | Correct identity | Shuffled identity |
| --- | ---: | ---: |
| mean masked L1 | 0.35778 | 0.52394 |
| slots with lower Correct L1 | 6/6 | - |

Correct reduces mean latent L1 by 0.16615, or 31.71% relative to Shuffled.

### Twenty-one decoded frames after masked replacement

| Metric | Correct identity | Shuffled identity |
| --- | ---: | ---: |
| mean masked L1 | 0.12155 | 0.23446 |
| mean masked MSE | 0.02292 | 0.08125 |
| mean masked PSNR | 16.58 dB | 10.90 dB |
| frames with lower Correct L1 | 21/21 | - |

Correct reduces mean decoded L1 by 0.11291, or 48.16% relative to Shuffled.
Raw transported-latent decoding, without coarse-latent background replacement,
also favors Correct in 21/21 frames.

The unmodified future coarse-RGB VAE encode/decode baseline has full-frame mean
L1 0.00600 and PSNR 39.14 dB. It is a reconstruction ceiling, not a competing
future-prediction method.

## Visual inspection

The first, middle, and final comparisons were inspected. Correct transport
preserves recognizable white collar, black belt, white front trim, cuffs, and
trouser trim substantially better than Shuffled. Shuffled transport largely
becomes a red-white mixed foreground on the identical support.

Correct transport is nevertheless blurred and partly transparent in later
frames. Hard cell assignment, many-to-one averaging, sparse spatial support,
causal temporal compression, and direct replacement of an encoded latent are
all plausible contributors. The current evidence does not identify which one
dominates, so no soft splatting, visibility model, or denoising intervention is
added at this stage.

## Runtime and memory

On the RTX 5060 using Torch 2.12.0+cu130:

- VAE model load: about 0.48 s;
- source encode/decode: about 0.40/0.30 s;
- 21-frame encode: about 2.35 s;
- each 21-frame VAE decode: about 4.06 s;
- complete five-decode probe and artifact export: about 28.9 s;
- peak Torch CUDA allocation: 4,320.4 MiB;
- peak Torch CUDA reservation: 6,190 MiB.

The probe fits the local 8 GB GPU. The full diffusion video model is not
expected to fit this budget and has not been attempted.

## Tests and outputs

Three VAE contract tests cover the causal frame mapping for one and 21 frames
and reject an invalid zero-frame input. Together with the previous repository
tests, the regression suite passes 22/22.

Output root:

```text
artifacts/transport_validation/santa_cloth_21f/wan_vae
```

Important files:

- `baseline_report.json` and `source_original_reconstruction.png`;
- `report.json` with metrics, checks, timings, and memory;
- `vae_latent_outputs.pt` with source, target, Correct, Shuffled, fused latent,
  mask, counts, time indices, and permutation;
- Correct, Shuffled, raw, fused, target, and original-reconstruction videos;
- `original_correct_shuffled_target.mp4`;
- `comparison_first.png`, `comparison_mid.png`, and `comparison_final.png`.

## Commands

```bash
source /home/a/miniforge3/etc/profile.d/conda.sh
conda activate deformtransport-stage1
cd /home/a/DeformTransport

python -m unittest discover -s tests -v
python scripts/run_wan_vae_transport_probe.py --baseline-only
python scripts/run_wan_vae_transport_probe.py \
  --seed 0 \
  --visual-inspection pass \
  --visual-note "Correct retains garment parts better; later frames remain blurred."
```

## Scientific boundary and next gate

Proved by current code and outputs:

- the exact RealWonder Wan VAE can encode and decode the intended data and
  dimensions on this machine;
- the persistent point-identity transport operator accepts real 16-channel Wan
  latent features without NaN, collapse, support mismatch, or shape error;
- Correct identity outperforms a strictly matched Shuffled identity control in
  latent and decoded coarse-proxy comparisons for this Santa sequence.

Positive but not yet a paper-level conclusion:

- persistent material-aligned identity carries useful local appearance signal
  in this real Wan latent space;
- the result justifies a later full-generator smoke test on adequate hardware.

Not validated:

- improvement over RealWonder in generated future-video quality;
- response of the diffusion denoiser to transported latent conditioning;
- robot action, robot-cloth contact, real robot data, or cross-case
  generalization;
- whether hard transport is better than optical-flow warp in a generated-video
  comparison.

The VAE-only gate passes. The next scientific gate is one small, fixed-seed
full-generator comparison using existing or separately authorized weights on a
larger-GPU server. Downloading or running that model is outside this stage.
