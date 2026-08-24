# Stage 3: checkpoint-free validation matrix

## Scope and verdict

This stage exhausts the validation that is justified by the current local
runtime and RealWonder's bundled preprocessed assets. It tests the geometry,
physics, projection, point identity, optical-flow, action-conditioning, and
export contracts. It does **not** test Wan latent transport or future-video
quality because no video checkpoint is installed. It also does **not** count as
a robot-cloth experiment because no bundled robot-deformable preprocessed case
is available.

The checkpoint-free geometry/physics interface passes. The only failed check is
strict bitwise repeatability on the GPU; practical numerical repeatability
passes by a wide margin and is reported separately rather than hidden.

## Runtime used

- Windows 11 with the complete WSL distribution stored on drive F;
- WSL Ubuntu 22.04 and Python 3.11;
- RTX 5060 with Torch 2.12.0+cu130 and PyTorch3D 0.7.9;
- Genesis commit `3aa206cd84729bc7cc14fb4007aeb95a0bead7aa`;
- NumPy 1.26.4 and GsTaichi 2.1.1;
- no Wan, RealWonder-video, SAM, FLUX, or QWM checkpoints.

CUDA tensor operations, PyTorch3D CUDA point rasterization, and a minimal
Genesis GPU step were independently smoke-tested before the experiments.

## Static validation on every bundled preprocessed case

All points are finite and all four overlays were visually inspected against the
official RGB backgrounds.

| Case | Materials / objects | Points | Valid in 512 render | Valid after 480x832 crop | Occupied 60x104 latent cells |
| --- | --- | ---: | ---: | ---: | ---: |
| `lamp` | one rigid object | 19,394 | 100% | 100% | 709 |
| `tree` | one MPM elastic object | 15,774 | 100% | 100% | 715 |
| `santa_cloth` | one PBD cloth object | 28,264 | 100% | 100% | 1,257 |
| `persimmon` | three rigid objects | 12,381 / 12,661 / 12,069 | 100% / 100% / 100% | 100% / 100% / 85.96% | per-object export verified |

The lower `persimmon` crop ratio is not a projection failure: the overlay shows
the bottom fruit extending below the model's vertical video crop. This is an
important future source-observation-mask edge case.

## Four-frame dynamic validation across materials

Each run uses the case's official `frame_steps`, exports the ordered point
states and the RealWonder flow generated in the same render call, and checks
initial-state equality, fixed object/point counts, finite coordinates, and
projection validity. Deformable objects additionally require the exported
`[N, 5]` binding to exactly match the simulator's fixed binding.

| Case | Object | Binding | Min flow coverage | Max median EPE | Min mean cosine | Final mean motion |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `lamp` | rigid | not applicable | 100% | 0.0031 px | 0.999999 | 0.90 px |
| `tree` | MPM elastic | `[15774, 5]`, exact | 99.87% | 0.0637 px | 0.9978 | 2.77 px |
| `santa_cloth` | PBD cloth | `[28264, 5]`, exact | 97.22% | 0.0773 px | 0.9738 | 13.89 px |
| `persimmon` | rigid target 0 | not applicable | 99.98% | 0.7048 px | 0.9423 | 9.68 px |
| `persimmon` | rigid object 1 | not applicable | 86.76% | 0.3159 px | 0.9396 | 2.78 px |
| `persimmon` | rigid object 2 | not applicable | 76.92% | 0.3711 px | 0.7552 | 2.96 px |

All objects remain finite and 100% valid in the 512 render projection. Initial
and final images were visually compared for every case, so these are not
success-by-exit-code results.

The multi-object flow comparison uses the point ID selected by RealWonder's
source-frame point rasterizer. Without that filter, an occluded point can sample
flow written by a different foreground object at the same pixel. The lower
coverage and cosine for the bottom persimmon are therefore an honest boundary
and occlusion limitation, not silently discarded data. This change affects only
the validation metric; it does not alter RealWonder flow or simulation.

## Santa cloth action causality

Four-frame runs use the same initial image, handler, camera, simulation settings,
and seed. The final mean horizontal UV response is:

| Condition | Final mean horizontal displacement |
| --- | ---: |
| no wind | +0.249 px |
| right direction with strength 0 | +0.249 px |
| left, strength 1 | -12.913 px |
| right, strength 0.5 | +8.222 px |
| right, strength 1 | +13.453 px |
| right, strength 2 | +20.539 px |

The results pass three causal checks: left and right have opposite responses,
wind produces much more horizontal motion than no wind, and right-wind response
is monotonic for strengths 0.5, 1, and 2. No wind is not a static scene: gravity
and cloth relaxation produce about 4.43 px of mean vertical motion. The explicit
zero-strength right-force run is practically equivalent to the no-direction
run, confirming that the action parameter does not introduce motion by itself.

## Same-seed repeatability

Two independent right-wind, strength-1, four-frame GPU runs were compared.

Strict equality fails for points, UVs, dense flow, and rendered images. This is
small GPU/GsTaichi numerical nondeterminism rather than experimental-scale
instability:

- point error: mean 0.0026 mm, p99 0.0255 mm, maximum 0.2006 mm;
- UV error: mean 0.00045 px, p99 0.00464 px, maximum 0.0261 px;
- dense-flow error on shared support: mean 0.00050 px, p99 0.00460 px;
- flow-support disagreement: 0.000885% of union support;
- rendered 8-bit image MAE: 0.00120 intensity levels, with 99% of values exact;
- maximum change in the reported median-EPE metric: 0.000036 px;
- maximum change in the reported cosine metric: 0.000131.

An isolated maximum dense-flow difference of about 4.94 px occurs at a tiny
raster-support boundary. It is retained in the report; aggregate and
visibility-aware metrics remain stable. Experiments on this workstation should
therefore use tolerance-based regression, fixed seeds, and saved artifacts, not
claim bitwise determinism.

## Twenty-one-frame horizon

The right-wind, strength-1 cloth run covers simulation steps 10 through 210.

- all 28,264 point identities and the exact `[28264, 5]` binding are retained;
- all 3D values are finite and projection validity remains 100%;
- minimum flow comparison coverage is 97.08%;
- maximum median flow endpoint error is 0.0773 px;
- minimum mean motion cosine is 0.9556;
- final mean UV displacement is `[53.505, -12.716]` px;
- total runtime is 8.44 s after kernel caching;
- Torch peak allocation is about 428 MiB (Genesis allocations excluded);
- output size is about 91.8 MB.

The initial and final rendered frames were visually inspected. The cloth has
large, visible rightward deformation while remaining in view; motion is not a
zero-update artifact.

## Automated regression status

`tests/test_realwonder_trajectory.py` passes 5/5 tests, covering projection,
video-to-latent crop mapping, identity preservation, Genesis tensor-subclass
serialization, and mixed rigid/deformable recording. The action suite reports:

- all causal and horizon checks: pass;
- practical same-seed repeatability checks: pass;
- no-direction versus zero-force equivalence checks: pass;
- strict bitwise repeatability: fail, explicitly retained.

The machine-readable report is
`artifacts/exhaustive_validation/santa_action_suite_20260802/suite_report.json`.

## What this establishes, and what it does not

Established:

1. RealWonder's ordered reconstructed points can be exported through real
   rigid, PBD-cloth, and MPM-elastic simulations without losing point count or
   coordinate validity.
2. Deformable-point bindings are fixed across time and can support
   identity-preserving feature transport.
3. Projected point displacement agrees closely with the dense flow produced by
   the same geometry, including over a 21-frame horizon.
4. Cloth trajectories respond causally to action direction and strength.
5. Hard-cell latent aggregation is necessary because many observed points map
   to the same Wan cell; the earlier Santa static probe found 28,264 points in
   only 1,257 occupied cells.

Not established:

1. point-latent transport improves generated future videos;
2. Wan VAE features can yet be sampled, transported, fused, or decoded locally;
3. the robot-action path runs end to end;
4. granular-sand evidence generalizes to cloth;
5. any learned, soft-splatting, depth-conflict, occlusion-completion, CamProbe,
   QWM, training, or fine-tuning mechanism is needed or beneficial.

The next valid gate remains the robot-action asset described in
`02_robot_action_asset_audit.md`. Hard point-latent transport should begin only
after that gate passes, unless the research plan is explicitly changed to run
cloth geometry and robot action as separate validation tracks.
