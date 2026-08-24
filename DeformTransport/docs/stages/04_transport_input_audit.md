# Stage 4: transport-input audit

## Goal and decision

Audit the saved 21-frame Santa cloth run as an input to hard point transport.
This stage does not rerun Genesis and does not re-establish trajectory
stability. The existing artifact contains all expensive simulation outputs.
Only a deterministic, checkpoint-free conversion boundary is missing.

Source visibility must not be inferred from projection validity. All 28,264
Santa points project into the initial 512 render and the 480x832 crop, but only
23,921 distinct point IDs are selected by RealWonder's source-frame point
rasterizer. The latter set is the supported definition of `source_visible`.

## Audited source artifact

Source directory:

`artifacts/exhaustive_validation/santa_action_suite_20260802/right_s1_21f`

Directly reusable data:

| Data | Saved shape / representation | Status |
| --- | --- | --- |
| future 3D points | `[21, 28264, 3]`, float32 | ready |
| future render UV | `[21, 28264, 2]`, float32 | ready |
| future depth | `[21, 28264]`, float32 | ready |
| render projection validity | `[21, 28264]`, bool | ready |
| initial 3D/UV/depth/validity | `[28264, ...]` | ready |
| point-particle binding | `[28264, 5]`, int64 | ready and previously verified |
| frame IDs | `[21]`, int64, 0 through 20 | ready |
| simulation steps | `[21]`, int64, 10 through 210 | ready |
| camera K | `[21, 1, 4, 4]`, float32 | ready |
| camera R | `[21, 1, 3, 3]`, float32 | ready |
| camera T | `[21, 1, 3]`, float32 | ready |
| initial render | `frame_initial.png`, 512x512 | ready |
| simulated future renders | 21 `frame_*.png` files, 512x512 | ready |
| RealWonder flow | `[21, 2, 512, 512]`, float32 | ready |
| source raster point IDs | `[21, 512, 512]`, int32 | ready |

The first slice of `flow_source_point_indices.npy` was captured after
`render_preview()` and before the first future render. It therefore describes
the exact initial point raster used as the source of flow frame zero. Values of
`-1` are background. Non-negative values are global point indices in the same
concatenated object order as the trajectory export.

## Fields missing from the stage-1 export

The underlying data exists, but the following transport-facing fields are not
materialized in `point_trajectories.pt`:

- global `point_id: [N]` and per-point `object_id: [N]`;
- source raster visibility as `source_visible: [N]` and its ID list;
- initial and future coordinates on the verified 60x104 spatial grid;
- validity after the 832 resize and 480-high crop;
- explicit render/video/latent dimensions;
- paths to the initial RGB, future coarse RGB frames, flow, and raster IDs;
- a single validation function for the flattened transport contract.

A distinct initial camera entry is not present in the stage-1 file. This does
not block transport: the recorder already saved the initial projected UV and
the camera is static in this case. The conversion must use the saved initial UV
rather than claiming to reconstruct an unrecorded initial camera state.

## Required conversion

Add a small `deform_transport.transport_ready` module plus an export script.
They must:

1. load and validate the existing trajectory;
2. flatten objects without changing the within-object or object-concatenation
   order;
3. assign global point IDs in that exact order;
4. derive source visibility only from the first source-raster ID image;
5. reuse `map_image_uv_to_latent()` for initial and future coordinates;
6. combine render validity with crop validity for transport validity;
7. keep all points, including points sharing the same source cell;
8. preserve plain CPU Torch tensors at the serialization boundary;
9. store resource paths without copying the 91.8 MB simulation artifact.

No change to Genesis, point binding, projection, or flow generation is needed.
The four official simulations and the Santa action suite must not be rerun.

## Transport-ready contract for Santa

The standardized artifact must contain at least:

```text
format_version:                 int
case_name:                      str
frame_ids:                      [T] int64
simulation_steps:               [T] int64
point_id:                       [N] int64
object_id:                      [N] int64
material_type:                  list[str], indexed by object ID
points_3d:                      [T,N,3] float32
points_2d_render:               [T,N,2] float32
points_2d_latent:               [T,N,2] int64
depth:                          [T,N] float32
render_projection_valid:        [T,N] bool
projection_valid:               [T,N] bool after crop
source_points_3d:               [N,3] float32
source_points_2d_render:        [N,2] float32
source_points_2d_latent:        [N,2] int64
source_render_projection_valid: [N] bool
source_valid:                   [N] bool after crop
source_visible:                 [N] bool
source_visible_point_ids:       [Nv] int64
point_particle_binding:         [N,5] int64 for Santa
camera:                         K/R/T tensors
paths:                          initial/coarse/flow/raster source paths
```

For the current artifact, `T=21`, `N=28264`, `Nv=23921`, and `K=5`.
Dimensions are fixed metadata:

```text
render_height = render_width = 512
video_height = 480
video_width = 832
latent_height = 60
latent_width = 104
```

`points_2d_latent` stores the integer nearest-cell coordinates returned by the
already-tested `map_image_uv_to_latent()` convention. It does not merge point
IDs. `projection_valid` is the transport validity after both the 512 render
projection and the 480x832 crop; the original render-only mask remains available
under the explicit `render_projection_valid` name.

## Acceptance tests

- all standardized tensor shapes match `T`, `N`, and `K`;
- all source-visible IDs are in range and unique;
- `source_visible` is a subset of `source_valid`;
- latent coordinates exactly equal the existing mapping function output;
- crop-excluded points are invalid rather than clamped;
- global point order is stable across all frames and bindings;
- save/load round-trip passes validation;
- every saved tensor has the base `torch.Tensor` type on CPU;
- the generated artifact is
  `artifacts/transport_validation/santa_cloth_21f/transport_ready.pt`.
