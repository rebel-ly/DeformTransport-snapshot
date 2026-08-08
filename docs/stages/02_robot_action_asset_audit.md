# Stage 2: robot-action asset audit

## Status

Blocked on preprocessed reconstruction assets, not on the Franka action code or
the trajectory recorder.  Do not treat this stage as a passed robot-action
experiment.

## Evidence available in the repository

`cases/sand_house` contains:

- `input.png` and `inpainted.png`;
- `config.yaml` with `material_type: ["mpm_sand"]` and the physical settings;
- a complete `SandHouse` case handler that creates two Franka arms, solves
  inverse kinematics, applies the scripted actions, and exposes the active robot
  mesh to the renderer;
- the Franka MJCF and its visual/collision assets under
  `cases/xml/franka_emika_panda`.

The checked [official sand directory](https://github.com/liuwei283/RealWonder/tree/main/cases/sand_house)
contains the same three case files.  The checked
[official interactive demo data](https://github.com/liuwei283/RealWonder/tree/main/demo_web/demo_data)
contains lamp, persimmon, santa_cloth, and tree, but no sand case.

## Missing low-cost demo contract

`InteractiveSimulator` requires a preprocessed directory containing at least:

- `config.yaml`;
- `camera.pt` with `K`, `R`, `T`, and focal length;
- `bg_points.pt` with points and colors;
- one or more `fg_pcs/pc_*.pt` files with ordered points and colors;
- matching `fg_meshes/mesh_*.obj` files;
- optional `fg_masks/` and `ground_plane_normal.npy`.

None of the required camera, background point, foreground point, or mesh files
are published for `sand_house` in the current repository.

## Why the offline path is not a minimal substitute

The offline `SingleViewReconstructor` imports SAM 3D Objects, segments with
SAM2, loads MoGe for depth, and then produces the missing point clouds and
meshes.  In the imported source, the SAM2 checkpoint path is hard-coded to the
original author's filesystem.  The relevant submodules, Python packages, and
checkpoints are absent locally.  Installing them would be a multi-model
reconstruction deployment rather than a short trajectory probe, and it is not
justified on the 8 GB RTX 5060 before a preprocessed package is sought.

## Smallest valid unblock

Prefer an official or author-exported `sand_house` preprocessed directory that
matches the contract above.  Once available, first run only 2-4 render frames
and require:

- both Franka entities build and the intended arm action advances;
- the robot mesh appears in rendered frames;
- the sand point count and fixed `[N, 5]` binding remain constant;
- exported positions are finite and visibly respond to contact;
- trajectory frame IDs match flow frames;
- projected point displacement and RealWonder flow agree;
- the result is described as granular sand, never as cloth.

Only if no preprocessed package can be obtained should the full
SAM2/SAM-3D-Objects/MoGe reconstruction path be deployed, preferably on a
larger-GPU server and in a separate environment.
