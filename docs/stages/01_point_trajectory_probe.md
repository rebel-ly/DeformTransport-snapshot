# Stage 1: RealWonder point-trajectory probe

## Goal

Verify the minimum geometry interface needed by DeformTransport before touching
Wan or downloading video-generation checkpoints:

```
observed RealWonder foreground point
  -> fixed simulation-particle binding
  -> per-frame 3D point position
  -> RealWonder 512x512 projection
  -> cropped 480x832 video / 60x104 Wan latent coordinate
```

## Why this implementation

RealWonder already binds every deformable reconstructed foreground point to its
nearest simulation particles once (`closest_indices`) and updates the point from
the same particle identities at every step. Its optical flow is computed from
the projected displacement of the same ordered foreground points. Reusing those
intermediates is both cheaper and less error-prone than adding a separate point
tracker.

Wan-Move motivates transporting first-frame condition features along dense point
trajectories. At this stage we only establish the required trajectory and
coordinate contract; no feature copying, diffusion intervention, or training is
introduced.

## Export contract

`point_trajectories.pt` stores, per object:

- initial observed points and their projected coordinates;
- `[T, N, 3]` identity-preserving simulated point positions;
- `[T, N, 2]` projected coordinates and validity masks;
- the original `[N, K]` point-to-particle binding when applicable;
- the exact camera matrices used for each rendered frame.

The recorder is opt-in with `export_point_trajectories: true`, so the upstream
RealWonder path remains unchanged by default.

## Validation gates

1. Bundled `santa_cloth` points project to finite, in-frame coordinates.
2. The resize/crop mapping produces valid indices on RealWonder's 60x104 spatial
   latent grid.
3. Dynamic export preserves a constant `N` and the same binding row for every
   frame.
4. Once Genesis is available, compare exported UV displacements against
   RealWonder's saved optical-flow field on the same run.

Gates 1-2 can run immediately on the repository's real preprocessed data. Gates
3-4 require the pinned Genesis/PyTorch3D runtime but no video model checkpoint.

On the RTX 5060 workstation, run `scripts/setup_wsl_f.ps1` once from an
Administrator PowerShell terminal.  It enables the required Windows features
without restarting automatically.  If it reports that a restart is required,
restart Windows and run the same script a second time; the Ubuntu 22.04 virtual
disk and Linux workspace are then created under `F:\WSL`.

For a short dynamic probe under WSL2, create an isolated Python 3.11 Conda
environment.  On RealWonder's original CUDA 12.1 hardware profile use
`requirements-stage1-wsl.txt`.  RTX 50-series GPUs must instead use the
Blackwell profile because Torch 2.5.1 predates `sm_120` support:

```bash
sudo apt-get install -y libopengl0
conda create -n deformtransport-stage1 \
  python=3.11 pip=25.0.1 setuptools=75.8.2 wheel=0.45.1 -y
conda activate deformtransport-stage1
python -m pip install -r requirements-stage1-wsl-blackwell.txt
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYOPENGL_PLATFORM=egl
python scripts/run_realwonder_trajectory_probe.py --frames 4
```

`libopengl0` is required for PyMeshLab's remeshing plugin.  WSL exposes the
NVIDIA driver bridge in `/usr/lib/wsl/lib`; GsTaichi otherwise reports that
`libcuda.so` is missing and silently falls back to CPU.  The local Conda
environment can place the two exports above in `etc/conda/activate.d/` so they
are applied automatically.

Installation alone is not an acceptance gate.  Before running the probe, test a
real CUDA tensor operation, import PyTorch3D's compiled extension, render a tiny
point cloud, and step a minimal Genesis GPU scene.

## Verified RTX 5060 result (2026-08-02)

The four-frame right-wind probe on the bundled `santa_cloth` case passed all
dynamic gates:

- one object and 28,264 reconstructed surface points at every render;
- a fixed `[28264, 5]` point-to-particle binding matching the simulator export;
- finite `[4, 28264, 3]` positions and 100% valid projections in all frames;
- flow comparison coverage of 97.22%-97.30%;
- median endpoint error of 0.059-0.077 pixels and mean direction cosine of
  0.974-0.980;
- visible non-zero cloth motion over simulation steps 10, 20, 30, and 40;
- approximately 15.3 MB of output, 5.0 seconds after kernel caching, and 428 MiB
  peak Torch allocation (excluding GsTaichi allocations).

Torch 2.12 requires two compatibility boundaries that do not change the method:
video writing falls back to ImageIO after `torchvision.io.write_video` removal,
and Genesis tensor subclasses are converted to plain CPU Torch tensors before
trajectory serialization.
