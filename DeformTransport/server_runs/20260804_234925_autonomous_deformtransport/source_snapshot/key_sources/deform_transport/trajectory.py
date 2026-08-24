"""Point-trajectory recording utilities for RealWonder.

This module intentionally mirrors RealWonder's projection convention in
``SingleViewReconstructor._proj_uv``.  Keeping that convention in one small,
tested module prevents the later latent-transport stage from silently using a
different camera or crop coordinate system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


TRAJECTORY_FORMAT_VERSION = 1


def _plain_cpu_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    """Detach runtime tensor subclasses at the trajectory export boundary."""
    tensor = torch.as_tensor(value)
    if type(tensor) is not torch.Tensor:
        tensor = tensor.as_subclass(torch.Tensor)
    tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    if type(tensor) is not torch.Tensor:
        tensor = tensor.as_subclass(torch.Tensor)
    return tensor


def _remove_singleton_batch(tensor: torch.Tensor, expected_dims: int) -> torch.Tensor:
    if tensor.ndim == expected_dims + 1 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def project_points_realwonder(
    points_xyz: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    image_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project PyTorch3D-space points exactly as RealWonder does.

    Returns ``(uv, depth, valid)`` where ``uv`` is in RealWonder's 512-square
    render coordinates and ``valid`` only tests positive depth and image bounds.
    Occlusion is deliberately left to the renderer.
    """

    points_xyz = torch.as_tensor(points_xyz)
    K = _remove_singleton_batch(torch.as_tensor(K), 2)
    R = _remove_singleton_batch(torch.as_tensor(R), 2)
    T = _remove_singleton_batch(torch.as_tensor(T), 1)

    if points_xyz.ndim != 2 or points_xyz.shape[-1] != 3:
        raise ValueError(f"points_xyz must have shape [N, 3], got {tuple(points_xyz.shape)}")
    if K.shape not in ((3, 3), (4, 4)):
        raise ValueError(f"K must have shape [3, 3] or [4, 4], got {tuple(K.shape)}")
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape [3, 3], got {tuple(R.shape)}")
    if T.shape != (3,):
        raise ValueError(f"T must have shape [3], got {tuple(T.shape)}")

    device = points_xyz.device
    dtype = points_xyz.dtype
    intrinsics = K[:3, :3].clone().to(device=device, dtype=dtype)
    rotation = R.to(device=device, dtype=dtype)
    translation = T.to(device=device, dtype=dtype)

    # RealWonder receives PyTorch3D's 4x4 calibration matrix, whose [2, 2]
    # element is zero, and explicitly restores it before projection.
    intrinsics[2, 2] = 1.0
    camera_xyz = (rotation @ points_xyz.T).T + translation
    image_xyz = (intrinsics @ camera_xyz.T).T
    depth = camera_xyz[:, 2]
    uv = image_xyz[:, :2] / image_xyz[:, 2:3].clamp_min(1e-3)
    uv = float(image_size) - uv

    finite = torch.isfinite(points_xyz).all(dim=1) & torch.isfinite(uv).all(dim=1)
    valid = (
        finite
        & (depth > 1e-3)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < image_size)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < image_size)
    )
    return uv, depth, valid


def map_image_uv_to_latent(
    uv: torch.Tensor,
    *,
    source_size: int = 512,
    resized_size: int = 832,
    crop_start: int = 176,
    output_height: int = 480,
    latent_stride: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map RealWonder render coordinates into its cropped Wan latent grid.

    The offline pipeline first resizes 512x512 to 832x832, takes a 480x832
    vertical crop, and spatially downsamples by eight to a 60x104 latent grid.
    """

    uv = torch.as_tensor(uv)
    if uv.ndim != 2 or uv.shape[-1] != 2:
        raise ValueError(f"uv must have shape [N, 2], got {tuple(uv.shape)}")

    scale = float(resized_size) / float(source_size)
    video_uv = uv * scale
    video_uv = video_uv.clone()
    video_uv[:, 1] -= float(crop_start)
    valid = (
        torch.isfinite(video_uv).all(dim=1)
        & (video_uv[:, 0] >= 0)
        & (video_uv[:, 0] < resized_size)
        & (video_uv[:, 1] >= 0)
        & (video_uv[:, 1] < output_height)
    )

    latent_xy = torch.floor(video_uv / float(latent_stride)).to(torch.long)
    return latent_xy, video_uv, valid


class PointTrajectoryRecorder:
    """Accumulate identity-preserving point trajectories at render times."""

    def __init__(
        self,
        initial_object_points: Sequence[torch.Tensor],
        *,
        material_types: Sequence[str],
        camera: Any,
        binding_indices: Mapping[int, torch.Tensor] | None = None,
        image_size: int = 512,
    ) -> None:
        if len(initial_object_points) != len(material_types):
            raise ValueError("initial_object_points and material_types must have equal length")

        self.image_size = int(image_size)
        self.material_types = list(material_types)
        self.initial_object_points = [
            _plain_cpu_tensor(points, dtype=torch.float32)
            for points in initial_object_points
        ]
        self.binding_indices = {
            int(object_id): _plain_cpu_tensor(indices, dtype=torch.long)
            for object_id, indices in (binding_indices or {}).items()
        }
        self._initial_camera = self._camera_tensors(camera)
        self._frames: list[dict[str, Any]] = []

    @staticmethod
    def _camera_tensors(camera: Any) -> dict[str, torch.Tensor]:
        try:
            return {
                "K": _plain_cpu_tensor(camera.K, dtype=torch.float32),
                "R": _plain_cpu_tensor(camera.R, dtype=torch.float32),
                "T": _plain_cpu_tensor(camera.T, dtype=torch.float32),
            }
        except AttributeError as exc:
            raise TypeError("camera must expose K, R, and T tensors") from exc

    def record(
        self,
        *,
        frame_id: int,
        simulation_step: int,
        object_points: Sequence[torch.Tensor],
        camera: Any,
    ) -> None:
        if len(object_points) != len(self.initial_object_points):
            raise ValueError("object count changed during simulation")

        frame_points = []
        for object_id, (points, initial) in enumerate(zip(object_points, self.initial_object_points)):
            points_cpu = _plain_cpu_tensor(points, dtype=torch.float32)
            if points_cpu.shape != initial.shape:
                raise ValueError(
                    f"object {object_id} point identity changed: "
                    f"expected {tuple(initial.shape)}, got {tuple(points_cpu.shape)}"
                )
            if not torch.isfinite(points_cpu).all():
                raise ValueError(f"object {object_id} contains non-finite positions")
            frame_points.append(points_cpu)

        self._frames.append(
            {
                "frame_id": int(frame_id),
                "simulation_step": int(simulation_step),
                "object_points": frame_points,
                "camera": self._camera_tensors(camera),
            }
        )

    def state_dict(self) -> dict[str, Any]:
        if not self._frames:
            raise RuntimeError("cannot export an empty trajectory")

        camera_K = torch.stack([frame["camera"]["K"] for frame in self._frames])
        camera_R = torch.stack([frame["camera"]["R"] for frame in self._frames])
        camera_T = torch.stack([frame["camera"]["T"] for frame in self._frames])
        objects = []

        for object_id, initial_points in enumerate(self.initial_object_points):
            initial_uv, initial_depth, initial_valid = project_points_realwonder(
                initial_points,
                self._initial_camera["K"],
                self._initial_camera["R"],
                self._initial_camera["T"],
                self.image_size,
            )
            points_3d = torch.stack(
                [frame["object_points"][object_id] for frame in self._frames]
            )
            expected_shape = (len(self._frames), *initial_points.shape)
            if tuple(points_3d.shape) != expected_shape:
                raise ValueError(
                    f"object {object_id} trajectory shape changed: "
                    f"expected {expected_shape}, got {tuple(points_3d.shape)}"
                )
            if not torch.isfinite(points_3d).all():
                raise ValueError(f"object {object_id} trajectory contains non-finite positions")
            uv_frames = []
            depth_frames = []
            valid_frames = []
            # Explicit indexing avoids Tensor.__iter__ -> unbind(), which can be
            # intercepted by the Genesis/PyTorch3D runtime under Torch 2.12.
            for frame_index in range(points_3d.shape[0]):
                points = points_3d[frame_index]
                uv, depth, valid = project_points_realwonder(
                    points,
                    camera_K[frame_index],
                    camera_R[frame_index],
                    camera_T[frame_index],
                    self.image_size,
                )
                uv_frames.append(uv)
                depth_frames.append(depth)
                valid_frames.append(valid)

            objects.append(
                {
                    "object_id": object_id,
                    "material_type": self.material_types[object_id],
                    "initial_points_3d": initial_points,
                    "initial_points_uv": initial_uv,
                    "initial_depth": initial_depth,
                    "initial_projection_valid": initial_valid,
                    "points_3d": points_3d,
                    "points_uv": torch.stack(uv_frames),
                    "depth": torch.stack(depth_frames),
                    "projection_valid": torch.stack(valid_frames),
                    "binding_particle_indices": self.binding_indices.get(object_id),
                }
            )

        state = {
            "format_version": TRAJECTORY_FORMAT_VERSION,
            "coordinate_system": "pytorch3d_world_and_realwonder_512_uv",
            "image_size": self.image_size,
            "frame_ids": torch.tensor(
                [frame["frame_id"] for frame in self._frames], dtype=torch.long
            ),
            "simulation_steps": torch.tensor(
                [frame["simulation_step"] for frame in self._frames], dtype=torch.long
            ),
            "camera": {"K": camera_K, "R": camera_R, "T": camera_T},
            "objects": objects,
        }
        validate_trajectory_export(state)
        return state

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path


def validate_trajectory_export(state: Mapping[str, Any]) -> None:
    """Raise a descriptive error when an export breaks the stage-1 contract."""

    if state.get("format_version") != TRAJECTORY_FORMAT_VERSION:
        raise ValueError(f"unsupported trajectory format: {state.get('format_version')}")
    frame_ids = state.get("frame_ids")
    simulation_steps = state.get("simulation_steps")
    objects = state.get("objects")
    if not isinstance(frame_ids, torch.Tensor) or frame_ids.ndim != 1:
        raise ValueError("frame_ids must be a 1D tensor")
    if not isinstance(simulation_steps, torch.Tensor) or simulation_steps.shape != frame_ids.shape:
        raise ValueError("simulation_steps must match frame_ids")
    if frame_ids.numel() == 0:
        raise ValueError("trajectory export has no frames")
    if not isinstance(objects, list) or not objects:
        raise ValueError("trajectory export has no objects")

    expected_frames = frame_ids.numel()
    for object_state in objects:
        points_3d = object_state["points_3d"]
        points_uv = object_state["points_uv"]
        valid = object_state["projection_valid"]
        initial = object_state["initial_points_3d"]
        if points_3d.ndim != 3 or points_3d.shape[0] != expected_frames or points_3d.shape[2] != 3:
            raise ValueError("points_3d must have shape [T, N, 3]")
        if points_uv.shape != points_3d.shape[:2] + (2,):
            raise ValueError("points_uv must have shape [T, N, 2]")
        if valid.shape != points_3d.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("projection_valid must be bool with shape [T, N]")
        if initial.shape != points_3d.shape[1:]:
            raise ValueError("initial point identity does not match trajectory point identity")
        if not torch.isfinite(points_3d).all() or not torch.isfinite(points_uv).all():
            raise ValueError("trajectory export contains non-finite values")
        bindings = object_state.get("binding_particle_indices")
        if bindings is not None and (bindings.ndim != 2 or bindings.shape[0] != initial.shape[0]):
            raise ValueError("binding_particle_indices must have shape [N, K]")
