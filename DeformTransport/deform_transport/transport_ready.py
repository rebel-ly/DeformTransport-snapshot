"""Standardize saved RealWonder trajectories for point-feature transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .trajectory import map_image_uv_to_latent, validate_trajectory_export


TRANSPORT_READY_FORMAT_VERSION = 1
RENDER_HEIGHT = 512
RENDER_WIDTH = 512
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 832
LATENT_HEIGHT = 60
LATENT_WIDTH = 104


def _plain_cpu(value: torch.Tensor) -> torch.Tensor:
    """Remove runtime tensor subclasses without changing dtype or values."""

    tensor = torch.as_tensor(value)
    if type(tensor) is not torch.Tensor:
        tensor = tensor.as_subclass(torch.Tensor)
    tensor = tensor.detach().to(device="cpu").contiguous()
    if type(tensor) is not torch.Tensor:
        tensor = tensor.as_subclass(torch.Tensor)
    return tensor


def _plain_structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _plain_cpu(value)
    if isinstance(value, dict):
        return {key: _plain_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_structure(item) for item in value)
    return value


def _map_frames_to_latent(
    points_uv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points_uv.ndim != 3 or points_uv.shape[-1] != 2:
        raise ValueError(
            f"points_uv must have shape [T, N, 2], got {tuple(points_uv.shape)}"
        )
    frame_count, point_count = points_uv.shape[:2]
    latent_xy, video_uv, crop_valid = map_image_uv_to_latent(
        points_uv.reshape(-1, 2)
    )
    return (
        latent_xy.reshape(frame_count, point_count, 2),
        video_uv.reshape(frame_count, point_count, 2),
        crop_valid.reshape(frame_count, point_count),
    )


def _flatten_trajectory_objects(
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    objects = trajectory["objects"]
    points_3d = []
    points_uv = []
    depth = []
    render_valid = []
    source_points_3d = []
    source_points_uv = []
    source_depth = []
    source_render_valid = []
    object_ids = []
    material_types = []
    bindings_by_object = []
    object_point_ranges = []
    point_offset = 0

    for object_index, object_state in enumerate(objects):
        object_id = int(object_state["object_id"])
        if object_id != object_index:
            raise ValueError(
                "transport flattening requires object list order to match object_id"
            )
        point_count = int(object_state["points_3d"].shape[1])
        points_3d.append(_plain_cpu(object_state["points_3d"]))
        points_uv.append(_plain_cpu(object_state["points_uv"]))
        depth.append(_plain_cpu(object_state["depth"]))
        render_valid.append(_plain_cpu(object_state["projection_valid"]))
        source_points_3d.append(_plain_cpu(object_state["initial_points_3d"]))
        source_points_uv.append(_plain_cpu(object_state["initial_points_uv"]))
        source_depth.append(_plain_cpu(object_state["initial_depth"]))
        source_render_valid.append(
            _plain_cpu(object_state["initial_projection_valid"])
        )
        object_ids.append(torch.full((point_count,), object_id, dtype=torch.long))
        material_types.append(str(object_state["material_type"]))
        binding = object_state.get("binding_particle_indices")
        bindings_by_object.append(None if binding is None else _plain_cpu(binding))
        object_point_ranges.append(
            {
                "object_id": object_id,
                "start": point_offset,
                "end": point_offset + point_count,
            }
        )
        point_offset += point_count

    if all(binding is None for binding in bindings_by_object):
        flattened_binding = None
    elif all(binding is not None for binding in bindings_by_object):
        widths = {int(binding.shape[1]) for binding in bindings_by_object}
        flattened_binding = (
            torch.cat(bindings_by_object, dim=0) if len(widths) == 1 else None
        )
    else:
        flattened_binding = None

    return {
        "points_3d": torch.cat(points_3d, dim=1),
        "points_uv": torch.cat(points_uv, dim=1),
        "depth": torch.cat(depth, dim=1),
        "render_valid": torch.cat(render_valid, dim=1),
        "source_points_3d": torch.cat(source_points_3d, dim=0),
        "source_points_uv": torch.cat(source_points_uv, dim=0),
        "source_depth": torch.cat(source_depth, dim=0),
        "source_render_valid": torch.cat(source_render_valid, dim=0),
        "object_id": torch.cat(object_ids, dim=0),
        "material_type": material_types,
        "point_particle_binding": flattened_binding,
        "point_particle_binding_by_object": bindings_by_object,
        "object_point_ranges": object_point_ranges,
    }


def build_transport_ready(
    trajectory: Mapping[str, Any],
    source_raster_point_indices: torch.Tensor,
    *,
    case_name: str,
    source_trajectory_path: str | Path | None = None,
    initial_rgb_path: str | Path | None = None,
    coarse_rgb_paths: Sequence[str | Path] = (),
    flow_path: str | Path | None = None,
    source_raster_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the flattened, source-visible transport contract.

    ``source_raster_point_indices`` may be the initial ``[H, W]`` raster or the
    saved ``[T, H, W]`` stack. In the latter case only frame zero is the visual
    source used to read first-frame payloads.
    """

    validate_trajectory_export(trajectory)
    flat = _flatten_trajectory_objects(trajectory)
    frame_count, point_count = flat["points_3d"].shape[:2]
    point_id = torch.arange(point_count, dtype=torch.long)

    raster = _plain_cpu(torch.as_tensor(source_raster_point_indices)).to(torch.long)
    if raster.ndim == 3:
        if raster.shape[0] != frame_count:
            raise ValueError(
                "source raster frame count must match the trajectory frame count"
            )
        source_raster = raster[0]
    elif raster.ndim == 2:
        source_raster = raster
    else:
        raise ValueError(
            "source_raster_point_indices must have shape [H, W] or [T, H, W]"
        )
    selected = source_raster[source_raster >= 0]
    if selected.numel() and int(selected.max()) >= point_count:
        raise ValueError("source raster contains a point ID outside the trajectory")
    raster_visible_ids = torch.unique(selected, sorted=True)

    source_latent, source_video, source_crop_valid = map_image_uv_to_latent(
        flat["source_points_uv"]
    )
    future_latent, future_video, future_crop_valid = _map_frames_to_latent(
        flat["points_uv"]
    )
    source_valid = flat["source_render_valid"] & source_crop_valid
    projection_valid = flat["render_valid"] & future_crop_valid
    source_visible = torch.zeros(point_count, dtype=torch.bool)
    source_visible[raster_visible_ids] = True
    source_visible &= source_valid
    source_visible_point_ids = point_id[source_visible]

    if coarse_rgb_paths and len(coarse_rgb_paths) != frame_count:
        raise ValueError(
            f"expected {frame_count} coarse RGB paths, got {len(coarse_rgb_paths)}"
        )

    def path_string(path: str | Path | None) -> str | None:
        return str(Path(path).resolve()) if path is not None else None

    state = {
        "format_version": TRANSPORT_READY_FORMAT_VERSION,
        "case_name": str(case_name),
        "coordinate_system": {
            "points_3d": trajectory.get("coordinate_system"),
            "points_2d_render": "realwonder_512_uv_xy",
            "points_2d_video": "realwonder_resized_832_crop_480_uv_xy",
            "points_2d_latent": "wan_spatial_cell_xy_nearest",
        },
        "frame_ids": _plain_cpu(trajectory["frame_ids"]).to(torch.long),
        "simulation_steps": _plain_cpu(trajectory["simulation_steps"]).to(
            torch.long
        ),
        "point_id": point_id,
        "object_id": flat["object_id"],
        "material_type": flat["material_type"],
        "object_point_ranges": flat["object_point_ranges"],
        "points_3d": flat["points_3d"],
        "points_2d_render": flat["points_uv"],
        "points_2d_video": future_video,
        "points_2d_latent": future_latent,
        "depth": flat["depth"],
        "render_projection_valid": flat["render_valid"],
        "projection_valid": projection_valid,
        "source_points_3d": flat["source_points_3d"],
        "source_points_2d_render": flat["source_points_uv"],
        "source_points_2d_video": source_video,
        "source_points_2d_latent": source_latent,
        "source_depth": flat["source_depth"],
        "source_render_projection_valid": flat["source_render_valid"],
        "source_valid": source_valid,
        "source_visible": source_visible,
        "source_visible_point_ids": source_visible_point_ids,
        "source_raster_visible_point_ids": raster_visible_ids,
        "point_particle_binding": flat["point_particle_binding"],
        "point_particle_binding_by_object": flat[
            "point_particle_binding_by_object"
        ],
        "camera": _plain_structure(trajectory["camera"]),
        "render_height": RENDER_HEIGHT,
        "render_width": RENDER_WIDTH,
        "video_height": VIDEO_HEIGHT,
        "video_width": VIDEO_WIDTH,
        "latent_height": LATENT_HEIGHT,
        "latent_width": LATENT_WIDTH,
        "paths": {
            "source_trajectory": path_string(source_trajectory_path),
            "initial_rgb": path_string(initial_rgb_path),
            "coarse_rgb_frames": [path_string(path) for path in coarse_rgb_paths],
            "flow": path_string(flow_path),
            "source_raster_point_indices": path_string(source_raster_path),
        },
    }
    state = _plain_structure(state)
    validate_transport_ready(state)
    return state


def _require_tensor(
    state: Mapping[str, Any],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = state.get(name)
    if type(value) is not torch.Tensor:
        raise ValueError(f"{name} must be a plain torch.Tensor")
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(
            f"{name} must be {dtype} with shape {shape}, got "
            f"{value.dtype} {tuple(value.shape)}"
        )
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be stored on CPU")
    return value


def validate_transport_ready(state: Mapping[str, Any]) -> None:
    """Validate the flattened transport contract and verified crop mapping."""

    if state.get("format_version") != TRANSPORT_READY_FORMAT_VERSION:
        raise ValueError(
            f"unsupported transport-ready format: {state.get('format_version')}"
        )
    frame_ids = state.get("frame_ids")
    point_id = state.get("point_id")
    if type(frame_ids) is not torch.Tensor or frame_ids.ndim != 1:
        raise ValueError("frame_ids must be a plain 1D tensor")
    if type(point_id) is not torch.Tensor or point_id.ndim != 1:
        raise ValueError("point_id must be a plain 1D tensor")
    frame_count = int(frame_ids.numel())
    point_count = int(point_id.numel())
    if frame_count == 0 or point_count == 0:
        raise ValueError("transport-ready export cannot be empty")
    if not torch.equal(point_id, torch.arange(point_count, dtype=torch.long)):
        raise ValueError("point_id must preserve contiguous flattened point order")

    _require_tensor(
        state, "simulation_steps", shape=(frame_count,), dtype=torch.long
    )
    _require_tensor(state, "object_id", shape=(point_count,), dtype=torch.long)
    points_3d = _require_tensor(
        state,
        "points_3d",
        shape=(frame_count, point_count, 3),
        dtype=torch.float32,
    )
    render_uv = _require_tensor(
        state,
        "points_2d_render",
        shape=(frame_count, point_count, 2),
        dtype=torch.float32,
    )
    _require_tensor(
        state,
        "points_2d_video",
        shape=(frame_count, point_count, 2),
        dtype=torch.float32,
    )
    latent_uv = _require_tensor(
        state,
        "points_2d_latent",
        shape=(frame_count, point_count, 2),
        dtype=torch.long,
    )
    depth = _require_tensor(
        state,
        "depth",
        shape=(frame_count, point_count),
        dtype=torch.float32,
    )
    render_valid = _require_tensor(
        state,
        "render_projection_valid",
        shape=(frame_count, point_count),
        dtype=torch.bool,
    )
    projection_valid = _require_tensor(
        state,
        "projection_valid",
        shape=(frame_count, point_count),
        dtype=torch.bool,
    )
    source_points = _require_tensor(
        state,
        "source_points_3d",
        shape=(point_count, 3),
        dtype=torch.float32,
    )
    source_render_uv = _require_tensor(
        state,
        "source_points_2d_render",
        shape=(point_count, 2),
        dtype=torch.float32,
    )
    _require_tensor(
        state,
        "source_points_2d_video",
        shape=(point_count, 2),
        dtype=torch.float32,
    )
    source_latent_uv = _require_tensor(
        state,
        "source_points_2d_latent",
        shape=(point_count, 2),
        dtype=torch.long,
    )
    source_depth = _require_tensor(
        state, "source_depth", shape=(point_count,), dtype=torch.float32
    )
    source_render_valid = _require_tensor(
        state,
        "source_render_projection_valid",
        shape=(point_count,),
        dtype=torch.bool,
    )
    source_valid = _require_tensor(
        state, "source_valid", shape=(point_count,), dtype=torch.bool
    )
    source_visible = _require_tensor(
        state, "source_visible", shape=(point_count,), dtype=torch.bool
    )
    source_visible_ids = state.get("source_visible_point_ids")
    if type(source_visible_ids) is not torch.Tensor or source_visible_ids.dtype != torch.long:
        raise ValueError("source_visible_point_ids must be a plain int64 tensor")
    if source_visible_ids.ndim != 1:
        raise ValueError("source_visible_point_ids must be one-dimensional")
    if source_visible_ids.numel() and (
        int(source_visible_ids.min()) < 0
        or int(source_visible_ids.max()) >= point_count
    ):
        raise ValueError("source_visible_point_ids contains an out-of-range ID")
    if not torch.equal(source_visible_ids, point_id[source_visible]):
        raise ValueError("source_visible IDs and mask disagree")
    if bool((source_visible & ~source_valid).any()):
        raise ValueError("source_visible must be a subset of source_valid")

    expected_source_latent, _, source_crop_valid = map_image_uv_to_latent(
        source_render_uv
    )
    expected_future_latent, _, future_crop_valid = _map_frames_to_latent(render_uv)
    if not torch.equal(source_latent_uv, expected_source_latent):
        raise ValueError("source latent coordinates do not match the verified mapping")
    if not torch.equal(latent_uv, expected_future_latent):
        raise ValueError("future latent coordinates do not match the verified mapping")
    if not torch.equal(source_valid, source_render_valid & source_crop_valid):
        raise ValueError("source_valid does not combine render and crop validity")
    if not torch.equal(projection_valid, render_valid & future_crop_valid):
        raise ValueError("projection_valid does not combine render and crop validity")

    valid_source_cells = source_latent_uv[source_valid]
    valid_future_cells = latent_uv[projection_valid]
    for cells, label in (
        (valid_source_cells, "source"),
        (valid_future_cells, "future"),
    ):
        if cells.numel() and not bool(
            (cells[:, 0] >= 0).all()
            and (cells[:, 0] < int(state["latent_width"])).all()
            and (cells[:, 1] >= 0).all()
            and (cells[:, 1] < int(state["latent_height"])).all()
        ):
            raise ValueError(f"valid {label} latent coordinates are out of bounds")

    for tensor, name in (
        (points_3d, "points_3d"),
        (render_uv, "points_2d_render"),
        (depth, "depth"),
        (source_points, "source_points_3d"),
        (source_render_uv, "source_points_2d_render"),
        (source_depth, "source_depth"),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN or Inf")

    binding = state.get("point_particle_binding")
    if binding is not None:
        if type(binding) is not torch.Tensor or binding.dtype != torch.long:
            raise ValueError("point_particle_binding must be a plain int64 tensor")
        if binding.ndim != 2 or binding.shape[0] != point_count:
            raise ValueError("point_particle_binding must have shape [N, K]")

    camera = state.get("camera")
    if not isinstance(camera, dict) or set(camera) != {"K", "R", "T"}:
        raise ValueError("camera must contain K, R, and T")
    for name, tensor in camera.items():
        if type(tensor) is not torch.Tensor or tensor.shape[0] != frame_count:
            raise ValueError(f"camera {name} must be a plain per-frame tensor")

    paths = state.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("paths metadata is required")
    coarse_paths = paths.get("coarse_rgb_frames")
    if not isinstance(coarse_paths, list) or len(coarse_paths) not in (0, frame_count):
        raise ValueError("coarse_rgb_frames must be empty or match frame count")


def save_transport_ready(state: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and save a portable CPU-tensor transport artifact."""

    validate_transport_ready(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    validate_transport_ready(loaded)
    return path
