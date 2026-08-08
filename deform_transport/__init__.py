"""Minimal geometry-to-latent interfaces built on top of RealWonder."""

from .trajectory import (
    PointTrajectoryRecorder,
    map_image_uv_to_latent,
    project_points_realwonder,
    validate_trajectory_export,
)

__all__ = [
    "PointTrajectoryRecorder",
    "map_image_uv_to_latent",
    "project_points_realwonder",
    "validate_trajectory_export",
]
