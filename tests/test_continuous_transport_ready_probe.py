from __future__ import annotations

import copy
import unittest

import torch

from scripts.build_continuous_transport_ready_probe import (
    validate_continuous_extension,
)


class ContinuousTransportReadyProbeTest(unittest.TestCase):
    @staticmethod
    def _state() -> dict:
        video_height = 480
        video_width = 832
        latent_height = 60
        latent_width = 104

        source_video = torch.tensor(
            [
                [273.80859375, 239.18759155273438],
                [275.3367919921875, 234.30838012695312],
            ],
            dtype=torch.float32,
        )

        target_video = torch.tensor(
            [
                [
                    [275.87518310546875, 240.0660400390625],
                    [277.2738952636719, 234.90899658203125],
                ],
                [
                    [278.5, 242.25],
                    [280.75, 237.5],
                ],
            ],
            dtype=torch.float32,
        )

        scale = torch.tensor(
            [
                latent_width / video_width,
                latent_height / video_height,
            ],
            dtype=torch.float32,
        )

        source_continuous = source_video * scale
        target_continuous = target_video * scale

        point_count = source_video.shape[0]
        frame_count = target_video.shape[0]

        state = {
            "format_version": 1,
            "case_name": "synthetic",
            "coordinate_system": {
                "points_3d": "synthetic",
                "points_2d_render": "synthetic",
                "points_2d_video": "synthetic",
                "points_2d_latent": "wan_spatial_cell_xy_floor_legacy",
                "points_2d_latent_continuous":
                    "wan_spatial_xy_continuous_legacy_scale",
            },
            "frame_ids": torch.arange(frame_count, dtype=torch.long),
            "simulation_steps": torch.tensor([10, 20], dtype=torch.long),
            "point_id": torch.arange(point_count, dtype=torch.long),
            "object_id": torch.zeros(point_count, dtype=torch.long),
            "material_type": ["cloth"],
            "object_point_ranges": [
                {
                    "object_id": 0,
                    "start": 0,
                    "end": point_count,
                }
            ],
            "points_3d": torch.zeros(
                frame_count,
                point_count,
                3,
                dtype=torch.float32,
            ),
            "points_2d_render": target_video / (832.0 / 512.0)
                + torch.tensor([0.0, 176.0 / (832.0 / 512.0)]),
            "points_2d_video": target_video,
            "points_2d_latent": torch.floor(
                target_continuous
            ).to(torch.long),
            "points_2d_latent_continuous": target_continuous,
            "depth": torch.ones(
                frame_count,
                point_count,
                dtype=torch.float32,
            ),
            "render_projection_valid": torch.ones(
                frame_count,
                point_count,
                dtype=torch.bool,
            ),
            "projection_valid": torch.ones(
                frame_count,
                point_count,
                dtype=torch.bool,
            ),
            "source_points_3d": torch.zeros(
                point_count,
                3,
                dtype=torch.float32,
            ),
            "source_points_2d_render": source_video / (832.0 / 512.0)
                + torch.tensor([0.0, 176.0 / (832.0 / 512.0)]),
            "source_points_2d_video": source_video,
            "source_points_2d_latent": torch.floor(
                source_continuous
            ).to(torch.long),
            "source_points_2d_latent_continuous": source_continuous,
            "source_depth": torch.ones(
                point_count,
                dtype=torch.float32,
            ),
            "source_render_projection_valid": torch.ones(
                point_count,
                dtype=torch.bool,
            ),
            "source_valid": torch.ones(
                point_count,
                dtype=torch.bool,
            ),
            "source_visible": torch.ones(
                point_count,
                dtype=torch.bool,
            ),
            "source_visible_point_ids": torch.arange(
                point_count,
                dtype=torch.long,
            ),
            "source_raster_visible_point_ids": torch.arange(
                point_count,
                dtype=torch.long,
            ),
            "point_particle_binding": None,
            "point_particle_binding_by_object": [None],
            "camera": {
                "K": torch.zeros(
                    frame_count,
                    4,
                    4,
                    dtype=torch.float32,
                ),
                "R": torch.zeros(
                    frame_count,
                    3,
                    3,
                    dtype=torch.float32,
                ),
                "T": torch.zeros(
                    frame_count,
                    3,
                    dtype=torch.float32,
                ),
            },
            "render_height": 512,
            "render_width": 512,
            "video_height": video_height,
            "video_width": video_width,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "paths": {
                "source_trajectory": None,
                "initial_rgb": None,
                "coarse_rgb_frames": [],
                "flow": None,
                "source_raster_point_indices": None,
            },
            "continuous_coordinate_extension_version": 1,
            "continuous_coordinate_mapping": {
                "source": "source_points_2d_video * scale_xy",
                "target": "points_2d_video * scale_xy",
                "scale_xy": [0.125, 0.125],
                "legacy_recovery":
                    "floor(continuous) == legacy_discrete",
                "pixel_center_offset": False,
                "align_corners": None,
            },
        }

        return state

    def test_continuous_coordinates_recover_legacy_cells(self):
        state = self._state()
        summary = validate_continuous_extension(state)

        self.assertTrue(summary["source_floor_recovers_legacy"])
        self.assertTrue(summary["target_floor_recovers_legacy"])
        self.assertGreater(summary["source_noninteger_points"], 0)
        self.assertGreater(summary["target_noninteger_points"], 0)

    def test_fractional_information_cannot_be_removed(self):
        state = self._state()

        state["source_points_2d_latent_continuous"] = (
            state["source_points_2d_latent"].to(torch.float32)
        )

        with self.assertRaisesRegex(
            ValueError,
            "source continuous coordinates do not match",
        ):
            validate_continuous_extension(state)

    def test_floor_incompatibility_is_rejected(self):
        state = self._state()

        broken = copy.deepcopy(state)

        broken[
            "points_2d_latent_continuous"
        ][0, 0, 0] += 1.0

        with self.assertRaises(ValueError):
            validate_continuous_extension(broken)


if __name__ == "__main__":
    unittest.main()
