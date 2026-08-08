from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.trajectory import map_image_uv_to_latent  # noqa: E402
from deform_transport.transport_ready import (  # noqa: E402
    build_transport_ready,
    save_transport_ready,
    validate_transport_ready,
)


class TransportReadyContractTest(unittest.TestCase):
    def _trajectory(self, tensor_subclass=None):
        def maybe_subclass(tensor):
            return tensor if tensor_subclass is None else tensor.as_subclass(tensor_subclass)

        frame_ids = torch.tensor([0, 1], dtype=torch.long)
        simulation_steps = torch.tensor([5, 10], dtype=torch.long)
        source_uv = [
            torch.tensor([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]),
            torch.tensor([[400.0, 400.0], [250.0, 250.0]]),
        ]
        future_uv = [
            torch.stack(
                [
                    source_uv[0] + torch.tensor([8.0, 0.0]),
                    source_uv[0] + torch.tensor([16.0, 0.0]),
                ]
            ),
            torch.stack(
                [
                    source_uv[1] + torch.tensor([0.0, -8.0]),
                    torch.tensor([[400.0, 50.0], [250.0, 242.0]]),
                ]
            ),
        ]
        objects = []
        for object_id, (initial_uv, points_uv) in enumerate(
            zip(source_uv, future_uv)
        ):
            point_count = initial_uv.shape[0]
            initial_points = torch.stack(
                [
                    torch.linspace(0.0, 1.0, point_count),
                    torch.full((point_count,), float(object_id)),
                    torch.ones(point_count),
                ],
                dim=1,
            )
            points_3d = torch.stack(
                [initial_points, initial_points + torch.tensor([0.1, 0.0, 0.0])]
            )
            binding = torch.arange(point_count * 5, dtype=torch.long).reshape(
                point_count, 5
            )
            objects.append(
                {
                    "object_id": object_id,
                    "material_type": "pbd_cloth" if object_id == 0 else "mpm_elastic",
                    "initial_points_3d": maybe_subclass(initial_points),
                    "initial_points_uv": maybe_subclass(initial_uv),
                    "initial_depth": maybe_subclass(torch.ones(point_count)),
                    "initial_projection_valid": maybe_subclass(
                        torch.ones(point_count, dtype=torch.bool)
                    ),
                    "points_3d": maybe_subclass(points_3d),
                    "points_uv": maybe_subclass(points_uv),
                    "depth": maybe_subclass(torch.ones(2, point_count)),
                    "projection_valid": maybe_subclass(
                        torch.ones(2, point_count, dtype=torch.bool)
                    ),
                    "binding_particle_indices": maybe_subclass(binding),
                }
            )
        return {
            "format_version": 1,
            "coordinate_system": "synthetic_test_world_and_realwonder_512_uv",
            "image_size": 512,
            "frame_ids": maybe_subclass(frame_ids),
            "simulation_steps": maybe_subclass(simulation_steps),
            "camera": {
                "K": maybe_subclass(torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 1, 1, 1)),
                "R": maybe_subclass(torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 1, 1, 1)),
                "T": maybe_subclass(torch.zeros(2, 1, 3)),
            },
            "objects": objects,
        }

    @staticmethod
    def _source_raster():
        raster = torch.full((2, 512, 512), -1, dtype=torch.long)
        raster[0, 100, 100] = 0  # Render-valid but above the 480 crop.
        raster[0, 200, 200] = 1
        raster[0, 300, 300] = 2
        raster[0, 250, 250] = 4
        return raster

    def test_source_visibility_is_raster_selected_and_crop_valid(self):
        state = build_transport_ready(
            self._trajectory(), self._source_raster(), case_name="synthetic"
        )
        self.assertEqual(tuple(state["source_visible"].shape), (5,))
        self.assertEqual(state["source_raster_visible_point_ids"].tolist(), [0, 1, 2, 4])
        self.assertEqual(state["source_visible_point_ids"].tolist(), [1, 2, 4])
        self.assertFalse(bool(state["source_valid"][0]))
        self.assertTrue(bool((state["source_visible"] <= state["source_valid"]).all()))

    def test_latent_mapping_and_crop_validity_reuse_verified_mapping(self):
        trajectory = self._trajectory()
        state = build_transport_ready(
            trajectory, self._source_raster(), case_name="synthetic"
        )
        expected_source, _, _ = map_image_uv_to_latent(
            torch.cat([obj["initial_points_uv"] for obj in trajectory["objects"]])
        )
        expected_future, _, _ = map_image_uv_to_latent(
            torch.cat([obj["points_uv"] for obj in trajectory["objects"]], dim=1)
            .reshape(-1, 2)
        )
        self.assertTrue(
            torch.equal(state["source_points_2d_latent"], expected_source)
        )
        self.assertTrue(
            torch.equal(state["points_2d_latent"].reshape(-1, 2), expected_future)
        )
        self.assertFalse(bool(state["projection_valid"][1, 3]))
        self.assertLess(int(state["points_2d_latent"][1, 3, 1]), 0)

    def test_flattened_point_order_and_bindings_are_preserved(self):
        trajectory = self._trajectory()
        state = build_transport_ready(
            trajectory, self._source_raster(), case_name="synthetic"
        )
        self.assertEqual(state["point_id"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(state["object_id"].tolist(), [0, 0, 0, 1, 1])
        expected_points = torch.cat(
            [obj["points_3d"] for obj in trajectory["objects"]], dim=1
        )
        expected_binding = torch.cat(
            [obj["binding_particle_indices"] for obj in trajectory["objects"]]
        )
        self.assertTrue(torch.equal(state["points_3d"], expected_points))
        self.assertTrue(
            torch.equal(state["point_particle_binding"], expected_binding)
        )

    def test_save_load_round_trip_strips_tensor_subclasses(self):
        class RuntimeTensor(torch.Tensor):
            pass

        state = build_transport_ready(
            self._trajectory(RuntimeTensor),
            self._source_raster().as_subclass(RuntimeTensor),
            case_name="synthetic",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_transport_ready(state, Path(directory) / "transport_ready.pt")
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        validate_transport_ready(loaded)
        self.assertIs(type(loaded["points_3d"]), torch.Tensor)
        self.assertIs(type(loaded["points_2d_latent"]), torch.Tensor)
        self.assertIs(type(loaded["point_particle_binding"]), torch.Tensor)
        self.assertIs(type(loaded["camera"]["K"]), torch.Tensor)

    def test_out_of_range_raster_point_id_is_rejected(self):
        raster = self._source_raster()
        raster[0, 0, 0] = 5
        with self.assertRaisesRegex(ValueError, "outside the trajectory"):
            build_transport_ready(
                self._trajectory(), raster, case_name="synthetic"
            )


if __name__ == "__main__":
    unittest.main()
