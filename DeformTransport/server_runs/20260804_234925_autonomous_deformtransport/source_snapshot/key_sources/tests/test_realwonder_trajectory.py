from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.trajectory import (  # noqa: E402
    PointTrajectoryRecorder,
    map_image_uv_to_latent,
    project_points_realwonder,
    validate_trajectory_export,
)


class RealWonderTrajectoryContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_root = REPO_ROOT / "demo_web" / "demo_data" / "santa_cloth"
        cls.camera_data = torch.load(data_root / "camera.pt", map_location="cpu")
        cls.points = torch.load(
            data_root / "fg_pcs" / "pc_00.pt", map_location="cpu"
        )["points"].to(torch.float32)
        cls.camera = SimpleNamespace(**cls.camera_data)

    def test_bundled_cloth_points_project_into_realwonder_render(self):
        uv, depth, valid = project_points_realwonder(
            self.points,
            self.camera_data["K"],
            self.camera_data["R"],
            self.camera_data["T"],
        )
        self.assertEqual(tuple(uv.shape), (28264, 2))
        self.assertTrue(torch.isfinite(uv).all())
        self.assertTrue((depth > 0).all())
        self.assertGreater(float(valid.float().mean()), 0.99)

    def test_crop_mapping_targets_wan_spatial_latent(self):
        uv, _, render_valid = project_points_realwonder(
            self.points,
            self.camera_data["K"],
            self.camera_data["R"],
            self.camera_data["T"],
        )
        latent_xy, _, crop_valid = map_image_uv_to_latent(uv, crop_start=176)
        valid = render_valid & crop_valid
        self.assertGreater(float(valid.float().mean()), 0.99)
        self.assertTrue((latent_xy[valid, 0] >= 0).all())
        self.assertTrue((latent_xy[valid, 0] < 104).all())
        self.assertTrue((latent_xy[valid, 1] >= 0).all())
        self.assertTrue((latent_xy[valid, 1] < 60).all())

    def test_recorder_preserves_real_point_identity(self):
        recorder = PointTrajectoryRecorder(
            [self.points], material_types=["pbd_cloth"], camera=self.camera
        )
        recorder.record(
            frame_id=0,
            simulation_step=0,
            object_points=[self.points],
            camera=self.camera,
        )
        export = recorder.state_dict()
        validate_trajectory_export(export)
        object_state = export["objects"][0]
        self.assertEqual(tuple(object_state["points_3d"].shape), (1, 28264, 3))
        self.assertTrue(torch.equal(object_state["initial_points_3d"], self.points))

    def test_recorder_strips_runtime_tensor_subclasses(self):
        class RuntimeTensor(torch.Tensor):
            pass

        runtime_points = self.points.as_subclass(RuntimeTensor)
        runtime_binding = torch.zeros(
            (self.points.shape[0], 5), dtype=torch.long
        ).as_subclass(RuntimeTensor)
        recorder = PointTrajectoryRecorder(
            [runtime_points],
            material_types=["pbd_cloth"],
            camera=self.camera,
            binding_indices={0: runtime_binding},
        )
        recorder.record(
            frame_id=0,
            simulation_step=0,
            object_points=[runtime_points],
            camera=self.camera,
        )

        buffer = io.BytesIO()
        torch.save(recorder.state_dict(), buffer)
        buffer.seek(0)
        export = torch.load(buffer, map_location="cpu")
        object_state = export["objects"][0]
        self.assertIs(type(object_state["initial_points_3d"]), torch.Tensor)
        self.assertIs(type(object_state["points_3d"]), torch.Tensor)
        self.assertIs(type(object_state["binding_particle_indices"]), torch.Tensor)

    def test_recorder_supports_mixed_rigid_and_bound_objects(self):
        rigid_points = self.points[:8]
        deformable_points = self.points[8:20]
        binding = torch.arange(60, dtype=torch.long).reshape(12, 5)
        recorder = PointTrajectoryRecorder(
            [rigid_points, deformable_points],
            material_types=["rigid", "mpm_elastic"],
            camera=self.camera,
            binding_indices={1: binding},
        )
        recorder.record(
            frame_id=0,
            simulation_step=5,
            object_points=[rigid_points, deformable_points],
            camera=self.camera,
        )

        export = recorder.state_dict()
        validate_trajectory_export(export)
        self.assertEqual(len(export["objects"]), 2)
        self.assertIsNone(export["objects"][0]["binding_particle_indices"])
        self.assertTrue(
            torch.equal(export["objects"][1]["binding_particle_indices"], binding)
        )
        self.assertEqual(tuple(export["objects"][0]["points_3d"].shape), (1, 8, 3))
        self.assertEqual(tuple(export["objects"][1]["points_3d"].shape), (1, 12, 3))


if __name__ == "__main__":
    unittest.main()
