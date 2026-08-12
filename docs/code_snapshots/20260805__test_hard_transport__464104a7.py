from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.hard_transport import (  # noqa: E402
    hard_point_transport,
    make_objectwise_feature_permutation,
)


class HardPointTransportTest(unittest.TestCase):
    @staticmethod
    def _grid(height=5, width=8):
        y = torch.arange(height, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(width, dtype=torch.float32).unsqueeze(0)
        return (10.0 * y + x).unsqueeze(0)

    @staticmethod
    def _run(source_grid, source_uv, target_uv, **kwargs):
        point_count = source_uv.shape[0]
        frame_count = target_uv.shape[0]
        return hard_point_transport(
            source_grid,
            source_uv,
            target_uv,
            torch.ones(point_count, dtype=torch.bool),
            torch.ones(point_count, dtype=torch.bool),
            torch.ones(frame_count, point_count, dtype=torch.bool),
            torch.arange(point_count),
            **kwargs,
        )

    def test_identity_transport_restores_occupied_source_features(self):
        grid = self._grid()
        source_uv = torch.tensor([[1, 1], [4, 2], [2, 3]])
        result = self._run(grid, source_uv, source_uv.unsqueeze(0))
        output = result["transported_grid"][0, 0]
        for x, y in source_uv.tolist():
            self.assertAlmostEqual(float(output[y, x]), float(grid[0, y, x]))

    def test_horizontal_translation_moves_features_right(self):
        grid = self._grid()
        source_uv = torch.tensor([[1, 1], [2, 2], [3, 3]])
        target_uv = (source_uv + torch.tensor([2, 0])).unsqueeze(0)
        result = self._run(grid, source_uv, target_uv)
        for (source_x, source_y), (target_x, target_y) in zip(
            source_uv.tolist(), target_uv[0].tolist()
        ):
            self.assertEqual(
                float(result["transported_grid"][0, 0, target_y, target_x]),
                float(grid[0, source_y, source_x]),
            )

    def test_vertical_translation_does_not_swap_xy_or_hw(self):
        grid = self._grid(height=6, width=8)
        source_uv = torch.tensor([[1, 1], [5, 2]])
        target_uv = (source_uv + torch.tensor([0, 2])).unsqueeze(0)
        result = self._run(grid, source_uv, target_uv)
        self.assertEqual(
            float(result["transported_grid"][0, 0, 3, 1]),
            float(grid[0, 1, 1]),
        )
        self.assertEqual(
            float(result["transported_grid"][0, 0, 4, 5]),
            float(grid[0, 2, 5]),
        )

    def test_points_sharing_a_source_cell_can_diverge(self):
        grid = self._grid()
        source_uv = torch.tensor([[2, 2], [2, 2]])
        target_uv = torch.tensor([[[1, 3], [6, 1]]])
        result = self._run(grid, source_uv, target_uv)
        expected = float(grid[0, 2, 2])
        self.assertEqual(float(result["transported_grid"][0, 0, 3, 1]), expected)
        self.assertEqual(float(result["transported_grid"][0, 0, 1, 6]), expected)
        self.assertEqual(int(result["valid_point_mask"].sum()), 2)

    def test_compression_averages_contributions(self):
        grid = torch.zeros(1, 5, 8)
        grid[0, 1, 1] = 1.0
        grid[0, 1, 2] = 3.0
        source_uv = torch.tensor([[1, 1], [2, 1]])
        target_uv = torch.tensor([[[4, 3], [4, 3]]])
        result = self._run(grid, source_uv, target_uv)
        self.assertEqual(float(result["transported_grid"][0, 0, 3, 4]), 2.0)
        self.assertEqual(int(result["contribution_count"][0, 0, 3, 4]), 2)

    def test_out_of_frame_points_are_dropped_without_boundary_pollution(self):
        grid = self._grid()
        source_uv = torch.tensor([[1.0, 1.0], [-0.1, 2.0], [3.0, 3.0], [4.0, 2.0]])
        target_uv = torch.tensor(
            [[[1.0, 1.0], [0.0, 2.0], [-0.1, 3.0], [8.0, 2.0]]]
        )
        result = self._run(grid, source_uv, target_uv)
        self.assertEqual(int(result["valid_point_mask"].sum()), 1)
        self.assertEqual(int(result["transport_mask"].sum()), 1)
        self.assertFalse(bool(result["transport_mask"][0, 0, 3, 0]))
        self.assertFalse(bool(result["transport_mask"][0, 0, 2, 7]))

    def test_correct_and_shuffled_are_different_on_nonuniform_grid(self):
        grid = self._grid()
        source_uv = torch.tensor([[x, 2] for x in range(8)])
        target_uv = source_uv.unsqueeze(0)
        correct = self._run(grid, source_uv, target_uv, mode="correct", seed=0)
        shuffled = self._run(grid, source_uv, target_uv, mode="shuffled", seed=0)
        self.assertFalse(torch.equal(shuffled["permutation"], torch.arange(8)))
        self.assertFalse(
            torch.equal(correct["transported_grid"], shuffled["transported_grid"])
        )
        self.assertTrue(
            torch.equal(correct["transport_mask"], shuffled["transport_mask"])
        )

    def test_shuffle_is_reproducible_and_stays_within_each_object(self):
        eligible = torch.ones(8, dtype=torch.bool)
        object_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        first = make_objectwise_feature_permutation(eligible, object_id, seed=7)
        repeated = make_objectwise_feature_permutation(eligible, object_id, seed=7)
        different = make_objectwise_feature_permutation(eligible, object_id, seed=8)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))
        self.assertTrue(bool((first[:4] < 4).all()))
        self.assertTrue(bool((first[4:] >= 4).all()))

    def test_numerical_safety_count_and_mask_contract(self):
        grid = self._grid()
        source_uv = torch.tensor([[1, 1], [2, 1], [3, 1]])
        target_uv = torch.tensor(
            [[[4, 2], [4, 2], [4, 2]], [[2, 3], [3, 3], [4, 3]]]
        )
        result = self._run(grid, source_uv, target_uv)
        self.assertTrue(torch.isfinite(result["transported_grid"]).all())
        self.assertEqual(result["contribution_count"].dtype, torch.long)
        self.assertTrue(bool((result["contribution_count"] >= 0).all()))
        self.assertTrue(
            torch.equal(
                result["transport_mask"], result["contribution_count"] > 0
            )
        )


if __name__ == "__main__":
    unittest.main()

