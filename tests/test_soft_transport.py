from __future__ import annotations

import unittest

import torch

from deform_transport.soft_transport import (
    _bilinear_neighbors,
    soft_point_transport,
)


class SoftPointTransportTest(unittest.TestCase):
    @staticmethod
    def _run(source_grid, source_uv, target_uv, **kwargs):
        point_count = source_uv.shape[0]
        frame_count = target_uv.shape[0]

        return soft_point_transport(
            source_grid=source_grid,
            source_uv=source_uv,
            target_uv=target_uv,
            source_visible=torch.ones(point_count, dtype=torch.bool),
            source_valid=torch.ones(point_count, dtype=torch.bool),
            target_valid=torch.ones(
                frame_count,
                point_count,
                dtype=torch.bool,
            ),
            point_id=torch.arange(point_count),
            **kwargs,
        )

    def test_bilinear_neighbor_weights_sum_to_one_interior(self):
        uv = torch.tensor([[2.25, 3.75]])
        _, weights, valid = _bilinear_neighbors(
            uv,
            height=8,
            width=8,
        )

        self.assertTrue(bool(valid.all()))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

        expected = torch.tensor([[0.1875, 0.0625, 0.5625, 0.1875]])
        self.assertTrue(torch.allclose(weights, expected))

    def test_source_bilinear_sampling_matches_linear_grid(self):
        y = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        grid = (10.0 * y + x).unsqueeze(0)

        source_uv = torch.tensor([[2.25, 3.50]])
        target_uv = source_uv.unsqueeze(0)

        result = self._run(grid, source_uv, target_uv)

        expected_source_value = 10.0 * 3.50 + 2.25

        self.assertAlmostEqual(
            float(result["point_features"][0, 0]),
            expected_source_value,
            places=5,
        )

    def test_target_point_splats_to_four_neighbors(self):
        grid = torch.zeros(1, 6, 8)
        grid[0, 2, 2] = 4.0

        source_uv = torch.tensor([[2.0, 2.0]])
        target_uv = torch.tensor([[[3.25, 2.50]]])

        result = self._run(grid, source_uv, target_uv)

        mask = result["transport_mask"][0, 0]
        weights = result["transport_weight"][0, 0]

        self.assertEqual(int(mask.sum()), 4)
        self.assertAlmostEqual(float(weights[2, 3]), 0.375, places=6)
        self.assertAlmostEqual(float(weights[2, 4]), 0.125, places=6)
        self.assertAlmostEqual(float(weights[3, 3]), 0.375, places=6)
        self.assertAlmostEqual(float(weights[3, 4]), 0.125, places=6)

        output = result["transported_grid"][0, 0]

        for y, x in [(2, 3), (2, 4), (3, 3), (3, 4)]:
            self.assertAlmostEqual(float(output[y, x]), 4.0, places=6)

    def test_integer_coordinate_matches_single_cell_behavior(self):
        grid = torch.arange(
            1 * 5 * 8,
            dtype=torch.float32,
        ).reshape(1, 5, 8)

        source_uv = torch.tensor([[2.0, 2.0]])
        target_uv = torch.tensor([[[5.0, 3.0]]])

        result = self._run(grid, source_uv, target_uv)

        self.assertEqual(int(result["transport_mask"].sum()), 1)
        self.assertEqual(
            float(result["transported_grid"][0, 0, 3, 5]),
            float(grid[0, 2, 2]),
        )
        self.assertAlmostEqual(
            float(result["transport_weight"][0, 0, 3, 5]),
            1.0,
            places=6,
        )

    def test_collision_is_weight_normalized(self):
        grid = torch.zeros(1, 5, 8)
        grid[0, 1, 1] = 2.0
        grid[0, 1, 2] = 6.0

        source_uv = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        target_uv = torch.tensor([[[4.0, 3.0], [4.0, 3.0]]])

        result = self._run(grid, source_uv, target_uv)

        self.assertAlmostEqual(
            float(result["transported_grid"][0, 0, 3, 4]),
            4.0,
            places=6,
        )
        self.assertEqual(
            int(result["contribution_count"][0, 0, 3, 4]),
            2,
        )
        self.assertAlmostEqual(
            float(result["transport_weight"][0, 0, 3, 4]),
            2.0,
            places=6,
        )

    def test_boundary_renormalization_does_not_pollute_outside(self):
        grid = torch.ones(1, 5, 8)

        source_uv = torch.tensor([[1.0, 1.0]])
        target_uv = torch.tensor([[[7.75, 2.25]]])

        result = self._run(grid, source_uv, target_uv)

        self.assertEqual(int(result["transport_mask"].sum()), 2)
        self.assertTrue(
            torch.allclose(
                result["transported_grid"][
                    result["transport_mask"].expand_as(
                        result["transported_grid"]
                    )
                ],
                torch.ones(
                    int(result["transport_mask"].sum()),
                    dtype=grid.dtype,
                ),
            )
        )

    def test_correct_and_shuffled_have_identical_spatial_contract(self):
        y = torch.arange(5, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        grid = (10.0 * y + x).unsqueeze(0)

        source_uv = torch.tensor(
            [[0.2 + x, 2.25] for x in range(7)],
            dtype=torch.float32,
        )
        target_uv = (
            source_uv + torch.tensor([0.35, 0.20])
        ).unsqueeze(0)

        correct = self._run(
            grid,
            source_uv,
            target_uv,
            mode="correct",
            seed=0,
        )
        shuffled = self._run(
            grid,
            source_uv,
            target_uv,
            mode="shuffled",
            seed=0,
        )

        self.assertTrue(
            torch.equal(
                correct["transport_mask"],
                shuffled["transport_mask"],
            )
        )
        self.assertTrue(
            torch.equal(
                correct["contribution_count"],
                shuffled["contribution_count"],
            )
        )
        self.assertTrue(
            torch.allclose(
                correct["transport_weight"],
                shuffled["transport_weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                correct["valid_point_mask"],
                shuffled["valid_point_mask"],
            )
        )
        self.assertFalse(
            torch.equal(
                correct["transported_grid"],
                shuffled["transported_grid"],
            )
        )

    def test_nonfinite_and_out_of_frame_points_are_dropped(self):
        grid = torch.ones(1, 5, 8)

        source_uv = torch.tensor(
            [
                [1.0, 1.0],
                [float("nan"), 2.0],
                [-0.1, 3.0],
            ]
        )
        target_uv = torch.tensor(
            [
                [
                    [2.25, 2.25],
                    [3.0, 2.0],
                    [4.0, 3.0],
                ]
            ]
        )

        result = self._run(grid, source_uv, target_uv)

        self.assertEqual(int(result["valid_point_mask"].sum()), 1)
        self.assertTrue(torch.isfinite(result["transported_grid"]).all())
        self.assertTrue(torch.isfinite(result["transport_weight"]).all())

    def test_mask_matches_positive_weight(self):
        grid = torch.ones(2, 5, 8)

        source_uv = torch.tensor([[1.25, 1.25], [3.50, 2.50]])
        target_uv = torch.tensor(
            [
                [[2.25, 2.25], [4.50, 3.25]],
                [[3.00, 2.00], [5.25, 3.50]],
            ]
        )

        result = self._run(grid, source_uv, target_uv)

        self.assertTrue(
            torch.equal(
                result["transport_mask"],
                result["transport_weight"] > result["eps"],
            )
        )
        self.assertTrue(
            bool((result["contribution_count"] >= 0).all())
        )


if __name__ == "__main__":
    unittest.main()
