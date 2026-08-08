from __future__ import annotations

import unittest

import torch

from deform_transport.visibility_contract import (
    select_target_validity,
)


class VisibilityContractTest(unittest.TestCase):
    @staticmethod
    def _state():
        return {
            "projection_valid": torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, False],
                    [True, True, False, False],
                ],
                dtype=torch.bool,
            ),
            "source_visible": torch.tensor(
                [True, True, False, False],
                dtype=torch.bool,
            ),
            "point_id": torch.arange(4),
            "frame_ids": torch.arange(3),
            "simulation_steps": torch.tensor(
                [0, 10, 20]
            ),
        }

    @staticmethod
    def _visibility():
        return {
            "aligned_visible": torch.tensor(
                [
                    [True, True, False, False],
                    [True, False, False, False],
                    [False, True, False, False],
                ],
                dtype=torch.bool,
            ),
            "point_id": torch.arange(4),
            "frame_ids": torch.arange(3),
            "simulation_steps": torch.tensor(
                [0, 10, 20]
            ),
        }

    def test_projection_only_preserves_projection_contract(self):
        result = select_target_validity(
            state=self._state(),
            latent_indices=torch.tensor([0, 2]),
            visibility_state=None,
            mode="projection_only",
        )

        expected = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ],
            dtype=torch.bool,
        )

        self.assertTrue(
            torch.equal(
                result["selected_target_valid"],
                expected,
            )
        )

        self.assertIsNone(
            result["selected_future_visible"]
        )

    def test_future_visible_ands_projection_and_raster(self):
        result = select_target_validity(
            state=self._state(),
            latent_indices=torch.tensor([0, 1, 2]),
            visibility_state=self._visibility(),
            mode="source_and_future_visible",
        )

        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, False, False, False],
                [False, True, False, False],
            ],
            dtype=torch.bool,
        )

        self.assertTrue(
            torch.equal(
                result["selected_target_valid"],
                expected,
            )
        )

    def test_future_visible_requires_visibility_artifact(self):
        with self.assertRaises(ValueError):
            select_target_validity(
                state=self._state(),
                latent_indices=torch.tensor([0]),
                visibility_state=None,
                mode="source_and_future_visible",
            )

    def test_visibility_shape_mismatch_is_rejected(self):
        visibility = self._visibility()
        visibility["aligned_visible"] = torch.ones(
            2,
            4,
            dtype=torch.bool,
        )

        with self.assertRaises(ValueError):
            select_target_validity(
                state=self._state(),
                latent_indices=torch.tensor([0]),
                visibility_state=visibility,
                mode="source_and_future_visible",
            )

    def test_frame_zero_must_reproduce_source_visibility(self):
        visibility = self._visibility()
        visibility["aligned_visible"][0, 2] = True

        with self.assertRaises(ValueError):
            select_target_validity(
                state=self._state(),
                latent_indices=torch.tensor([0]),
                visibility_state=visibility,
                mode="source_and_future_visible",
            )

    def test_visibility_must_be_projection_subset(self):
        state = self._state()
        visibility = self._visibility()

        visibility["aligned_visible"][2, 2] = True

        with self.assertRaises(ValueError):
            select_target_validity(
                state=state,
                latent_indices=torch.tensor([2]),
                visibility_state=visibility,
                mode="source_and_future_visible",
            )

    def test_temporal_contract_mismatch_is_rejected(self):
        visibility = self._visibility()
        visibility["simulation_steps"] = torch.tensor(
            [0, 11, 20]
        )

        with self.assertRaises(ValueError):
            select_target_validity(
                state=self._state(),
                latent_indices=torch.tensor([1]),
                visibility_state=visibility,
                mode="source_and_future_visible",
            )


if __name__ == "__main__":
    unittest.main()
