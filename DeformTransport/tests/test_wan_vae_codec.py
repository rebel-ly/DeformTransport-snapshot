from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.wan_vae_codec import (  # noqa: E402
    causal_latent_frame_end_indices,
)


class WanVAEContractTest(unittest.TestCase):
    def test_single_frame_maps_to_one_latent_slot(self):
        self.assertTrue(
            torch.equal(causal_latent_frame_end_indices(1), torch.tensor([0]))
        )

    def test_twenty_one_frames_map_to_actual_causal_chunk_ends(self):
        self.assertEqual(
            causal_latent_frame_end_indices(21).tolist(), [0, 4, 8, 12, 16, 20]
        )

    def test_invalid_frame_count_is_rejected(self):
        with self.assertRaises(ValueError):
            causal_latent_frame_end_indices(0)


if __name__ == "__main__":
    unittest.main()
