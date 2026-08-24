import tempfile
import unittest
from pathlib import Path

import torch

from deform_transport.pipeline_integration import load_precomputed_transport_latent


class PipelineIntegrationTest(unittest.TestCase):
    def _write_artifact(self, path: Path, *, shape=(1, 2, 16, 3, 4)):
        correct = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
        shuffled = correct.flip(2)
        flow = correct.flip(3)
        blend = correct.mul(0.5)
        mask = torch.ones((shape[1], 1, shape[3], shape[4]), dtype=torch.bool)
        count = torch.ones_like(mask, dtype=torch.long)
        torch.save(
            {
                "format_version": 1,
                "correct_fused_latent": correct,
                "shuffled_fused_latent": shuffled,
                "flow_fused_latent": flow,
                "blend_fused_latent": blend,
                "transport_mask": mask,
                "contribution_count": count,
            },
            path,
        )
        return correct, shuffled

    def test_selects_correct_and_shuffled_without_changing_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transport.pt"
            correct, shuffled = self._write_artifact(path)
            reference = torch.zeros_like(correct, dtype=torch.bfloat16)
            loaded_correct = load_precomputed_transport_latent(
                path, mode="correct", reference_latent=reference
            )
            loaded_shuffled = load_precomputed_transport_latent(
                path, mode="shuffled", reference_latent=reference
            )
            loaded_flow = load_precomputed_transport_latent(
                path, mode="flow", reference_latent=reference
            )
            loaded_blend = load_precomputed_transport_latent(
                path, mode="blend", reference_latent=reference
            )
            self.assertEqual(loaded_correct.dtype, reference.dtype)
            self.assertEqual(loaded_correct.device, reference.device)
            self.assertTrue(torch.equal(loaded_correct, correct.to(torch.bfloat16)))
            self.assertTrue(torch.equal(loaded_shuffled, shuffled.to(torch.bfloat16)))
            self.assertFalse(torch.equal(loaded_correct, loaded_shuffled))
            self.assertTrue(torch.equal(loaded_flow, correct.flip(3).to(torch.bfloat16)))
            self.assertTrue(torch.equal(loaded_blend, correct.mul(0.5).to(torch.bfloat16)))

    def test_rejects_case_or_resolution_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transport.pt"
            self._write_artifact(path)
            reference = torch.zeros((1, 3, 16, 3, 4))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_precomputed_transport_latent(
                    path, mode="correct", reference_latent=reference
                )

    def test_rejects_nonfinite_or_inconsistent_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transport.pt"
            correct, shuffled = self._write_artifact(path)
            correct[0, 0, 0, 0, 0] = torch.nan
            mask = torch.ones((2, 1, 3, 4), dtype=torch.bool)
            torch.save(
                {
                    "correct_fused_latent": correct,
                    "shuffled_fused_latent": shuffled,
                    "transport_mask": mask,
                    "contribution_count": torch.zeros_like(mask, dtype=torch.long),
                },
                path,
            )
            reference = torch.zeros_like(correct)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_precomputed_transport_latent(
                    path, mode="correct", reference_latent=reference
                )


if __name__ == "__main__":
    unittest.main()
