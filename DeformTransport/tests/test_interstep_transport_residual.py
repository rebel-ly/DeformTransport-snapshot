from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from deform_transport.pipeline_integration import (
    load_precomputed_transport_residual,
)
from vidgen.pipeline_sdedit import (
    apply_transport_residual,
)


class TransportResidualLoaderTest(unittest.TestCase):
    def build_artifact(self, path: Path) -> None:
        target = torch.zeros(
            1,
            2,
            3,
            2,
            2,
            dtype=torch.float32,
        )

        correct = target.clone()
        shuffled = target.clone()

        correct[:, :, :, 0, 0] = 2.0
        shuffled[:, :, :, 0, 0] = -3.0

        # These values must be removed by masking.
        correct[:, :, :, 1, 1] = 100.0
        shuffled[:, :, :, 1, 1] = -100.0

        mask = torch.zeros(
            2,
            1,
            2,
            2,
            dtype=torch.bool,
        )
        mask[:, :, 0, 0] = True

        count = mask.to(torch.int64)

        torch.save(
            {
                "target_latent": target,
                "correct_transported_latent": correct,
                "shuffled_transported_latent": shuffled,
                "transport_mask": mask,
                "contribution_count": count,
            },
            path,
        )

    def test_correct_residual_is_artifact_local_and_masked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.pt"
            self.build_artifact(artifact)

            # Deliberately different from artifact target.
            reference = torch.full(
                (1, 2, 3, 2, 2),
                77.0,
                dtype=torch.float32,
            )

            residual = load_precomputed_transport_residual(
                artifact,
                mode="correct",
                reference_latent=reference,
            )

            expected = torch.zeros_like(reference)
            expected[:, :, :, 0, 0] = 2.0

            self.assertTrue(
                torch.equal(residual, expected)
            )

    def test_shuffled_residual_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.pt"
            self.build_artifact(artifact)

            reference = torch.zeros(
                1,
                2,
                3,
                2,
                2,
            )

            correct = load_precomputed_transport_residual(
                artifact,
                mode="correct",
                reference_latent=reference,
            )

            shuffled = load_precomputed_transport_residual(
                artifact,
                mode="shuffled",
                reference_latent=reference,
            )

            self.assertFalse(
                torch.equal(correct, shuffled)
            )


    def test_explicit_residual_is_preferred(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.pt"
            self.build_artifact(artifact)

            state = torch.load(
                artifact,
                map_location="cpu",
                weights_only=True,
            )

            explicit = torch.zeros_like(
                state["target_latent"]
            )
            explicit[:, :, :, 0, 0] = 5.0

            state[
                "correct_transport_residual"
            ] = explicit

            torch.save(state, artifact)

            reference = torch.full(
                (1, 2, 3, 2, 2),
                77.0,
                dtype=torch.float32,
            )

            residual = (
                load_precomputed_transport_residual(
                    artifact,
                    mode="correct",
                    reference_latent=reference,
                )
            )

            self.assertTrue(
                torch.equal(
                    residual,
                    explicit,
                )
            )


class ApplyTransportResidualTest(unittest.TestCase):
    def test_scale_zero_is_exact_noop(self) -> None:
        clean = torch.randn(
            1,
            2,
            3,
            2,
            2,
            dtype=torch.bfloat16,
        )
        residual = torch.randn_like(clean)

        result = apply_transport_residual(
            clean,
            residual,
            0.0,
        )

        self.assertTrue(torch.equal(clean, result))

    def test_scaled_residual(self) -> None:
        clean = torch.zeros(
            1,
            1,
            1,
            1,
            1,
            dtype=torch.float32,
        )
        residual = torch.full_like(clean, 4.0)

        result = apply_transport_residual(
            clean,
            residual,
            0.25,
        )

        self.assertTrue(
            torch.equal(
                result,
                torch.ones_like(result),
            )
        )


if __name__ == "__main__":
    unittest.main()
