from __future__ import annotations

from pathlib import Path


def replace_once(
    path: Path,
    old: str,
    new: str,
    *,
    label: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match in "
            f"{path}, found {count}"
        )

    path.write_text(
        text.replace(old, new, 1),
        encoding="utf-8",
    )


# ================================================================
# 1. Extend the residual loader with explicit residual keys.
# ================================================================

integration = Path(
    "deform_transport/pipeline_integration.py"
)

replace_once(
    integration,
    '''_RAW_TRANSPORT_KEYS = {
    "correct": "correct_transported_latent",
    "shuffled": "shuffled_transported_latent",
}
''',
    '''_RAW_TRANSPORT_KEYS = {
    "correct": "correct_transported_latent",
    "shuffled": "shuffled_transported_latent",
}

_EXPLICIT_RESIDUAL_KEYS = {
    "correct": "correct_transport_residual",
    "shuffled": "shuffled_transport_residual",
}
''',
    label="add explicit residual key mapping",
)

replace_once(
    integration,
    '''    residual = (
        transported.to(torch.float32)
        - target.to(torch.float32)
    )

    mask_5d = (
        mask.unsqueeze(0)
        .expand_as(residual)
    )
''',
    '''    explicit_residual_key = (
        _EXPLICIT_RESIDUAL_KEYS[mode]
    )

    explicit_residual = state.get(
        explicit_residual_key
    )

    if explicit_residual is not None:
        if not isinstance(
            explicit_residual,
            torch.Tensor,
        ):
            raise ValueError(
                f"{explicit_residual_key} must be a tensor"
            )

        if tuple(explicit_residual.shape) != tuple(
            reference_latent.shape
        ):
            raise ValueError(
                f"{explicit_residual_key} shape "
                f"{tuple(explicit_residual.shape)} does not "
                "match freshly encoded sim_latent "
                f"{tuple(reference_latent.shape)}"
            )

        if (
            not explicit_residual.dtype.is_floating_point
            or not bool(
                torch.isfinite(
                    explicit_residual
                ).all()
            )
        ):
            raise ValueError(
                f"{explicit_residual_key} must be a "
                "finite floating-point tensor"
            )

        residual = explicit_residual.to(
            torch.float32
        )
    else:
        # Backward-compatible path for historical artifacts.
        residual = (
            transported.to(torch.float32)
            - target.to(torch.float32)
        )

    mask_5d = (
        mask.unsqueeze(0)
        .expand_as(residual)
    )
''',
    label="prefer explicit residual with legacy fallback",
)


# ================================================================
# 2. Save explicit residuals in the confidence artifact.
# ================================================================

builder = Path(
    "scripts/build_confidence_weighted_residual_artifact.py"
)

replace_once(
    builder,
    '''        "correct_transported_latent": (
            weighted_correct.contiguous()
        ),
        "shuffled_transported_latent": (
            weighted_shuffled.contiguous()
        ),
        # These make the artifact compatible with the
''',
    '''        "correct_transported_latent": (
            weighted_correct.contiguous()
        ),
        "shuffled_transported_latent": (
            weighted_shuffled.contiguous()
        ),
        "correct_transport_residual": (
            weighted_correct_residual.contiguous()
        ),
        "shuffled_transport_residual": (
            weighted_shuffled_residual.contiguous()
        ),
        # These make the artifact compatible with the
''',
    label="save explicit weighted residual tensors",
)


# ================================================================
# 3. Add a regression test proving explicit residual precedence.
# ================================================================

tests = Path(
    "tests/test_interstep_transport_residual.py"
)

new_test = '''

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
'''

replace_once(
    tests,
    "\n\nclass ApplyTransportResidualTest",
    new_test
    + "\n\nclass ApplyTransportResidualTest",
    label="add explicit residual precedence test",
)

print("Explicit residual contract fix applied successfully.")
