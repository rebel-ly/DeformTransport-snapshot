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
            f"{label}: expected exactly one match in {path}, "
            f"found {count}"
        )

    path.write_text(
        text.replace(old, new, 1),
        encoding="utf-8",
    )


# ------------------------------------------------------------------
# 1. deform_transport/pipeline_integration.py
# ------------------------------------------------------------------

pipeline_integration = Path(
    "deform_transport/pipeline_integration.py"
)

replace_once(
    pipeline_integration,
    '''_LATENT_KEYS = {
    "correct": "correct_fused_latent",
    "shuffled": "shuffled_fused_latent",
    "flow": "flow_fused_latent",
    "blend": "blend_fused_latent",
}
''',
    '''_LATENT_KEYS = {
    "correct": "correct_fused_latent",
    "shuffled": "shuffled_fused_latent",
    "flow": "flow_fused_latent",
    "blend": "blend_fused_latent",
}

_RAW_TRANSPORT_KEYS = {
    "correct": "correct_transported_latent",
    "shuffled": "shuffled_transported_latent",
}
''',
    label="add raw transport keys",
)

residual_loader = r'''

def load_precomputed_transport_residual(
    artifact_path: str | Path,
    *,
    mode: TransportCondition,
    reference_latent: torch.Tensor,
) -> torch.Tensor:
    """Load an artifact-local masked transport residual.

    The residual is computed entirely inside the artifact:

        transported_latent - artifact_target_latent

    It is then moved to the device and dtype of the freshly encoded
    RealWonder reference latent. The artifact target is deliberately not
    assumed to be numerically identical to the runtime VAE encoding.
    """

    if mode not in _RAW_TRANSPORT_KEYS:
        raise ValueError(
            "residual transport currently supports only "
            f"'correct' and 'shuffled', received: {mode}"
        )

    if reference_latent.ndim != 5:
        raise ValueError(
            "reference_latent must have shape [B,T,C,H,W]"
        )

    path = Path(artifact_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, dict):
        raise ValueError(
            "transport artifact must contain a dictionary"
        )

    raw_key = _RAW_TRANSPORT_KEYS[mode]

    transported = state.get(raw_key)
    target = state.get("target_latent")

    for tensor, label in (
        (transported, raw_key),
        (target, "target_latent"),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"transport artifact is missing tensor {label!r}"
            )

        if tensor.ndim != 5:
            raise ValueError(
                f"{label} must have shape [B,T,C,H,W]"
            )

        if tuple(tensor.shape) != tuple(
            reference_latent.shape
        ):
            raise ValueError(
                f"{label} shape {tuple(tensor.shape)} does not "
                "match freshly encoded sim_latent "
                f"{tuple(reference_latent.shape)}"
            )

        if (
            not tensor.dtype.is_floating_point
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"{label} must be a finite floating-point tensor"
            )

    if tuple(transported.shape) != tuple(target.shape):
        raise ValueError(
            "transported latent and target latent shapes differ"
        )

    frame_count = transported.shape[1]
    height = transported.shape[3]
    width = transported.shape[4]

    mask = state.get("transport_mask")
    count = state.get("contribution_count")

    expected_mask_shape = (
        frame_count,
        1,
        height,
        width,
    )

    if (
        not isinstance(mask, torch.Tensor)
        or tuple(mask.shape) != expected_mask_shape
    ):
        raise ValueError(
            "transport_mask does not match transported latent"
        )

    if mask.dtype != torch.bool:
        raise ValueError(
            "transport_mask must be boolean"
        )

    if (
        not isinstance(count, torch.Tensor)
        or tuple(count.shape) != tuple(mask.shape)
    ):
        raise ValueError(
            "contribution_count does not match transport_mask"
        )

    if (
        count.dtype.is_floating_point
        or bool((count < 0).any())
    ):
        raise ValueError(
            "contribution_count must contain "
            "nonnegative integers"
        )

    if not torch.equal(mask, count > 0):
        raise ValueError(
            "transport_mask must equal contribution_count > 0"
        )

    residual = (
        transported.to(torch.float32)
        - target.to(torch.float32)
    )

    mask_5d = (
        mask.unsqueeze(0)
        .expand_as(residual)
    )

    residual = torch.where(
        mask_5d,
        residual,
        torch.zeros_like(residual),
    )

    if not bool(torch.isfinite(residual).all()):
        raise ValueError(
            "computed transport residual contains NaN or Inf"
        )

    return residual.to(
        device=reference_latent.device,
        dtype=reference_latent.dtype,
    ).contiguous()
'''

replace_once(
    pipeline_integration,
    "\n\ndef load_precomputed_transport_latent(",
    residual_loader
    + "\n\ndef load_precomputed_transport_latent(",
    label="insert residual loader",
)


# ------------------------------------------------------------------
# 2. infer_sim.py
# ------------------------------------------------------------------

infer_sim = Path("infer_sim.py")

replace_once(
    infer_sim,
    '''from deform_transport.pipeline_integration import load_precomputed_transport_latent
''',
    '''from deform_transport.pipeline_integration import (
    load_precomputed_transport_latent,
    load_precomputed_transport_residual,
)
''',
    label="extend pipeline integration imports",
)

replace_once(
    infer_sim,
    '''    parser.add_argument(
        "--transport_mode",
        choices=("correct", "shuffled", "flow", "blend"),
        default="correct",
        help="Which precomputed fused latent to inject when --transport_latent_path is set",
    )

    args, additional_args = parser.parse_known_args()
''',
    '''    parser.add_argument(
        "--transport_mode",
        choices=("correct", "shuffled", "flow", "blend"),
        default="correct",
        help="Which precomputed transport condition to use",
    )
    parser.add_argument(
        "--transport_injection_mode",
        choices=("replace", "inter_step_residual"),
        default="replace",
        help=(
            "replace preserves the existing precomputed-latent "
            "behavior; inter_step_residual keeps the fresh runtime "
            "sim_latent and injects an artifact-local residual "
            "between denoising steps"
        ),
    )
    parser.add_argument(
        "--transport_injection_scale",
        type=float,
        default=1.0,
        help="Scale applied to an inter-step transport residual",
    )
    parser.add_argument(
        "--transport_injection_step",
        type=int,
        default=0,
        help=(
            "Denoising-step index after which the residual is "
            "injected, before re-noising to the next step"
        ),
    )

    args, additional_args = parser.parse_known_args()

    if args.transport_injection_scale < 0:
        raise ValueError(
            "--transport_injection_scale must be nonnegative"
        )

    if (
        args.transport_injection_mode
        == "inter_step_residual"
        and not args.transport_latent_path
    ):
        raise ValueError(
            "--transport_latent_path is required for "
            "inter_step_residual injection"
        )
''',
    label="add injection arguments",
)

replace_once(
    infer_sim,
    '''    # Encode simulation frames to latent space for SDEdit
    sim_latent = None
''',
    '''    # Encode simulation frames to latent space for SDEdit
    sim_latent = None
    transport_residual = None
''',
    label="initialize residual",
)

replace_once(
    infer_sim,
    '''        if args.transport_latent_path:
            print(
                f"Loading {args.transport_mode} transported sim_latent from: "
                f"{args.transport_latent_path}"
            )
            sim_latent = load_precomputed_transport_latent(
                args.transport_latent_path,
                mode=args.transport_mode,
                reference_latent=sim_latent,
            )
            print(f"  transported sim_latent shape: {sim_latent.shape}")
''',
    '''        if args.transport_latent_path:
            if args.transport_injection_mode == "replace":
                print(
                    f"Loading {args.transport_mode} transported "
                    "sim_latent from: "
                    f"{args.transport_latent_path}"
                )
                sim_latent = load_precomputed_transport_latent(
                    args.transport_latent_path,
                    mode=args.transport_mode,
                    reference_latent=sim_latent,
                )
                print(
                    "  transported sim_latent shape: "
                    f"{sim_latent.shape}"
                )
            else:
                print(
                    f"Loading {args.transport_mode} artifact-local "
                    "transport residual from: "
                    f"{args.transport_latent_path}"
                )
                transport_residual = (
                    load_precomputed_transport_residual(
                        args.transport_latent_path,
                        mode=args.transport_mode,
                        reference_latent=sim_latent,
                    )
                )

                residual_float = transport_residual.to(
                    torch.float32
                )

                print(
                    "  transport residual shape: "
                    f"{transport_residual.shape}"
                )
                print(
                    "  transport residual scale: "
                    f"{args.transport_injection_scale}"
                )
                print(
                    "  transport residual step: "
                    f"{args.transport_injection_step}"
                )
                print(
                    "  transport residual mean_abs: "
                    f"{float(residual_float.abs().mean()):.8f}"
                )
                print(
                    "  transport residual max_abs: "
                    f"{float(residual_float.abs().max()):.8f}"
                )
''',
    label="split replace and residual loading",
)

replace_once(
    infer_sim,
    '''        sim_franka_masks=sim_franka_masks,
        low_memory=low_memory,
''',
    '''        sim_franka_masks=sim_franka_masks,
        transport_residual=transport_residual,
        transport_residual_scale=(
            args.transport_injection_scale
        ),
        transport_residual_step=(
            args.transport_injection_step
        ),
        low_memory=low_memory,
''',
    label="pass residual to pipeline",
)


# ------------------------------------------------------------------
# 3. vidgen/pipeline_sdedit.py
# ------------------------------------------------------------------

pipeline_sdedit = Path("vidgen/pipeline_sdedit.py")

replace_once(
    pipeline_sdedit,
    '''from vidgen.utils import extract_subdim


class CausalInferencePipelineSDEdit(torch.nn.Module):
''',
    '''from vidgen.utils import extract_subdim


def apply_transport_residual(
    clean_latent: torch.Tensor,
    transport_residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Add a transport residual without changing the input dtype."""

    if tuple(clean_latent.shape) != tuple(
        transport_residual.shape
    ):
        raise ValueError(
            "clean latent and transport residual shapes differ"
        )

    if scale < 0:
        raise ValueError(
            "transport residual scale must be nonnegative"
        )

    if not bool(torch.isfinite(transport_residual).all()):
        raise ValueError(
            "transport residual contains NaN or Inf"
        )

    if scale == 0:
        return clean_latent

    return (
        clean_latent.to(torch.float32)
        + float(scale)
        * transport_residual.to(torch.float32)
    ).to(dtype=clean_latent.dtype)


class CausalInferencePipelineSDEdit(torch.nn.Module):
''',
    label="add residual application helper",
)

replace_once(
    pipeline_sdedit,
    '''        sim_latent: Optional[torch.Tensor] = None,
        sim_masks: Optional[torch.Tensor] = None,
        sim_franka_masks: Optional[torch.Tensor] = None,
        return_latents: bool = False,
''',
    '''        sim_latent: Optional[torch.Tensor] = None,
        sim_masks: Optional[torch.Tensor] = None,
        sim_franka_masks: Optional[torch.Tensor] = None,
        transport_residual: Optional[torch.Tensor] = None,
        transport_residual_scale: float = 0.0,
        transport_residual_step: int = 0,
        return_latents: bool = False,
''',
    label="extend inference signature",
)

replace_once(
    pipeline_sdedit,
    '''            sim_franka_masks: Franka/mesh masks [B, T, H, W] (True = franka region, weak sdedit).
            return_latents: Whether to return latents alongside decoded video.
''',
    '''            sim_franka_masks: Franka/mesh masks [B, T, H, W] (True = franka region, weak sdedit).
            transport_residual: Artifact-local masked latent residual [B,T,C,H,W].
            transport_residual_scale: Residual strength.
            transport_residual_step: Step index after which to inject the residual.
            return_latents: Whether to return latents alongside decoded video.
''',
    label="extend inference docstring",
)

replace_once(
    pipeline_sdedit,
    '''            assert noise.shape == sim_latent.shape, (
                f"noise shape {noise.shape} != sim_latent shape {sim_latent.shape}"
            )

            # Add noise to simulated latent at the first denoising step
''',
    '''            assert noise.shape == sim_latent.shape, (
                f"noise shape {noise.shape} != sim_latent shape {sim_latent.shape}"
            )

            if transport_residual is not None:
                if tuple(transport_residual.shape) != tuple(
                    sim_latent.shape
                ):
                    raise ValueError(
                        "transport_residual shape "
                        f"{tuple(transport_residual.shape)} "
                        "does not match sim_latent "
                        f"{tuple(sim_latent.shape)}"
                    )

                if not bool(
                    torch.isfinite(
                        transport_residual
                    ).all()
                ):
                    raise ValueError(
                        "transport_residual contains NaN or Inf"
                    )

                if transport_residual_scale < 0:
                    raise ValueError(
                        "transport_residual_scale must be "
                        "nonnegative"
                    )

                if not (
                    0
                    <= transport_residual_step
                    < len(self.denoising_step_list) - 1
                ):
                    raise ValueError(
                        "transport_residual_step must identify "
                        "a denoising step that has a following step"
                    )

            # Add noise to simulated latent at the first denoising step
''',
    label="validate residual",
)

replace_once(
    pipeline_sdedit,
    '''            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
''',
    '''            transport_residual_current = None

            if transport_residual is not None:
                residual_start = (
                    current_start_frame
                    - num_input_frames
                )
                residual_end = (
                    residual_start
                    + current_num_frames
                )
                transport_residual_current = (
                    transport_residual[
                        :,
                        residual_start:residual_end,
                    ]
                )

            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
''',
    label="slice residual per causal block",
)

replace_once(
    pipeline_sdedit,
    '''                    )
                    next_timestep = self.denoising_step_list[index + 1]
''',
    '''                    )

                    if (
                        transport_residual_current
                        is not None
                        and index
                        == transport_residual_step
                        and transport_residual_scale != 0
                    ):
                        print(
                            "Applying inter-step transport "
                            f"residual: block_start="
                            f"{current_start_frame}, "
                            f"step_index={index}, "
                            f"scale="
                            f"{transport_residual_scale}"
                        )

                        denoised_pred = (
                            apply_transport_residual(
                                denoised_pred,
                                transport_residual_current,
                                transport_residual_scale,
                            )
                        )

                    next_timestep = self.denoising_step_list[index + 1]
''',
    label="inject residual before next-step re-noising",
)

print("Patch applied successfully.")
