import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/DeformTransport")
import infer_sim

ROOT = Path("/workspace/DeformTransport")

ASSETS = {
    "santa": ROOT / (
        "server_runs/20260804_234925_autonomous_deformtransport/"
        "prepared_inputs/"
        "official_santa_81f_aligned_final_sim_20260806_234410/"
        "noises.npy"
    ),
    "tree": ROOT / (
        "server_runs/20260804_234925_autonomous_deformtransport/"
        "prepared_inputs/"
        "tree_official_precomputed_aligned_final_sim_20260807_185055/"
        "noises.npy"
    ),
}

report = {}

for case, path in ASSETS.items():
    raw = np.load(path).astype(np.float32)
    assert raw.shape == (81, 60, 104, 32)

    out = infer_sim.load_noise(
        noise_path=str(path),
        target_frames=21,
        channel_dim=16,
        downsample_mode="nearest",
        eval_degradation=0.0,
    )

    a = out["structured_noise"].float()
    b = out["structured_noise_sde"].float()

    # Expected nearest temporal mapping.
    tids = (
        torch.linspace(0, 80, steps=21)
        .round()
        .long()
        .tolist()
    )

    x = torch.from_numpy(raw[tids]).permute(0, 3, 1, 2)

    d_a_first = float((a - x[:, :16]).abs().max())
    d_a_last  = float((a - x[:, 16:]).abs().max())

    d_b_first = float((b - x[:, :16]).abs().max())
    d_b_last  = float((b - x[:, 16:]).abs().max())

    rec = {
        "temporal_indices": tids,
        "structured_noise_shape": list(a.shape),
        "structured_noise_sde_shape": list(b.shape),
        "structured_vs_first16_maxabs": d_a_first,
        "structured_vs_last16_maxabs": d_a_last,
        "sde_vs_first16_maxabs": d_b_first,
        "sde_vs_last16_maxabs": d_b_last,
    }

    report[case] = rec

    print("\n====", case, "====")
    print("temporal indices:", tids)
    print("structured vs first16:", d_a_first)
    print("structured vs last16 :", d_a_last)
    print("sde vs first16       :", d_b_first)
    print("sde vs last16        :", d_b_last)

Path("a0_exact_mapping_audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print("\nA0_EXACT_MAPPING_AUDIT_DONE")
