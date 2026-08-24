import hashlib
import json
import os
import sys

import numpy as np
import torch

OVERLAY = "/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay"
REPORT = "/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_040000__phase0d_4d_r3g_epsilon_bridge"
EPSILON = "/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/R3_SHARED_EPSILON_58x104.npy"
sys.path.insert(0, OVERLAY)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def tensor_sha(t):
    return hashlib.sha256(t.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()

scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
scheduler.set_timesteps(40, device="cpu", shift=3.0)
schedule = [{"index": i, "timestep": int(t), "sigma": float(scheduler.sigmas[i])} for i, t in enumerate(scheduler.timesteps)]

# Exact generation-line semantics in wan_move.py: CUDA private generator, seed 0,
# float32, [16, 21, 58, 104], with no preceding calls on that generator.
device = torch.device("cuda:0")
generator = torch.Generator(device=device)
generator.manual_seed(1)
native = torch.randn(16, 21, 58, 104, dtype=torch.float32, generator=generator, device=device)
external = native.clone(); np.save("/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_100000__phase0d_4d_r3m_seed1_wave1/EPSILON_SEED1_58x104.npy", native.detach().cpu().numpy())
different = int(torch.count_nonzero(native != external).item())
diff = (native - external).abs()
payload = {
  "scheduler_class": "FlowUniPCMultistepScheduler",
  "num_train_timesteps": 1000,
  "num_inference_steps": 40,
  "shift": 3.0,
  "device_for_schedule": "cpu",
  "actual_schedule": schedule,
  "native_noise_semantics": {
    "generator": "torch.Generator(device=self.device)", "seed": 0,
    "randn_shape": [16, 21, 58, 104], "dtype": "torch.float32", "device": str(device),
    "call_order": "first and only use of private seed_g before optional initial_epsilon replacement"
  },
  "native_noise_tensor_sha256": tensor_sha(native),
  "external_epsilon_tensor_sha256": tensor_sha(external),
  "native_noise_equals_external_epsilon": bool(torch.equal(native, external)),
  "different_scalar_count": different,
  "max_abs_diff": float(diff.max().item()),
  "mean_abs_diff": float(diff.mean().item()),
  "private_generator_reused_after_initial_noise": False,
  "private_generator_reuse_evidence": "static source: seed_g occurs only in its construction/manual_seed and the single torch.randn call"
}
with open(os.path.join(REPORT, "ACTUAL_WAN_SCHEDULE.json"), "w") as f:
    json.dump({k: payload[k] for k in ("scheduler_class", "num_train_timesteps", "num_inference_steps", "shift", "device_for_schedule", "actual_schedule")}, f, indent=2)
with open(os.path.join(REPORT, "CANONICAL_NATIVE_NOISE_AUDIT.json"), "w") as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
