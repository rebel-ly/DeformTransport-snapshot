# Phase 0D-4D-R2 — Contract Correction and Begin0 Exact Replay

## 1. 阶段目标

执行 R2 的 raw-preview contract correction，并以 begin0 exact canonical replay 验证冻结 epsilon；不重开 0D-4C，不启动 C1/C2。

## 2. 审计问题

此前 `[1,C,T,H,W]` requirement 是 audit-protocol error。R2 正确撤回该要求：formal loader 需要 unbatched tensor。R2 随后假定 raw formal runtime shape 为 `[16,21,60,104]`，该假定在真实 begin0 runtime 中失败。

## 3. 使用的数据

raw preview: `preview_wan_vae_latent_e1_832x480.npy`, SHA256 `9d71791f70fa519001708d0986c4b1cee297941b8f66a3d1a15e03ef8ce8bb8f`, shape `[16,21,60,104]`.

persisted provisional epsilon: `FINAL_C_SHARED_EPSILON.npy`, shape `[16,21,60,104]`, tensor-content SHA `b6f3bbbd7bbc3412b70dd57c39e1709e70ba7d38b9e0cb60bac7820776208b51` under the persisted shape+dtype+bytes definition. It exactly round-trips with its `.pt` companion.

The formal overlay remains unchanged: `wan/wan_move.py` SHA256 `eae7f5a86f39164f3ad1ce3b8db4a974f4a71f42c2898402f029bb9db77c32f7`; `generate.py` SHA256 `45f7323f22d7bb7d593b949fa48e6cf764d08cafeaf8863d726df9a663b21b85`.

## 4. 使用的方法

An external preflight validator checked raw unbatched preview, persisted epsilon, overlay hashes, canonical K=1257 files, and legal K=0 files without importing or changing formal source. Begin0 then used the formal parity-proven overlay, raw preview, persisted epsilon, Correct V3D K=1257 carriers, 40 steps, shift 3, seed 0, and start index 0 on clean GPU2.

After failure, the permitted first-divergence audit read the unchanged grid formula. With the canonical 832×480 source image, floating-point evaluation yields `sqrt(480*832*(480/832)) = 479.999…`; the source applies floor division before patch-grid restoration, resulting in `lat_h=58`, `lat_w=104`. Thus real formal noise shape is `[16,21,58,104]`.

## 5. 关键命令/脚本

- `validate_contract.py` (external CPU-only preflight)
- `run_begin0.sh` and `run_begin0_inside.sh` (external detached wrappers)
- formal error evidence: `begin0_gpu2/stderr.log`, `stdout.log`, `exit_code.txt`

## 6. 关键结果

The external preflight passed the then-specified artifact checks, and derived batched preview never entered a formal run. Begin0 reached formal `wan_move.generate` after model initialization but stopped before first denoiser call:

`ValueError: initial_epsilon shape is incompatible with formal noise shape`

Exit code was 1 at `2026-08-14T17:19:10Z`; no MP4 was produced. The actual source formula audit proves the initial reconstructed epsilon's height 60 does not match the runtime height 58.

## 7. PASS/FAIL/UNRESOLVED

- `FORMAL_OVERLAY_MODIFIED=False` — PASS
- `OLD_BATCHED_PREVIEW_REQUIREMENT_WITHDRAWN=True` — PASS
- `DERIVED_BATCHED_PREVIEW_USED_IN_FORMAL_RUN=False` — PASS
- `TRANSPORT_MODE_CONTRACT_GUARD=PASS` — PASS
- `BEGIN0 runtime epsilon shape check` — FAIL: `[16,21,60,104] != [16,21,58,104]`
- `BEGIN0_VS_CANONICAL_RGB_EXACT` — UNRESOLVED (no output)
- `FINAL_SHARED_EPSILON_FROZEN=False`
- `ENABLED_PATH_SANITY_STATUS=HARD_STOP`

## 8. 对后续实验影响

R2 pre-registration gate cannot pass. No epsilon regeneration, scheduler change, source change, begin15 probe, C1, C2, or shuffled arm was performed. A later authorized correction must first resolve the raw-preview/epsilon artifact geometry against the unchanged formal latent-grid computation, then restart from the appropriate formal gate.

## 9. 遗留问题

The authoritative recovered raw preview and epsilon are 60×104, whereas this exact formal call computes 58×104. The cause is now localized to the contract/artifact-grid mismatch, but no new artifact is created under this stop rule.
