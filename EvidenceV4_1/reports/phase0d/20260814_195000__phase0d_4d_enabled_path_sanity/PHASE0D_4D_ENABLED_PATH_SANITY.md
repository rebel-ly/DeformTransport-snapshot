# Phase 0D-4D Enabled-Path Sanity

## 1. 阶段目标

在不重开 0D-4C disabled-path exact parity 的前提下，验证 formal overlay 的 enabled Preview-SDEdit path 是否可为 C1 Preview-only 与 C2 Preview+Correct 冻结公平协议。

## 2. 审计问题

关键问题是 current parity-proven overlay 是否同时提供：(a) preview latent、external epsilon、start-index 的 reachable enabled inputs，及 (b) 不改变其他条件的 clean Preview-only transport-off arm。

## 3. 使用的数据

读取 B2-G2 persisted provenance、formal overlay source、以及 preview artifact `preview_wan_vae_latent_e1_832x480.npy`。未读取或修改 C candidate，未生成 epsilon。

## 4. 使用的方法

CPU-only static source audit：检查 `generate.py` CLI 到 `WanMove.generate` 的参数传递、`wan_move.py` enabled branch、transport construction和 UniPC 调用；以 SHA256、NumPy load 和 tensor statistics 验证 preview artifact。没有加载 Wan 模型、没有 GPU runtime probe。

## 5. 关键命令/脚本

- `rg` / `sed` 读取 overlay `generate.py`、`wan_move.py`、`fm_solvers_unipc.py`
- `sha256sum` 对 B2-G2 provenance 与 current overlay 作机械比较
- CUDA-disabled Python/NumPy 读取 preview latent

## 6. 关键结果

overlay source 与 B2-G2 exact-parity provenance 相同：`wan_move.py` SHA256 `eae7f5a86f39164f3ad1ce3b8db4a974f4a71f42c2898402f029bb9db77c32f7`，`generate.py` SHA256 `45f7323f22d7bb7d593b949fa48e6cf764d08cafeaf8863d726df9a663b21b85`；formal source 未修改。

Preview/external-epsilon/start-index CLI and runtime inputs are reachable. However, `wan_move.py:291-295` unconditionally computes track features, calls `replace_feature`, and uses `edited_y` for `y_cond`; no transport enable/disable switch exists. Thus a clean C1 Preview-only arm cannot be expressed by this formal overlay.

Preview artifact SHA256 matches the frozen value `9d71791f70fa519001708d0986c4b1cee297941b8f66a3d1a15e03ef8ce8bb8f`, is finite float32, range [-3.6171555519104004, 4.851651668548584], mean 0.11778508126735687, but actual shape is `[16,21,60,104]`, not the phase-required `[1,16,21,60,104]`. No reshape was applied.

## 7. PASS/FAIL/UNRESOLVED

`ENABLED_SANITY_OVERLAY_MATCHES_EXACT_PARITY_SOURCE = True`  
`C1_TRANSPORT_DISABLED_CLEANLY = False`  
`PREVIEW_LATENT_SHAPE_GATE = FAIL`  
`PREREGISTRATION_GATE = FAIL`  
`ENABLED_PATH_SANITY_STATUS = FAIL`  
`C_AUTHORIZED = False`

## 8. 对后续实验影响

The protocol cannot legitimately freeze shared epsilon, run a bounded enabled runtime probe, or preregister C1/C2 until a separately authorized design decision provides a transport-off interface and resolves the batch-shape contract. No source was patched here because doing so would sever direct coverage by 0D-4C exact disabled-path parity.

## 9. 遗留问题

RW/Wan sigma redo, UniPC runtime semantics, RNG differential proof, epsilon freeze, start-state verification, GPU probe, and C manifests were intentionally not run after the mandatory C1 gate failure. No C1, C2, Shuffled, or GPU3 work was launched.
