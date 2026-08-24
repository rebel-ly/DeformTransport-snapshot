# Disabled-path parity probe — stopped before comparison

## 阶段目标

在同一 GPU、同一运行时中比较 original formal Wan-Move 与 patched overlay（所有 Preview-SDEdit 参数未传入）。本文件只记录本次 mandatory parity probe 的第一处失败。

## 已冻结 provenance

- GPU: GPU0, NVIDIA L40, UUID `GPU-14bb1875-6456-dba9-fde5-e1383c8d480b`.
- Original source: `/workspace/Wan-Move/wan/wan_move.py`, SHA-256 `aca79f9cc4bf32ea363c4440ed2c7e7d90ef5aa763f3e96ae6c2b8eff35c1857`.
- Patched overlay source: `/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay/wan/wan_move.py`, SHA-256 `eae7f5a86f39164f3ad1ce3b8db4a974f4a71f42c2898402f029bb9db77c32f7`.
- Both sources, their `wan.wan_move` modules, scheduler module, checkpoint, Python executable, Correct K=1257 tracks/visibility/depth/IDs, source image, prompt, seed 0, 40-step UniPC shift 3, and bf16 settings were frozen before launch. GPU3 was not used.

## 第一处失败

The original (A) formal baseline execution stopped during model initialization. Its stdout ends after `Creating WanMove pipeline.`; stderr has only pre-existing AMP deprecation warnings. No MP4 was created, no `exit_code.txt` was written, and GPU0 returned to 7 MiB use. The output directory was verified writable under UID:GID 10011:10011 after the event.

Because A has no valid output, there is no authoritative decoded RGB/tensor parity comparison and the patched-overlay (B) run was not started.

## Gate decision

`ADAPTER_DISABLED_PATH_PARITY=FAIL` for this execution attempt, specifically `ORIGINAL_BASELINE_INTEGRITY_FAILURE_BEFORE_COMPARISON`. This does **not** establish a scientific or overlay regression; it prevents the required comparison. Per protocol: no shared epsilon is frozen, no C0/C1/C2 runs are launched, and no candidate metric is inspected.
