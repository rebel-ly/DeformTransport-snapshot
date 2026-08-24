# Phase0D-2R-E — Formal Recovered Same-Seed Replay Determinism Audit

## 1. 阶段目标

Run the first authorized recovered scientific replay: corrected-v2 Santa, Correct V3D, seed 0, independently on L40 GPU1 and GPU2, then compare final decoded RGB exactly.

## 2. 审计问题

Determine whether the frozen seed-0 formal contract reproduces exactly across the two audited devices after the earlier GLIBC, lazy-import, container-runtime, and Python-binding recovery lineage.

## 3. 使用的数据

Used the 0D-2R-D PASS runtime contract and wrapper, frozen corrected-v2 N=1257/T=81 inputs, frozen source checksums, existing container paths, and fresh A/B output directories. Phase0D-2 and 0D-2R-C historical engineering failures remain preserved.

## 4. 使用的方法

Rechecked prior-stage, source/runtime, container-path, asset-shape, import, GPU, and paired-manifest gates. Executed only seed 0 A on GPU1 and B on GPU2 in parallel. Captured independent exit codes. Verified MP4s, then decoded both through the same existing decord 0.6.0 CPU RGB pipeline without resize/crop/resampling and compared raw RGB bytes exactly.

## 5. 关键命令/脚本

The 0D-2R-D `run_with_formal_wanmove_python.sh` wrapper bound `/workspace/tools/miniforge3/envs/wan-move/bin/python`. Both invocations requested `480*832`, 81 frames, 40 steps, shift 3.0, bf16, `t5_cpu`, `offload_model`, Correct V3D, and seed 0. The existing container has no ffmpeg/ffprobe; this was documented and the existing frozen-runtime decord decoder was used uniformly.

## 6. 关键结果

All pre-generation gates passed and `N1277_PATH_HITS=0`. A and B both exited 0, produced non-empty 7,729,377-byte MP4 files, and those MP4 SHA256 values are identical. Both decoded to 81×464×832×3 uint8 RGB frames (93,809,664 bytes) with identical raw RGB SHA256. Exact comparison found zero scalar mismatches, zero differing frames, and maximum/mean absolute difference 0.

The actual decoded height is 464. The frozen command requested height 480, but Wan VAE has spatial stride `(4,8,8)` and 480 is not divisible by 8; no decode-stage resize/crop/resampling was applied.

## 7. PASS/FAIL/UNRESOLVED 判断

`PHASE0D2RE_STATUS = PASS`; `SAME_SEED_REPLAY=PASS`; `CROSS_DEVICE_REPLAY_EXACT=True`; `RUNTIME_OUTPUT_NONDETERMINISM=NOT_DETECTED` for this audited contract.

## 8. 对后续实验影响

For the frozen corrected-v2 Santa Correct V3D seed-0 contract, final decoded RGB output was exactly replay reproducible across the audited L40 GPUs. `PROCEED_TO_PHASE0D3=True`; this phase does not enter 0D-3.

## 9. 遗留问题

This establishes only final decoded RGB reproducibility for seed 0 across these two devices. It does not establish bitwise latent-tensor determinism, all-seed determinism, all-hardware determinism, task quality stability, or future-video GT evaluation. Phase0D remains `PAIRWISE_OUTPUT_STOCHASTICITY_ONLY`.
