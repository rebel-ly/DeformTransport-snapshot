# Phase 0D-4C Corrected Parity Closure

## 1. 阶段目标

恢复 authoritative canonical DT-FULL seed0 protocol，并验证 original Wan-Move 路径（A2）与 overlay-disabled 路径（B2-G2）在冻结 canonical V3D 条件下的输出 parity。

## 2. 审计问题

旧 A/B 不能作为证据：其 launcher 缺少 canonical `DT_TRANSPORT_VARIANT=v3d`、material-IDs 与 depth 配置。第一轮 corrected A2/B2-G1 则在外部 GPU 资源竞争期间因 CUDA OOM 失败，属于 infrastructure failure，而非科学 parity 结果。

## 3. 使用的数据

三份唯一锁定的最终 artifact：

- A2 original：`.../corrected_parity_20260814/a2_gpu0_formal_rerun/santa_correct_v3d_seed000.mp4`
- B2-G2 overlay-disabled：`.../corrected_parity_20260814/b2_gpu2_hardgate/santa_correct_v3d_seed000.mp4`
- canonical：`.../parallel_acceleration/20260814_000000__dtfull_seed_floor/outputs/gpu0_seed0_eligibility/santa_correct_v3d_seed000.mp4`

所有三份均为 81 帧、832×464、16 fps，MP4 SHA256 均为 `08785a3ce49d4faa98c8fe8850fed1b2912d8b8fc4fa80ee377de6dfe7bff935`。

## 4. 使用的方法

先验证 A2 `exit_code=0`、`completion.marker=COMPLETE`、MP4 可解码；再用同一 CPU OpenCV decoder 将三视频解码为 `81×464×832×3 uint8`，逐 channel 比较。随后使用 SHA256 已核验的 frozen corrected-v2 evaluator（`e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5`）进行 TC-MAR/TC-ME 一致性检查。

## 5. 关键命令/脚本

- `corrected_parity_20260814/final_parity_cpu_audit.py`（CUDA disabled）
- `conditioning_alignment/.../generated/eval_v3_corrected_v2_recovered.py`
- `20260814_095925__dt_full_5seed_drop_zero62_formal_evaluation/run_formal_candidate.sh`（未修改；通过 Bash 调用，不修改权限）

## 6. 关键结果

| Comparison | MP4 exact | decoded RGB exact | changed channels | max / mean abs diff |
|---|---:|---:|---:|---:|
| A2 vs canonical | True | True | 0 | 0 / 0.0 |
| B2-G2 vs canonical | True | True | 0 | 0 / 0.0 |
| A2 vs B2-G2 | True | True | 0 | 0 / 0.0 |

Decoded RGB SHA256 for all three: `935d9301a208abb73e437346c3297a31705563e60ce2aa2be4cf46b44ce7cbc6`.

Frozen formal values for all three: TC-MAR `17.144317299874714`; TC-ME `0.7265499674289193`.

B2-G2 experienced recorded external resource contention and thermal slowdown, but its final artifact is byte- and decoded-RGB-exact to canonical; the observed effect is therefore runtime-only for this run.

## 7. PASS/FAIL/UNRESOLVED

`CANONICAL_PROTOCOL_RECOVERED = EXACT_PASS`  
`ADAPTER_DISABLED_PATH_PARITY = EXACT_PASS`  
`OVERLAY_DISABLED_OUTPUT_REGRESSION = NOT_DETECTED`  
`PATCH_DISABLED_PATH_REGRESSION_CONFIRMED = False`  
`FINAL_PARITY_GATE = PASS`

This is CASE 1 of the preregistered decision table.

## 8. 对后续实验影响

B2-G1 need not be rerun: clean A2 reproduces original canonical output and B2-G2 reproduces overlay-disabled output, both exactly. B2-G1 remains `INFRASTRUCTURE_FAILURE_EXTERNAL_RESOURCE_CONTENTION_OOM` and is excluded from scientific parity. Phase 0D-4C can close; C remains unauthorized pending enabled-path runtime sanity and final shared-epsilon freeze/preregistration.

## 9. 遗留问题

`ENABLED_PATH_SANITY_STATUS = NOT_YET_RUN`; `FINAL_SHARED_EPSILON_FROZEN = False`; `C_AUTHORIZED = False`. No C arm was launched and no generation source, overlay, or frozen manifest was modified during closure.
