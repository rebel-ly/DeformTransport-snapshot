# Phase0D-2R-D — Formal Wan-Move Python Runtime Binding Recovery

## 1. 阶段目标

Identify, verify, and freeze an existing unmodified Wan-Move-compatible interpreter without starting formal generation or modifying packages.

## 2. 审计问题

Phase0D-2R-C used default container `/usr/bin/python` 3.8.10, which lacks `easydict`; it failed before generation. The question was which interpreter historically ran successful Wan-Move work.

## 3. 使用的数据

Used the actual `deformtransport-dev` container, its mounted workspaces/tools, historical Wan-Move successful run logs and wrappers, frozen source contracts, and the 0D-2R-D protocol.

## 4. 使用的方法

Performed a bounded existing-interpreter inventory; ranked candidates by historical provenance; ran import-only dependency/config/generate smokes; ran no-model CUDA identity smokes; verified frozen source SHA; and executed only the new wrapper's dry-run.

## 5. 关键命令/脚本

Selected `/workspace/tools/miniforge3/envs/wan-move/bin/python`. The new inert `run_with_formal_wanmove_python.sh --dry-run` printed its exact bound Python and translated container command; it did not invoke `generate.py`.

## 6. 关键结果

Historical successful Santa Wan-Move resource logs explicitly show this Python launching `generate.py`. It is Python 3.11.0, imports `easydict`, Torch 2.5.1+cu121, the patched prompt module (without eager DashScope), `wan_move_14B`, and `generate`. Both GPU1 and GPU2 appear as NVIDIA L40 under individual CUDA visibility. All three critical source hashes match the frozen values.

## 7. PASS/FAIL/UNRESOLVED 判断

`PHASE0D2RD_STATUS = PASS`. `NEXT_RUNTIME_BLOCKER=NONE` for the no-model formal import path. No package/environment/source edit occurred.

## 8. 对后续实验影响

The frozen formal runtime is now bound to the existing proven container Python through the new wrapper. `PROCEED_TO_PHASE0D2RE=True` means only a later expressly authorized recovery replay may use this contract.

## 9. 遗留问题

No 14B checkpoint load, generation, seed replay, MP4, decoding, or determinism comparison occurred. Phase0D-2R-C remains an engineering failure before generation and same-seed determinism remains `NOT_EVALUATED`.
