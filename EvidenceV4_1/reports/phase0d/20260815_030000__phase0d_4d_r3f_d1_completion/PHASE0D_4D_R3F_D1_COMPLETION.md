# Phase 0D-4D-R3F — D1 Completion

## 1. 阶段目标

Check D1's engineering completion and exact canonical bridge without opening C1/C2 results.

## 2. 审计问题

Whether the enabled external-epsilon begin0 path exactly reproduces canonical decoded RGB.

## 3. 使用的数据

D1 output and the pre-existing canonical output, decoded with the frozen Phase0D-4C OpenCV RGB comparator.

## 4. 使用的方法

Read-only exit/marker/log checks; metadata verification; canonical decode/RGB hash comparison.

## 5. 关键命令/脚本

Existing `final_parity_cpu_audit.py` decode semantics: OpenCV frame order, BGR→RGB conversion, contiguous uint8 SHA.

## 6. 关键结果

D1 completed normally but decoded RGB was not exact: 58,606,024 channel values differ; max difference 254; mean difference 5.840064121751891.

## 7. PASS/FAIL/UNRESOLVED

Engineering PASS; canonical bridge FAIL_RGB_NOT_EXACT. C1 remains running and C2 remains sealed.

## 8. 对后续实验影响

Epsilon is not frozen FINAL. No C1/C2 metrics were opened and no rerun was performed.

## 9. 遗留问题

The bridge mismatch requires ordered D1 diagnosis before any cross-baseline interpretation or promotion.
