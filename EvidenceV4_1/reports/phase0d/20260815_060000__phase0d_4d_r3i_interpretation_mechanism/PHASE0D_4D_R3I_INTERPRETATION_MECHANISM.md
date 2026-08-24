# R3I interpretation and mechanism audit

## 1. 阶段目标

Interpret the unsealed seed-0 C1/C2 result, audit preview proximity, and freeze a shuffled mechanism plan.

## 2. 审计问题

Separate end-to-end benchmark numbers from the incremental Correct-versus-K0 transport contrast.

## 3. 使用的数据

C1/C2, canonical RW and DT-FULL seed-0 outputs, canonical 480-domain preview frames, and persisted DT-FULL five-seed results.

## 4. 使用的方法

The maskless wrapper reused `common()` and the sharpness definition from the frozen diagnostic source. No support mask was fabricated.

## 5. 关键命令/脚本

`maskless_preview.py`; persisted corrected-v2 result records; Phase0B corrected physical-visibility contract.

## 6. 关键结果

C2 improves TC-ME by 11.11% versus C1. C2 is numerically lower than RW in both frozen metrics, but the backbone is unmatched. Maskless diagnostics show C2 is closer to preview than RW and has higher, not lower, gradient energy.

## 7. PASS/FAIL/UNRESOLVED

Seed-0 C2-versus-C1 is PASS. Preview imprinting red flag is not detected by maskless diagnostics. Shuffled hard gate FAILS because the only recovered canonical shuffled artifact has N=1277, unlike frozen C2 N=1257.

## 8. 对后续实验影响

No shuffled GPU job was launched. Cross-baseline results remain end-to-end only. Paired multiseed protocol is frozen but waits for a valid shuffled seed-0 mechanism test.

## 9. 遗留问题

Recover a canonical N=1257 shuffled correspondence artifact before any CS launch. Do not interrupt live E0.
