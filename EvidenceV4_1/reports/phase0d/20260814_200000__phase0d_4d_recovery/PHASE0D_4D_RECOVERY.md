# Phase 0D-4D-R Enabled-Path Recovery

## 1. 阶段目标

Recover a no-op C1 definition with K=0 data contract, close the preview-shape record, and establish canonical epsilon without reopening 0D-4C parity.

## 2. 审计问题

Whether the unchanged parity-proven overlay can consume the recovery-required batched preview artifact for the begin0 exact oracle.

## 3. 使用的数据

Historical F3 WM-0 evidence (`PHASE0D2F_F3_SUMMARY.md`, SHA `084b043e7f7031416ce9cd038ce0e9f30abd6d58079e769f7553c132e2a9d37f`), current overlay, frozen raw preview NPY, legal empty sidecars, and a private-generator canonical epsilon reconstruction.

## 4. 使用的方法

CPU-only current-overlay V3D K=0 functional gate, direct NPY header/load, derived batch-axis-only artifact, and mechanical 40-step shift-3 schedule. Canonical epsilon reconstruction used the unchanged formal generator semantics on a clean authorized GPU0, without model loading.

## 5. 关键命令/脚本

`current_overlay_k0_gate.py`, `derive_preview_and_schedule.py`, and `freeze_canonical_epsilon.py`; all are recovery evidence scripts outside formal overlay.

## 6. 关键结果

Historical WM-0 evidence was found. Current V3D K=0 gate passes with legal actual function inputs: tracks `[81,0,2]`, Boolean visibility `[81,0]`, IDs `[0]`, depth `[81,0]`; `edited_y == y` exactly, zero changed scalars and zero writes. C1 mechanism can therefore be K=0 data contract.

Raw preview SHA matches frozen value and its authoritative on-disk shape is `[16,21,60,104]`. The derived `[1,16,21,60,104]` artifact is a pure inserted batch axis and `batched[0] == raw` exactly. Wan schedule has sigma[0]=1 and index15 sigma=5/6. Canonical-equivalent epsilon reconstruction round-trips exactly, but remains PROVISIONAL.

## 7. PASS/FAIL/UNRESOLVED

`CURRENT_OVERLAY_K0_EXACT_NOOP=True`; however `PREVIEW_BATCH_CONTRACT=FAIL_CURRENT_FORMAL_LOADER_REQUIRES_UNBATCHED`. Unchanged `wan_move.py:316-317` compares preview shape directly with unbatched noise `[16,21,60,104]`; it rejects the recovery-required derived batched artifact. Thus begin0 oracle was not launched because it would deterministically fail shape validation.

## 8. 对后续实验影响

No code was patched and no C was launched. A user decision is required to resolve whether formal enabled execution should consume the raw unbatched artifact (matching current formal loader) or a separately authorized loader/API contract should accept the derived batched artifact. Until then epsilon is not FINAL and C remains unauthorized.

## 9. 遗留问题

Begin0 exact oracle, zero-preview leakage check, begin15 runtime sanity, C manifests/preregistration, and C authorization are blocked by the preview batch-contract contradiction. GPU3 was not used.
