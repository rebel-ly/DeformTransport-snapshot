# F4 Morning Handoff

## 1. F4 execution integrity

All three frozen seed0 videos are present and passed 81-frame, 832x464 integrity checks. GRID100 has exactly one evidenced launch. F3 manifests and artifacts remain checksum-valid; no unauthorized post-F4 GPU experiment was found.

## 2. Formal evaluation recovery

The exact authorized recovered evaluator was found and used (SHA-256 `e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5`). Its container compatibility path was a symlink only; neither evaluator nor inputs were modified. Five candidate bindings completed appearance and motion evaluation logs under `evaluation/frozen_evaluator_runs/`.

## 3. Mandatory baseline gate — blocked

TC-MAR Lab means reproduced exactly. TC-ME means did not reproduce bit-for-bit and the frozen F4-R1 contract defines no tolerance:

| Baseline | Expected TC-ME | Actual TC-ME | Residual |
|---|---:|---:|---:|
| RW | 0.5869890665947547 | 0.586977941501839 | -0.00001112509291572936 |
| DT-FULL | 0.7265499674289193 | 0.7265571851768204 | 0.0000072177479011076 |

Therefore `RW_BASELINE_REPRODUCTION = FAIL`, `DTFULL_BASELINE_REPRODUCTION = FAIL`, and `F4_FORMAL_RESULTS_VALID = False`.

## 4. What was deliberately not concluded

Per the hard-stop rule, no candidate ranking, primary-direction pass, Pareto analysis, subgroup interpretation, promotion, or next-route decision was made. The candidate logs are preserved as diagnostics only.

## 5. What cannot yet be claimed

This remains seed0-only. There is no statistical superiority claim. FRAG is not a pure fragmentation intervention because it strongly reduces trajectory count. No further GPU generation or replay was run.

## 6. Required next decision

Resolve whether an explicit frozen numerical tolerance / deterministic runtime setting applies to the tiny TC-ME baseline residuals, then authorize a fresh formal interpretation only if the baseline gate can be validly satisfied.
