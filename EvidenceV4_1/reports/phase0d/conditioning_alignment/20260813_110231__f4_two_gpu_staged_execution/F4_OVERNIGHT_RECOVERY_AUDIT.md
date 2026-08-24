# F4 Overnight Recovery Audit

## 1. Audit objective

Read-only recovery and success audit after the expected connection-maintenance window. No generation, evaluation, replay, manifest mutation, or deletion was performed by this audit.

## 2. Infrastructure continuity

- Host boot: `2026-08-04 03:11:05`; current uptime was nine days. `HOST_REBOOT_DURING_WINDOW = False`.
- `deformtransport-dev`: running, started `2026-08-03T22:46:10.611847405Z`, restart count `0`. `CONTAINER_RESTART_DURING_WINDOW = False`.
- `INFRASTRUCTURE_INTERRUPTION_AFFECTING_GPU_JOBS = False`.

## 3. WM-0 generation

- Launched: True. Host PID `87487`; detached container PID `211282`; GPU1 (`CUDA_VISIBLE_DEVICES=1`).
- Completion evidence: `runtime/wm0_container2_stdout.log` records `Finished.` at `2026-08-13 04:24:08.366+00:00`; no fatal error signature was found.
- Host exit status is unavailable because the original detached supervisor was no longer a parent; video integrity independently passes.
- Video: `outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4`; size `6,213,220`; SHA-256 `9caae850cfaec1bf94a0c49e3975a1b95bcd57820ecaff2e96d65db64a93962a`; `81` frames, `832x464`, 16 fps.
- `WM0_COMPLETED = True`; `WM0_VIDEO_INTEGRITY = PASS`.

## 4. FRAG generation

- Launched: True. Host PID `87492`; detached container PID `211297`; GPU2 (`CUDA_VISIBLE_DEVICES=2`).
- Completion evidence: `runtime/frag_container2_stdout.log` records `Finished.` at `2026-08-13 04:24:39.081+00:00`; no fatal error signature was found.
- Host exit status is likewise unavailable; video integrity independently passes.
- Video: `outputs/frag_container_seed0/santa_correct_v3d_seed000.mp4`; size `6,801,423`; SHA-256 `052d97b74b0270d9883fafc327b0dbb7cf1d399c0a38eb79f22968714db556f6`; `81` frames, `832x464`, 16 fps.
- `FRAG_COMPLETED = True`; `FRAG_VIDEO_INTEGRITY = PASS`.

## 5. Watcher / GRID100 exactly-once audit

Ordered reconstructed timeline:

1. `03:49:32+00:00`: detached persistent watcher started with fixed WM0 PID `211282` and FRAG PID `211297`.
2. `04:24:08.366+00:00`: WM-0 logged successful completion.
3. `04:24:23+00:00` (`12:24:23+08:00`): the scheduling watcher recorded `WM-0` as first finished and `GPU1` as first free, atomically created `runtime/grid100_launch.lock`, and started GRID100 with the frozen grid100 artifact.
4. `04:24:39.081+00:00`: FRAG logged successful completion.
5. `04:24:23+00:00`: persistent watcher observed the existing lock and logged `GRID100_NOT_LAUNCHED existing_lock`, therefore did not duplicate the launch.
6. GRID100 video was subsequently written at `13:20:56+08:00`; no further grid process is present.

The earlier watcher did not persist its child exit code/end marker, but its stdout records the exact frozen GRID100 arguments and the completed video passes integrity. There is one lock, one GRID100 stdout/stderr pair, one GRID100 output directory, and no second GRID100 launch record.

- `WATCHER_SURVIVED_CONNECTION_LOSS = PASS` (the detached persistent watcher executed and safely honored the pre-existing lock).
- `GRID100_LAUNCH_LOCK_PRESENT = True`.
- `GRID100_LAUNCH_COUNT = 1`.
- `GRID100_DUPLICATE_GENERATION_DETECTED = False`.

## 6. GRID100 generation

- Launched: True, on GPU1; command log specifies frozen `grid100_center_tracks.npy` and `grid100_center_visibility.npy`, seed `0`, 40 steps, shift `3.0`.
- Exit status: `UNRESOLVED` (missing supervisor completion marker); no fatal error signature was found, and independent video integrity passes.
- Video: `outputs/grid100_container_seed0/santa_correct_v3d_seed000.mp4`; size `6,186,143`; SHA-256 `8a44ba0cfc17ccfed567df15fae321b4f0242f7bc809f0a6be0459168d0f974a`; `81` frames, `832x464`, 16 fps.
- `GRID100_COMPLETED = True`; `GRID100_VIDEO_INTEGRITY = PASS`.

## 7. Frozen artifact integrity

- WM0 manifest SHA-256: `9d1e9f09cb97ac6e0218ecc2168cb3606113d10f04f679c96c0ebc3c8a523ebf` — match.
- FRAG manifest SHA-256: `c603ed2d05837c49c7a3d89736a3e5429b31439de7e528aa25215f73feb5c20b` — match.
- GRID100 manifest SHA-256: `cda03b6272ec20da38cfc29a86a4998aba218d299990d92686ad673dccf2541f` — match.
- Full F3 `SHA256SUMS.txt` verification passed when evaluated at its recorded repository-relative paths.
- Generation logs name the matching frozen WM0/FRAG/GRID100 conditioning files.

`FROZEN_MANIFEST_INTEGRITY = PASS`.

## 8. Formal evaluator status

No formal evaluation log, output metrics, per-sample diagnostics, subgroup outputs, or report deliverables exist. The preregistered post-run did not run because the persistent watcher safely stopped after seeing the pre-existing grid lock.

- `FORMAL_EVALUATOR_EXECUTED = False`.
- `FORMAL_EVALUATOR_EXIT_SUCCESS = UNRESOLVED`.
- Current evaluator `eval_v3_corrected_v2.py` SHA-256: `1aceb35bd0f157425cf8d55089fe90970a0988599a515f6346a39c2fc233cecf`, which does not equal the audit instruction's expected `e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5`; `FORMAL_EVALUATOR_SHA_MATCH = FAIL`. It was not invoked.

## 9. Baseline reproduction

Not available because the formal evaluator was not executed. `RW_BASELINE_REPRODUCTION = UNRESOLVED`; `DTFULL_BASELINE_REPRODUCTION = UNRESOLVED`.

## 10. Candidate primary metrics

Not available. No numeric conclusions, promotion, Pareto decision, or superiority claim can be made.

## 11. Subgroup status

The frozen six subgroup artifacts remain checksum-valid, but their five-method evaluation was not executed. `SUBGROUP_EVALUATION_COMPLETE = False`.

## 12. Deliverable completeness

Missing: `MASTER_SEED0_COMPARISON.{csv,md}`, `seed0_pareto_analysis.{json,md}`, `SUBGROUP_SAFETY_TABLE.{csv,md}`, `generation_runtime_summary.json`, `NEXT_ROUTE_DECISION.json`, `MORNING_HANDOFF.md`, post-run immutability audit, all formal metric outputs, and `visual_comparison/FIVE_METHOD_CONTACT_SHEET.png`.

## 13. Protocol violation check

No unauthorized post-F4 GPU experiment was found in F4 runtime/evidence logs. No COUNT218, GRID100-STABLE, seed replay, alternative K, PHYS/blend/no-vis/19k, or F5 execution was launched. `UNAUTHORIZED_POST_F4_GPU_EXPERIMENT = False`.

The absence of post-run work is a workflow incompleteness, not a duplicate-generation or manifest-mutation violation. The evaluator hash discrepancy must be resolved before any later formal evaluation.

## 14. Final F4 overnight status

`F4_OVERNIGHT_STATUS = GENERATIONS_COMPLETE_POSTRUN_INCOMPLETE`.

All three required generation outputs are valid, exactly one GRID100 launch is evidenced, frozen inputs are intact, and no unauthorized experiment occurred. Formal evaluation and all dependent deliverables remain incomplete.

## 15. Missing work, if any

Only after a separate authorization and evaluator-identity resolution: formal corrected-v2 five-method evaluation, baseline reproduction, diagnostics/subgroups, visual/contact sheet, comparison/Pareto/safety/runtime/immutability reports, and route/handoff reports. No video regeneration is indicated or authorized.
