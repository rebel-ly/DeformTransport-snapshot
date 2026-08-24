# Wave-1 Duplicate-Wrapper Protocol Deviation

## Event

At Wave-1 start, the original three container generation processes were launched from host wrappers. A subsequent liveness misread (GPU telemetry was not a completion signal) caused a second detached wrapper group to be started for the same three frozen output paths.

## Lineage

Original container processes, identified by the first launch records and timestamps, are:

- GPU0 eligibility seed0: PID 27786, started 2026-08-14 00:10:57 local.
- GPU1 seed1: PID 27680, started 2026-08-14 00:10:56 local.
- GPU2 seed2: PID 27994, started 2026-08-14 00:10:57 local.

The later detached duplicate processes were PIDs 304085, 304111, and 304215, with persistent start timestamps 2026-08-13T16:13:45/46+00:00. They were sent SIGTERM only after their duplicate identity was established; all three persistent exit markers record code 143.

## Current containment and evidence

- Duplicate wrappers are not alive and their persistent stdout/stderr are empty.
- No Wave-1 MP4 or temporary output existed at the first post-containment audit, so duplicate substantive output writing was not evidenced.
- The original three PIDs remained alive after duplicate containment. An observation-only detached watcher records their natural exits and has no generation-launch capability.

`ORIGINAL_GROUP_IDENTIFIED = PASS`.

`DUPLICATE_GROUP_IDENTIFIED = PASS`.

`DUPLICATE_WRITERS_FULLY_TERMINATED = PASS`.

`DUPLICATE_OUTPUT_WRITE_DETECTED = False` for all three arms at audit time.

`OUTPUT_CONTAMINATION_RISK = False` provisionally, pending final per-output timestamp/integrity audit after originals complete.

`WAVE2_AUTHORIZED = False` and `WAVE2_LAUNCHED = False` pending original completion, integrity, and GPU0 decoded-RGB identity gate.
