# Wave-1 Final Failure Forensics

## 1. Audit objective

Determine whether the Wave-1 integrity failure was a real generation failure, an output-path problem, or a process-lineage/audit error. This was read-only: no generation was started, restarted, terminated, or deleted.

## 2. Exact final-audit predicates

| Arm | Expected output | Exists | Size | Decodable | Frames / HW | Exit code | Failing predicate |
|---|---|---:|---:|---|---|---|---|
| GPU0 seed0 eligibility | `outputs/gpu0_seed0_eligibility/santa_correct_v3d_seed000.mp4` | False | n/a | False | 0 / 0x0 | unavailable; process alive | Missing final MP4 at premature audit |
| GPU1 seed1 | `outputs/seed1/santa_correct_v3d_seed001.mp4` | False | n/a | False | 0 / 0x0 | unavailable; process alive | Missing final MP4 at premature audit |
| GPU2 seed2 | `outputs/seed2/santa_correct_v3d_seed002.mp4` | False | n/a | False | 0 / 0x0 | unavailable; process alive | Missing final MP4 at premature audit |

No temporary MP4, partial MP4, latent/output tensor, or alternate-path MP4 was found within the Wave-1 evidence tree or the searched canonical formal-output root.

## 3. Critical process lineage

The original jobs remain alive. The host PID / container PID-namespace pairs are:

| Arm | Host PID | Container PID | GPU UUID | GPU memory | Output |
|---|---:|---:|---|---:|---|
| GPU0 seed0 eligibility | 27786 | 303559 | `GPU-14bb1875-6456-dba9-fde5-e1383c8d480b` | 41094 MiB | eligibility seed0 path |
| GPU1 seed1 | 27680 | 303505 | `GPU-0e2857f8-18bc-0f5b-c1ff-5b67f892cd60` | 41094 MiB | seed1 path |
| GPU2 seed2 | 27994 | 303613 | `GPU-56d1a97e-c16c-ebf6-4fc6-8466b32d0bbf` | 41092 MiB | seed2 path |

Each is a live `generate.py` process with the frozen input image, corrected-v2 N=1257 tracks/visibility, V3D depth environment, intended seed, 40 steps, shift 3.0, bf16, and intended final output path. Their host PPID is 298120; their container PPID is 0. They have no controlling terminal. `SURVIVING_WAVE1_GENERATION_PROCESS_DETECTED = True`.

## 4. Why the coordinator falsely completed

The coordinator executed inside `deformtransport-dev` but monitored host PID values `27786`, `27680`, and `27994`. Those are not the corresponding PID values in the container namespace (`303559`, `303505`, `303613`). Its `kill -0` checks therefore failed immediately and it wrote `ALL_ORIGINAL_PIDS_EXITED`, then audited absent final files. This is a process-lineage tracking bug, not evidence that the generators exited.

## 5. Logs and progress evidence

The original host-wrapper stdout/stderr files are empty because the wrappers were not persistent log owners. No Python traceback, CUDA OOM, SIGTERM/signal, container-runtime, or file-write error is present in the available original logs. The furthest completed stage evidenced by persisted logs is `UNRESOLVED_PRE_FINAL_OUTPUT`; GPU allocations demonstrate models are loaded/actively resident, but telemetry is not used to infer semantic completion.

## 6. Duplicate-wrapper causality

The later detached duplicate group has persistent start/end markers and exit code 143. Its stdout/stderr are empty. The original three processes remained alive after duplicate containment, with distinct host/container PID mappings. There is no evidence that terminating the duplicate group killed a shared parent or process group of the originals.

`DUPLICATE_WRAPPER_CAUSALLY_INVOLVED = NOT_DETECTED`.

## 7. Failure classification

`PRIMARY_FAILURE_CLASS = CASE_B_PROCESS_LINEAGE_BUG_GENERATION_STILL_RUNNING`.

`ROOT_CAUSE = coordinator used host PID values inside the container PID namespace, so it falsely inferred completion and ran the integrity audit before final artifacts existed.`

## 8. Rerun eligibility recommendation

`SAFE_TO_RERUN_WAVE1 = False` while all three originals are alive.

Required engineering fix for future observation only: track either host PIDs from the host namespace, or map and monitor container PIDs (`303559`, `303505`, `303613`) from the container namespace; require PID natural exit plus final artifact integrity. Preserve all failed audit and duplicate-wrapper records. New output paths are not currently required because no rerun is authorized and no output-path contamination is evidenced.

## 9. Stop condition

Wave-2 remains not launched. No formal metrics, reruns, or configuration changes were performed.
