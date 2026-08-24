# F4 Connection-Loss Hardening

Audited at 2026-08-13T11:49:32+08:00 before the anticipated connection outage.

| Arm | Host PID | Container PID | Host PPID | Container PPID | GPU binding | Started | Log |
|---|---:|---:|---:|---:|---|---|---|
| WM-0 | 87487 | 211282 | 1 | 0 | `CUDA_VISIBLE_DEVICES=1` / GPU1 | 2026-08-13T11:20:05+08:00 | `runtime/wm0_container2_stdout.log`, `runtime/wm0_container2_stderr.log` |
| DT-FRAG-PRUNE | 87492 | 211297 | 1 | 0 | `CUDA_VISIBLE_DEVICES=2` / GPU2 | 2026-08-13T11:20:05+08:00 | `runtime/frag_container2_stdout.log`, `runtime/frag_container2_stderr.log` |

Both host launch processes have PPID 1 and no controlling TTY; both container processes have no TTY and container PPID 0. Therefore neither initial generation depends on the current desktop/SSH PTY. They were not restarted or duplicated.

The deterministic watcher is a detached container execution, container PID 215674, no TTY, with persistent transition log `runtime/persistent_watcher_transitions.log`, stdout/stderr redirect targets for GRID100, and post-run stdout/stderr targets. It monitors only fixed PIDs 211282 and 211297. It holds an atomic `runtime/grid100_launch.lock` before the sole permitted GRID100 launch; it exits rather than duplicates if a lock, output, or completion marker already exists. It launches only after exactly one initial PID is gone and its frozen designated GPU reports at least 20,000 MiB free.

`PROCESS_DEPENDS_ON_CURRENT_SSH_PTY = False` for WM-0 and DT-FRAG-PRUNE.

`CURRENT_GENERATION_CONNECTION_SAFE = True`.

`WATCHER_PERSISTENCE_MODE = docker exec --detach inside deformtransport-dev`.

`WATCHER_CONNECTION_SAFE = True`.

`GRID100_DUPLICATE_LAUNCH_GUARD = runtime/grid100_launch.lock`.

`PERSISTENT_LOGGING = True`.

`CURRENT_GENERATIONS_RESTARTED = False`.

`CURRENT_GENERATIONS_DUPLICATED = False`.

`EXPECTED_SSH_OUTAGE_SURVIVAL = PASS`.
