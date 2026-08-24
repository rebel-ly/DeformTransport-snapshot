# GPU Process Provenance Audit

- Audit window: 2026-08-04 23:57:39–23:58:19 +08:00
- Read-only methods: nvidia-smi compute-app query, eight one-second pmon samples, ps, and permitted /proc metadata reads
- Process mutation: none
- Raw evidence: gpu_process_audit_raw.json

## Decision

All five GPU processes belong to user pengzhennan_gyj. Their cgroups are host user.slice sessions with no Docker identifier; none belongs to deformtransport-dev. Their commands are ESM2 protein-model jobs, and no existing DeformTransport server_runs or logs record references their PIDs.

The kernel denies liuyu_qyh access to these processes' CWD and file descriptors. Therefore their unrelated log/output paths were not scanned, and file-growth status is unavailable rather than assumed. Each process gained approximately five CPU seconds during a five-second sample and repeatedly reported nonzero per-process SM utilization. They are live computations, not zombies, stale allocations, or tasks merely holding GPU memory.

Every process is classification **A: another user's active task**. All four GPUs are unavailable. Available GPU count is **0**.

## GPU 0 — unavailable (A)

- UUID: GPU-14bb1875-6456-dba9-fde5-e1383c8d480b
- Board snapshot: 18,168 MiB used, 27,217 MiB free, 100% utilization, 89 C

### PID 153796

- Owner: pengzhennan_gyj
- GPU memory: 10,994 MiB
- Start / elapsed: 2026-08-04 04:58:03 +08:00 / 18:59:41
- State / activity: R, 100% CPU, +503 CPU ticks over five seconds
- SM samples: 78, 42, 86, 93, 93, 52, 18, 94 percent
- Full command: python main.py --mode meta --model esm2 --train_size 40 --train_batch 1 --eval_batch 52000 --lora_r 16 --learning_rate 1e-4 --epochs 100 --patience 15 --list_size 10 --max_iter 5 --retr_metric cosine --augment GEMME --meta_tasks 3 --meta_train_batch 16 --meta_eval_batch 128 --adapt_lr 5e-3 --adapt_steps 5 --cross_validation 4 --protein all
- CWD: Permission denied; no unrelated path scan attempted
- Parent: PID 153795, sh run_active.sh, PPID 1
- Cgroup: /user.slice/user-2107.slice/session-228.scope
- Docker / deformtransport-dev: no / no
- DeformTransport link: none found in command or project PID records; CWD/FD evidence unavailable
- Log/output growth: inaccessible by OS permission
- Diagnosis: active computation

### PID 207627

- Owner: pengzhennan_gyj
- GPU memory: 7,156 MiB
- Start / elapsed: 2026-08-04 17:20:40 +08:00 / 06:37:04
- State / activity: R, 99.5% CPU, +503 CPU ticks over five seconds
- SM samples: 16, 55, 12, unavailable, unavailable, 44, 73, unavailable percent
- Full command: python main_multimodal.py --mode meta --model esm2 --protein all --train_size 80 --train_batch 1 --eval_batch 52000 --lora_r 16 --learning_rate 1e-4 --epochs 100 --patience 15 --list_size 10 --max_iter 5 --retr_metric cosine --augment GEMME --meta_tasks 3 --meta_train_batch 16 --meta_eval_batch 128 --adapt_lr 5e-3 --adapt_steps 5 --cross_validation 4 --use_structure --structure_dir data/structures --gearnet_weight checkpoints/gearnet_edge.pth --freeze_gearnet
- CWD: Permission denied
- Parent: PID 207626, sh run_80.sh, parent PID 190683
- Cgroup: /user.slice/user-2107.slice/session-1887.scope
- Docker / deformtransport-dev: no / no
- DeformTransport link: none found; CWD/FD evidence unavailable
- Log/output growth: inaccessible by OS permission
- Diagnosis: active computation with transient pmon sampling gaps

## GPU 1 — unavailable (A)

- UUID: GPU-0e2857f8-18bc-0f5b-c1ff-5b67f892cd60
- Board snapshot: 8,336 MiB used, 37,048 MiB free; utilization fluctuated 90% to 26%, 87 C

### PID 154746

- Owner: pengzhennan_gyj
- GPU memory: 8,324 MiB
- Start / elapsed: 2026-08-04 04:58:57 +08:00 / 18:58:47
- State / activity: R, 100% CPU, +503 CPU ticks over five seconds
- SM samples: 96, 87, 96, 86, 75, 89, 51, 23 percent
- Full command: python main.py --mode meta --model esm2 --train_size 40 --train_batch 1 --eval_batch 52000 --lora_r 16 --learning_rate 1e-4 --epochs 100 --patience 15 --list_size 10 --max_iter 5 --retr_metric cosine --augment GEMME --meta_tasks 3 --meta_train_batch 16 --meta_eval_batch 128 --adapt_lr 5e-3 --adapt_steps 5 --cross_validation 4 --protein all
- CWD: Permission denied
- Parent: PID 154745, sh run_active.sh, PPID 1
- Cgroup: /user.slice/user-2107.slice/session-228.scope
- Docker / deformtransport-dev: no / no
- DeformTransport link: none found; CWD/FD evidence unavailable
- Log/output growth: inaccessible by OS permission
- Diagnosis: active computation

## GPU 2 — unavailable (A)

- UUID: GPU-56d1a97e-c16c-ebf6-4fc6-8466b32d0bbf
- Board snapshot: 12,692 MiB used, 32,692 MiB free; utilization fluctuated 16% to 75%, 87 C

### PID 161499

- Owner: pengzhennan_gyj
- GPU memory: 12,680 MiB
- Start / elapsed: 2026-08-04 05:03:44 +08:00 / 18:54:00
- State / activity: S at one ps instant, but 100% CPU and +502 CPU ticks over five seconds
- SM samples: 56, 44, 76, 52, 76, 42, 57, 93 percent
- Full command: python main_multimodal.py --mode meta --model esm2 --protein all --train_size 20 --train_batch 1 --eval_batch 52000 --lora_r 16 --learning_rate 1e-4 --epochs 100 --patience 15 --list_size 10 --max_iter 5 --retr_metric cosine --augment GEMME --meta_tasks 3 --meta_train_batch 16 --meta_eval_batch 128 --adapt_lr 5e-3 --adapt_steps 5 --cross_validation 4 --use_structure --structure_dir data/structures --gearnet_weight checkpoints/gearnet_edge.pth --freeze_gearnet
- CWD: Permission denied
- Parent: PID 161498, sh run_20.sh, PPID 1
- Cgroup: /user.slice/user-2107.slice/session-228.scope
- Docker / deformtransport-dev: no / no
- DeformTransport link: none found; CWD/FD evidence unavailable
- Log/output growth: inaccessible by OS permission
- Diagnosis: active computation; one instantaneous S state is normal scheduling, not a stall

## GPU 3 — unavailable (A)

- UUID: GPU-111b4686-bb10-d8db-87fd-401b36dcbdf3
- Board snapshot: 12,502 MiB used, 32,882 MiB free; utilization fluctuated 63% to 68%, 88 C

### PID 162085

- Owner: pengzhennan_gyj
- GPU memory: 12,490 MiB
- Start / elapsed: 2026-08-04 05:04:14 +08:00 / 18:53:30
- State / activity: R, 100% CPU, +502 CPU ticks over five seconds
- SM samples: 38, 49, 33, 49, 79, 50, 61, 41 percent
- Full command: python main_multimodal.py --mode meta --model esm2 --protein all --train_size 60 --train_batch 1 --eval_batch 52000 --lora_r 16 --learning_rate 1e-4 --epochs 100 --patience 15 --list_size 10 --max_iter 5 --retr_metric cosine --augment GEMME --meta_tasks 3 --meta_train_batch 16 --meta_eval_batch 128 --adapt_lr 5e-3 --adapt_steps 5 --cross_validation 4 --use_structure --structure_dir data/structures --gearnet_weight checkpoints/gearnet_edge.pth --freeze_gearnet
- CWD: Permission denied
- Parent: PID 162084, sh run_60.sh, PPID 1
- Cgroup: /user.slice/user-2107.slice/session-228.scope
- Docker / deformtransport-dev: no / no
- DeformTransport link: none found; CWD/FD evidence unavailable
- Log/output growth: inaccessible by OS permission
- Diagnosis: active computation

## Recomputed concurrency

N = min(0 available GPUs, 4, host-memory term) = 0

- Heavy generation concurrency: 0
- CUDA tests that allocate or compute are deferred
- CPU-only audit, source review, payload validation, dependency repair, unit tests, command reconstruction, existing-result metrics, and reporting may continue
- GPU availability must be re-audited before any CUDA job; low instantaneous utilization alone is insufficient while another user's compute process remains
