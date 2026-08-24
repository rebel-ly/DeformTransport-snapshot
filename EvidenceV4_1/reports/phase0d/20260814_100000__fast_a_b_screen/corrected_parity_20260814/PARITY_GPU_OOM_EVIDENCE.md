# Parity GPU OOM Evidence

Generated 2026-08-14. This is a read-only evidence record. No generation, retry, signal, debugger attach, source change, or GPU3 use was performed to create it. All timestamps below are UTC (`Z`).

## 1. Scope and source records

| Arm | Wrapper directory | stdout | stderr | exit record | completion record | final MP4 |
|---|---|---|---|---|---|---|
| A2 | `a2_gpu0_hardgate` | `stdout.log` | `stderr.log` | `exit_code.txt` | `completion.marker` | `santa_correct_v3d_seed000.mp4` absent |
| B2-G1 | `b2_gpu1_hardgate` | `stdout.log` | `stderr.log` | `exit_code.txt` | `completion.marker` | `santa_correct_v3d_seed000.mp4` absent |

Raw command forms used for this evidence were read-only:

```text
rg -n -C 20 'torch\.OutOfMemoryError|CUDA out of memory' <stderr.log>
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
ps -o user,pid,ppid,lstart,etime,stat,pcpu,rss,cmd -p <PID>
tr '\0' ' ' < /proc/<PID>/cmdline
sed -n '1,22p' /proc/<PID>/status
```

## 2. A. Directly proven facts: A2 raw OOM evidence

```text
A2_START_TIME = 2026-08-14T07:45:36Z
A2_END_TIME = 2026-08-14T08:23:28Z
A2_EXIT_CODE = 1
A2_COMPLETION_MARKER = FAILED
A2_FINAL_MP4_EXISTS = False
```

The following is the raw terminal traceback context (lines 44--64 of `a2_gpu0_hardgate/stderr.log`):

```text
44-    return forward_call(*args, **kwargs)
45-           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
46-  File "/workspace/Wan-Move/wan/modules/model.py", line 302, in forward
47-    y = self.self_attn(
48-        ^^^^^^^^^^^^^^^
49-  File "/workspace/tools/miniforge3/envs/wan-move/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
50-    return self._call_impl(*args, **kwargs)
51-           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
52-  File "/workspace/tools/miniforge3/envs/wan-move/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
53-    return forward_call(*args, **kwargs)
54-           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
55-  File "/workspace/Wan-Move/wan/modules/model.py", line 150, in forward
56-    q=rope_apply(q, grid_sizes, freqs),
57-      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
58-  File "/workspace/tools/miniforge3/envs/wan-move/lib/python3.11/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
59-    return func(*args, **kwargs)
60-           ^^^^^^^^^^^^^^^^^^^^^
61-  File "/workspace/Wan-Move/wan/modules/model.py", line 70, in rope_apply
62-    return torch.stack(output).float()
63-           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
64:torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 620.00 MiB. GPU 0 has a total capacity of 44.32 GiB of which 84.31 MiB is free. Process 387090 has 36.25 GiB memory in use. Process 413847 has 4.07 GiB memory in use. Process 413851 has 3.89 GiB memory in use. Of the allocated memory 35.45 GiB is allocated by PyTorch, and 301.95 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management.
```

`A2_FINAL_ERROR` is therefore the quoted `torch.OutOfMemoryError`. The raw message reports total capacity, free memory, requested allocation, allocated memory, and reserved memory. It does **not** provide a separately labelled “non-PyTorch memory” field.

## 3. A. Directly proven facts: B2-G1 raw OOM evidence

```text
B2_G1_START_TIME = 2026-08-14T07:45:36Z
B2_G1_END_TIME = 2026-08-14T08:23:32Z
B2_G1_EXIT_CODE = 1
B2_G1_COMPLETION_MARKER = FAILED
B2_G1_FINAL_MP4_EXISTS = False
```

The following is the raw terminal traceback context (lines 44--64 of `b2_gpu1_hardgate/stderr.log`):

```text
44-    return forward_call(*args, **kwargs)
45-           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
46-  File "/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay/wan/modules/model.py", line 302, in forward
47-    y = self.self_attn(
48-        ^^^^^^^^^^^^^^^
49-  File "/workspace/tools/miniforge3/envs/wan-move/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
50-    return self._call_impl(*args, **kwargs)
51-           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
52-  File "/workspace/tools/miniforge3/envs/wan-move/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
53-    return forward_call(*args, **kwargs)
54-           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
55-  File "/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay/wan/modules/model.py", line 150, in forward
56-    q=rope_apply(q, grid_sizes, freqs),
57-      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
58-  File "/workspace/tools/miniforge3/envs/wan-move/lib/python3.11/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
59-    return func(*args, **kwargs)
60-           ^^^^^^^^^^^^^^^^^^^^^
61-  File "/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay/wan/modules/model.py", line 65, in rope_apply
62-    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
63-                             ~~~~^~~~~~~~~
64:torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.21 GiB. GPU 0 has a total capacity of 44.32 GiB of which 1.16 GiB is free. Process 387089 has 35.04 GiB memory in use. Process 413852 has 4.20 GiB memory in use. Process 413848 has 3.89 GiB memory in use. Of the allocated memory 34.24 GiB is allocated by PyTorch, and 301.95 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management.
```

`B2_G1_FINAL_ERROR` is therefore the quoted `torch.OutOfMemoryError`. “GPU 0” in this message is the process-visible CUDA ordinal; it is not by itself proof that the host physical assignment was GPU0 rather than the launcher-assigned GPU1.

## 4. B. Current external GPU-process evidence

Snapshot time: `2026-08-14T09:40:29Z`.

| Host GPU | UUID | PID | user / UID | process name from nvidia-smi | VRAM | UTC process start | command-line status |
|---|---|---:|---|---|---:|---|---|
| 0 | `GPU-14bb1875-6456-dba9-fde5-e1383c8d480b` | 413847 | `linjian` / 10001 | `[Not Found]` | 4166 MiB | 2026-08-14T08:00:37Z | readable via `ps` and `/proc` |
| 0 | `GPU-14bb1875-6456-dba9-fde5-e1383c8d480b` | 413851 | `linjian` / 10001 | `[Not Found]` | 3988 MiB | 2026-08-14T08:00:37Z | readable via `ps` and `/proc` |
| 1 | `GPU-0e2857f8-18bc-0f5b-c1ff-5b67f892cd60` | 413852 | `linjian` / 10001 | `[Not Found]` | 4298 MiB | 2026-08-14T08:00:37Z | readable via `ps` and `/proc` |
| 1 | `GPU-0e2857f8-18bc-0f5b-c1ff-5b67f892cd60` | 413848 | `linjian` / 10001 | `[Not Found]` | 3988 MiB | 2026-08-14T08:00:37Z | readable via `ps` and `/proc` |

For each of these PIDs, `ps` reported parent PID 1 and a live Python command beginning:

```text
/mnt/sdbd/home/linjian/anaconda3/envs/py38/bin/python3 run_baseline.py
```

The full `ps` and `/proc/<PID>/cmdline` command lines include that program’s dataset, seed, physics, loss, and experiment arguments. `/proc/<PID>/status` showed `Uid: 10001 ...`, `Gid: 1000 ...`, and `TracerPid: 0` for all four PIDs. The raw command output is preserved verbatim in the terminal evidence captured during this audit; the source logs and PIDs above remain independently queryable.

## 5. B. GPU contention timeline

```text
===== GPU CONTENTION TIMELINE =====
A2_START = 2026-08-14T07:45:36Z
B2_G1_START = 2026-08-14T07:45:36Z

GPU0_EXTERNAL_PID = 413847, 413851
GPU0_EXTERNAL_PID_START = 2026-08-14T08:00:37Z
GPU0_EXTERNAL_VRAM = 4166 MiB, 3988 MiB (09:40:29Z snapshot)

GPU1_EXTERNAL_PID = 413852, 413848
GPU1_EXTERNAL_PID_START = 2026-08-14T08:00:37Z
GPU1_EXTERNAL_VRAM = 4298 MiB, 3988 MiB (09:40:29Z snapshot)

A2_OOM_TIME = 2026-08-14T08:23:28Z (wrapper end time)
B2_G1_OOM_TIME = 2026-08-14T08:23:32Z (wrapper end time)

GPU0_EXTERNAL_STARTED_AFTER_A2 = True
GPU0_EXTERNAL_STARTED_BEFORE_A2_OOM = True
GPU1_EXTERNAL_STARTED_AFTER_B2_G1 = True
GPU1_EXTERNAL_STARTED_BEFORE_B2_G1_OOM = True
```

The OOM messages themselves name the same external PIDs: A2 names 413847 and 413851; B2-G1 names 413852 and 413848. The present snapshot confirms their host identities and currently allocated VRAM. The snapshot does not prove their exact VRAM allocation at the instant of each OOM, so that quantity is not inferred retrospectively.

## 6. B. B2-G2 control evidence

```text
START_TIME = 2026-08-14T07:45:37Z
CURRENT_TIME = 2026-08-14T09:40:29Z
STATUS = ACTIVE_PROGRESS
WRAPPER_PID = 386762
PYTHON_PID = 387213
GPU_UUID = GPU-56d1a97e-c16c-ebf6-4fc6-8466b32d0bbf
OUR_PROCESS_VRAM = 41092 MiB (current nvidia-smi snapshot)
EXIT_CODE = UNAVAILABLE (still running)
```

The pre-launch hard gate records `B2_G2_MATCH_CANONICAL_EXCEPT_SOURCE_AND_GPU = true`; its effective manifest records seed 0, 81 frames, `480*832`, bf16, 40 inference steps, shift 3, and the same Wan checkpoint and source inputs. Thus the **generation scale/config relevant to this control is the same parity configuration**, subject only to the pre-registered overlay source provenance and GPU assignment differences. Its continued progress establishes only that this configuration is not shown here to *inevitably* immediately OOM on an L40. It does not prove absence of future OOM or prove a cause for A2/B2-G1.

## 7. C. Pre-launch resource snapshot

`PRE_LAUNCH_CANONICAL_MANIFEST_GATE.json` records a passing manifest gate but has no persisted `resource_snapshot` or hardware-memory snapshot. Searches of the current corrected-parity directory found no raw pre-launch `nvidia-smi` output, GPU process snapshot, or GPU0/GPU1 free-VRAM fields for this launch.

```text
PRELAUNCH_RESOURCE_SNAPSHOT = UNAVAILABLE
A2_PRELAUNCH_GPU0_FREE_VRAM = UNAVAILABLE
B2_G1_PRELAUNCH_GPU1_FREE_VRAM = UNAVAILABLE
```

No missing snapshot has been reconstructed or fabricated.

## 8. Evidence classification

```text
CUDA_OOM_CONFIRMED_A2 = True
CUDA_OOM_CONFIRMED_B2_G1 = True
MODEL_INTRINSIC_OOM = NOT_SUPPORTED
EXTERNAL_GPU_PROCESS_PRESENT = True
EXTERNAL_PROCESS_ENTERED_AFTER_OUR_JOB = True
EXTERNAL_PROCESS_PRESENT_BEFORE_OOM = True
EXTERNAL_RESOURCE_CONTENTION_AS_OOM_CAUSE = STRONGLY_SUPPORTED
```

“Strongly supported” is limited to this documented temporal and memory relationship: external processes started after both parity jobs, before their OOM exits, are named in the corresponding OOM messages, and currently occupy the reported GPUs. It is not a responsibility attribution and does not prove a counterfactual execution outcome absent those processes. The absent pre-launch resource snapshot and unavailable instantaneous VRAM history remain limitations.

## 9. Administrator-facing summary (<=200 Chinese characters)

2026-08-14T07:45:36Z 启动的 A2、B2-G1 分别于 08:23:28Z、08:23:32Z 以 CUDA OOM 退出。日志记录当时仅余 84 MiB/1.16 GiB 可用显存，并列出外部 PID。该四个 PID 于 08:00:37Z 启动、位于对应 GPU，当前占约 4 GiB/进程。请管理员核验共享 GPU 资源竞争；本报告不作责任判断。
