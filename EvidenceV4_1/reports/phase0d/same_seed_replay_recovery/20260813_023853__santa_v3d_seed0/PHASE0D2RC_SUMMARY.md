# Phase0D-2R-C — Container Runtime Recovery + Recovered Same-Seed Replay Determinism Audit

## 1. 阶段目标

Translate the known host/container execution binding, then run the authorized corrected-v2 Santa Correct-V3D seed-0 Replay A/B and compare decoded RGB exactly. The protocol requires stopping if a new immediate runtime blocker occurs.

## 2. 审计问题

The frozen runner has host `/mnt/sdbd/home/liuyu_qyh/...` paths while the actual formal Docker environment exposes corresponding `/workspace/...` mounts. The container's actual runtime also had to pass an import smoke before a path-only execution wrapper or any generation was authorized.

## 3. 使用的数据

Used the frozen 0D-1R runner, 0D-2R-B source contract, actual `deformtransport-dev` container, the corrected-v2 N=1257/T=81 asset set, and the authorized 0D-2R-C protocol. All prior evidence, including the original Phase0D-2 `ENGINEERING_FAIL` and manual Attempt 1, was retained.

## 4. 使用的方法

Recorded container context and path map; verified container asset existence and available asset SHA256 values; verified the three re-frozen source hashes; ran only CPU/import-level container smokes; and recorded GPU, memory, and disk preflight. No container wrapper, translated runner, formal manifest, child process, GPU generation, model load, MP4, or decoder execution was created after the smoke gate failed.

## 5. 关键命令/脚本

`docker exec deformtransport-dev` was used for read-only context, asset, SHA, and import checks. Container `/usr/bin/python` (Python 3.8.10) ran `import wan.utils.prompt_extend` and `import generate`; both failed before the recovered DashScope import point.

## 6. 关键结果

The container mounts `/workspace/DeformTransport`, `/workspace/Wan-Move`, and `/workspace/DeformTransport_EvidenceV4_1` exist. Corrected-v2 asset hashes match the frozen formal values, and the runtime-source gate passed: trajectory, wan_move, and patched prompt_extend hashes are exact. GPU1 and GPU2 were idle with 7 MiB displayed use and host available RAM was 102 GiB.

However, the container's only discovered Python was `/usr/bin/python` 3.8.10, and it lacks `easydict`. Both import smokes stop at `wan/configs/wan_move_14B.py:3`, `from easydict import EasyDict`, with `ModuleNotFoundError`, before DashScope, checkpoint/model load, diffusion, VAE, GPU computation, output creation, or any network activity.

## 7. PASS/FAIL/UNRESOLVED 判断

Container execution context, path map, and runtime source gate passed. Required container import gates failed due to the new dependency blocker, so `PHASE0D2RC_STATUS = ENGINEERING_FAIL_RECOVERY_REPLAY`. This is an engineering failure and not a scientific or determinism failure.

## 8. 对后续实验影响

The protocol's generation authorization was never reached. A/B remain not started, no MP4 or decoded RGB exists, same-seed replay remains `NOT_EVALUATED`, and `PROCEED_TO_PHASE0D3=False`. A future separately authorized runtime recovery would need to address or select an equivalent container runtime for `easydict`; this phase did not install packages, alter environment, modify source, or automatically patch a second issue.

## 9. 遗留问题

The host/container path binding still requires a future path-only wrapper or translated runner, but it must not be executed until the new container dependency blocker is resolved under a separate authorization. The Phase0D scope remains `PAIRWISE_OUTPUT_STOCHASTICITY_ONLY`; no GT/quality metrics were calculated.
