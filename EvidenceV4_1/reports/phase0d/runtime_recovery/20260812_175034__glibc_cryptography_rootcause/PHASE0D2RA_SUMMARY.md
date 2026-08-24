# Phase0D-2R-A — Runtime Dependency Root-Cause Audit

## 1. 阶段目标
对 Phase0D-2 在模型加载前发生的 runtime import failure 做 CPU-only、read-only 根因定位。

## 2. 审计问题
确认 GLIBC/cryptography ABI、DashScope import path、formal prompt dependency 和可行恢复路径，不改变环境或 source。

## 3. 使用的数据
冻结 Phase0D-2 status、engineering failure、Run A/B stderr；冻结 Phase0D-1R runner；当前 OS、runner Python metadata 和 Wan-Move source。

## 4. 使用的方法
只读 glibc inventory、importlib.metadata、ELF `file`/`ldd`/`readelf`、wheel metadata、source-line audit及有界 local environment metadata inventory。没有 import cryptography、generation、GPU computation、安装或 source edit。

## 5. 关键命令/脚本
`ldd --version`, `getconf GNU_LIBC_VERSION`, `readelf --version-info`, `importlib.metadata`, `python -m pip debug --verbose` 和 `rg` source audit。

## 6. 关键结果
CentOS 7 provides GLIBC 2.17. Runner env contains cryptography 50.0.0 wheel `cp311-abi3-manylinux_2_28_x86_64`; `_rust.abi3.so` requires GLIBC 2.18, 2.25 and 2.28 (max 2.28). DashScope 1.26.6 is unconditionally imported at prompt_extend.py:12 due generate.py:20, even though frozen formal runner does not enable prompt extension. Thus failure occurs before model/checkpoint, diffusion, VAE, GPU work, or output creation.

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0D2RA_STATUS = ROOT_CAUSE_IDENTIFIED`。这是 engineering root cause, not a scientific/determinism result.

## 8. 对后续实验影响
Do not rerun Phase0D-2 until main dialogue authorizes a recovery. Recommended minimal candidate is R2: make DashScope lazy/optional only for prompt-extension use, then re-freeze source and import-smoke the formal no-extension path.

## 9. 遗留问题
No already-existing equivalent compatible runtime environment was found. R1 needs an approved compatible package artifact and environment re-freeze. Formal GT evaluator remains unresolved. Launcher must separately capture child exit codes in a future replay.
