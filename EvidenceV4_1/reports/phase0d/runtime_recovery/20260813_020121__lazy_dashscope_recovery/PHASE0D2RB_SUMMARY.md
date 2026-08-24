# Phase0D-2R-B — Minimal Optional-Dependency Recovery and Replay Infrastructure Fix

## 1. 阶段目标

Apply the authorized minimal lazy-import recovery for the unused DashScope dependency, re-freeze the runtime source contract, and fix independent child exit-code capture for a future replay. No formal generation was started.

## 2. 审计问题

The frozen Phase0D-2 failed before model loading because `prompt_extend.py` eagerly imported DashScope, which imported an incompatible cryptography Rust binding requiring GLIBC 2.18+ on this GLIBC 2.17 host. The formal command has prompt extension disabled.

## 3. 使用的数据

Used the frozen 0D-1R corrected-v2 runner, frozen 0D-2/0D-2R-A evidence, the live Wan-Move source at commit `80c58a7d2ad175fa82a4d57f79f2a1415317dcfa`, and the authorized protocol. Historical Phase0D-2 evidence was read only and not altered.

## 4. 使用的方法

Recorded pre-patch source and critical SHA256 values; moved the DashScope import from module scope into `DashScopePromptExpander.__init__`; re-froze source and patch hashes; performed CPU-only `prompt_extend` and `generate` import smokes; executed a constructor-only negative control with no API call; and statically/mock-tested a new paired replay launcher.

## 5. 关键命令/脚本

The frozen interpreter imported `wan.utils.prompt_extend` and `generate` only. `run_phase0d2_replay_pair.sh --mock` launched `exit 0` and `exit 7` children and stored both independent return codes. No checkpoint was loaded, no diffusion/VAE path was executed, and no GPU generation was invoked.

## 6. 关键结果

Only `wan/utils/prompt_extend.py` changed (+8/-7). `trajectory.py` stayed `0c6bc94d8ce1f885f0333314a9b201a650163cd209b2a3b3f95b4f3a35a49dae` and `wan_move.py` stayed `aca79f9cc4bf32ea363c4440ed2c7e7d90ef5aa763f3e96ae6c2b8eff35c1857`. Module import passed with `DASHSCOPE_LOADED_AFTER_PROMPT_MODULE_IMPORT=False`; `generate` import also passed. The negative control reached the deferred DashScope import and reproduced the expected GLIBC failure before any network request, proving the functionality was deferred rather than removed. The mock launcher captured RC_A=0 and RC_B=7.

## 7. PASS/FAIL/UNRESOLVED 判断

`PHASE0D2RB_STATUS = PASS`. The recovery scope gate, both frozen-core SHA gates, import-smoke gates, static formal prompt-path gate, and launcher mock gate passed. This is an engineering recovery result, not a replay-determinism result.

## 8. 对后续实验影响

Future Phase0D-2R-C replay may use only the runtime source lineage in `runtime_source_contract.json` and the new launcher. Phase0D-2 remains permanently `ENGINEERING_FAIL` and `SAME_SEED_REPLAY=NOT_EVALUATED`; it has not been superseded or relabeled.

## 9. 遗留问题

Actual formal generation, seed replay, MP4 creation, decoding, and determinism evaluation remain unexecuted and require the next explicitly authorized phase. DashScope prompt extension remains unavailable on this host if deliberately selected, as expected from the unresolved incompatible dependency; it is not on the frozen formal path.
