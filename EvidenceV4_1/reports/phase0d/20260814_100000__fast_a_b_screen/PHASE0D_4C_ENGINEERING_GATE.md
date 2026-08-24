# Phase0D-4C strict engineering gate

## 阶段目标

在不改变默认 Wan-Move 语义的前提下，验证 Preview-SDEdit 的工程接口；只有 E1–E7 全部通过才可启动 C0/C1/C2。

## 已冻结基线与数据

- Formal DT-FULL runner: `reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/scripts/run_dtfull_container_exact.sh`.
- Wan sources before any attempted modification: `wan_move.py` SHA-256 `aca79f9cc4bf32ea363c4440ed2c7e7d90ef5aa763f3e96ae6c2b8eff35c1857`; `fm_solvers_unipc.py` SHA-256 `0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9`.
- E1 input remained canonical 81-frame 832×480 preview. No 832×464 evaluation transform was applied before VAE.

## E1 — actual Wan VAE encode

The actual VAE-only output was persisted before the reporting step returned: `preview_wan_vae_latent_e1_832x480.npy`, SHA-256 `9d71791f70fa519001708d0986c4b1cee297941b8f66a3d1a15e03ef8ce8bb8f`, float32 shape `[16,21,60,104]`, min `-3.6171555519`, max `4.8516516685`, mean `0.1177850813`, finite `2096640`, nonfinite `0`. This is raw `C,T,H,W`; the batch-equivalent Wan representation is `[1,16,21,60,104]`. Therefore `PREVIEW_VAE_ENCODE=PASS`, `PREVIEW_LATENT_COMPATIBLE_WITH_WAN=True`, and `NO_EVAL_RESIZE_BEFORE_VAE=True`.

## E2/E3 — source-derived SDEdit mapping

Canonical RW config supplies raw `[500,250]`; warp mapping selects actual first drop-in timestep `833.333333` and flow sigma `0.8333333333`. The actual RW noising code is `(1-sigma)*x0 + sigma*epsilon`; signal/noise coefficients are `0.1666666667/0.8333333333`, SNR `0.04`, logSNR `-3.218875825`.

Current Wan UniPC with 40 steps and shift 3 selects index `15`, timestep `833.333333`, sigma `0.8333333333`, absolute sigma error `0`, and 25 denoising steps from start.

## E5 — UniPC semantics

`set_timesteps` resets `model_outputs`, `timestep_list`, `lower_order_nums`, `last_sample`, `_step_index`, and `_begin_index`. `set_begin_index(15)` makes the native first `step()` initialize `_step_index=15`; the existing `lower_order_nums` warm-up produces a clean first-order then second-order start rather than fabricated history. Thus the scheduler-native semantic is valid if its injection interface is reached.

## E4/E6 status and stop condition

The frozen `WanMove.generate` offers no arguments or hook for preview latent, frozen epsilon, start index, or transport-on/off pairing. A minimal opt-in source patch is required at its internal noise/scheduler construction point. `apply_patch` against `/mnt/sdbd/home/liuyu_qyh/Wan-Move/wan/wan_move.py` was rejected by the active workspace write boundary; it was not bypassed. A copied sampler or unvalidated monkey-patch would violate the protocol's thin-adapter and disabled-path-parity requirements.

Accordingly `ADAPTER_DISABLED_PATH_PARITY=UNRESOLVED`, `WAN_INTERMEDIATE_START_SEMANTICS=UNRESOLVED_AT_RUNNER_INTEGRATION`, `PAIRED_INITIAL_NOISE=False`, and `ENGINEERING_GATE=FAIL`. No C0/C1/C2 task was launched; no metrics were inspected; no GPU3, sweep, new seed, new case, Shuffled, server-side Codex, or LLM API was used.

## 后续影响

The scientific route remains untested rather than rejected. Continuing requires an approved writable location for the minimal opt-in adapter/source integration, followed by the mandatory disabled-path parity gate.
