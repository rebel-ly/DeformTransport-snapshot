# Phase0D canonical-preview regression and SDEdit freeze

## 阶段目标

复用已通过的 preview timeline，完成 B1/B2/B4-B6 的冻结门禁；只在全部硬门禁通过后才允许 C1/C2。

## 审计问题与数据

Canonical preview 是 `preview_reconstruction_20260814/frame_0000.png` 至 `frame_0080.png`，81 帧，832×480，S0–S800。其 manifest SHA-256 为 `11a184392508a5a96597a85b3030c6995506d6c12c3a7111e8e6b8aae8dc575b`。B1/B2 使用 frozen corrected-v2 evaluator 的同一 `to_common` 与 Lab patch TC-MAR 语义；该 evaluation-only 480→464 bicubic 变换没有用于 VAE。

## 关键结果

- B0: historical 65.71344520089995 来自 canonical `final_sim/frames/frame_0000.png..frame_0080.png`，S0–S800，输入 832×480；历史脚本 `audit_r2_rw_support.py` SHA-256 `72a1c623a65b22f8b44d5c1b663220ddbb42d750d55812e3989e926137cbb63b` 在 evaluator 之前额外 `/255`，然后 `to_common` 再 `/255`。
- B1: correctly-normalized canonical TC-MAR mean 4.620052, median 2.312909, p95 17.529045，较历史值 -61.093393（-92.969%）。这是已解释的 MATERIAL_DIFFERENCE，而不是 unexplained regression。
- B2: full-frame occupancy mean 0.156085；object-region coverage mean 0.386490；track patch FULL support 0.863813。valid-supported TC-MAR mean 4.132777，invalid/hole-intersection mean 8.795775；high TC-MAR 不是 hole-dominated，input-alignment risk 为 MODERATE。
- B3: 第一次 VAE-only 调用在模型加载前缺少 `wan` import path；修正 `PYTHONPATH` 后未生成 result JSON，且该容器任务已退出。因此 `PREVIEW_VAE_ENCODE=UNRESOLVED`，没有将其误记为通过。
- B4: canonical config raw `denoising_step_list=[500,250]`、`warp_denoising_step=True`。RW FlowMatch scheduler（1000 steps，shift=5）将 500 映射为 timestep/sigma 833.333333；`add_noise` 实际为 `(1-sigma)*x0 + sigma*epsilon`，故 signal=0.166667、noise=0.833333、SNR=0.04、logSNR=-3.218876。
- B5: Wan `FlowUniPCMultistepScheduler` 40-step、shift=3 的 exact nearest sigma mapping 是 index 15、timestep 833.333333、sigma 0.833333，absolute error 0，剩余 25 个 denoising steps。
- B6: scheduler 的 `set_timesteps` 确实重置 `model_outputs`、`lower_order_nums`、`_step_index` 与 `_begin_index`；随后 `set_begin_index(15)` 可使一个新 scheduler 的 history 空白且数学上有效。但是 frozen `WanMove.generate` 总是从 fresh random `noise` 开始、遍历完整 timesteps，且不接受 preview latent、begin index、共享 epsilon 或 K=0/K=1257 transport switch。以当前正式接口无法调用这条安全 intermediate-start path；改造它将改变现有 runner/model semantics。

## PASS/FAIL 与后续影响

`B_PREVIEW_METRIC_REGRESSION=PASS_EXPLAINED`，但 `PREVIEW_VAE_ENCODE=UNRESOLVED` 且 `WAN_INTERMEDIATE_START_SEMANTICS=FAIL_AT_FROZEN_RUNNER_INTEGRATION`。因此 `C_PREVIEW_SDEDIT_READY=False`，C1/C2 均未启动。没有执行任何额外 arm、seed、sweep 或 server-side LLM/API。

## 遗留问题

若要启动 C，需要先获得可审计、已冻结的 runner 接口，能够将 actual 832×480 preview VAE latent、shared epsilon、index 15 和 K OFF/ON 注入 Wan sampler；这不是当前 frozen generator 的已有能力，不能在本协议下自行改写。
