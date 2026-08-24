# Phase 0D-4B preview R2 / SDEdit preflight — gated unresolved

## 1. 阶段目标

验证 RealWonder Santa coarse RGB preview 的完整时间线、有效栅格支持及未来 Preview-SDEdit 两 arm 的安全性。

## 2. 审计问题

预览是否是能够与 corrected-v2 T=81 timeline 一一对应的完整物理序列，而非 anchor-only 或错位资产。

## 3. 使用的数据

审计了 Santa corrected-v2 contract、RealWonder run log、transport-ready producer code 与 workspace 中实际可发现的 preview/coarse assets。

## 4. 使用的方法

代码证据显示 transport contract 要求完整 `coarse_rgb_frames`，且对非空输入要求其长度等于 frame count；冻结 corrected-v2 timeline 记录为 step0,10,...,800。RealWonder Santa log 记录其 SDEdit `denoising_step_list=[500,250]`。

## 5. 关键命令/脚本

只读检索 `build_aligned_transport_visibility_contract.py`、`transport_ready.py`、RealWonder Santa run logs 与整个 server-runs preview asset index。

## 6. 关键结果

未找到 Santa 的实际、完整 81 张 coarse-preview RGB frame 序列或其 authoritative manifest。因此无法将 producer 文件顺序与 step0...800、tracks、visibility 和 Wan-Move timeline 作事实闭环。仅有代码契约不能替代实际预览工件验证。

## 7. PASS/FAIL/UNRESOLVED 判断

`PREVIEW_TIMELINE_CONTRACT=UNRESOLVED`，故 `C_PREVIEW_SDEDIT_AUTHORIZED=False`、`C_PREVIEW_SDEDIT_READY=False`。B2--B8 不执行：coverage、hole attribution、preview TC-MAR decomposition、VAE latent encode、SNR mapping 与 C manifests 都依赖此首要 gate。

## 8. 对后续实验影响

不得启动 Preview-only 或 Preview+Transport。先恢复 authoritative preview frames/manifest 后，重新运行 timeline gate；再按门顺序进行有效栅格和 VAE/SDEdit preflight。

## 9. 遗留问题

真实 preview asset location/manifest 以及其 raster/depth/material support 仍缺失。记录的 `[500,250]` 是 RealWonder schedule evidence，不可在没有完整 preview timeline 与 Wan schedule mapping 前转化成 Wan start timestep。
