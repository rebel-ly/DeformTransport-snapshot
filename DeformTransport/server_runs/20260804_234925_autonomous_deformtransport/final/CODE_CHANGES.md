# 代码与环境变更

## 项目源码

本轮没有修改 `infer_sim.py`、`deform_transport/`、`simulation/` 或测试等项目源码。接管时已有 dirty worktree 已完整保留，未 reset、checkout、覆盖或提交。

## 运行档案工具

仅在本次 `server_runs/20260804_234925_autonomous_deformtransport` 中新增或更新：

- GPU 进程审计、资源账本与中文报告；
- `final_sim` CPU 输入验证器；
- 视频 PSNR/SSIM/时序 proxy 脚本及自测；
- 真实 Wan VAE transport GPU smoke 脚本与安全启动器；
- Baseline/Correct/Shuffled 命令、队列、manifest 模板和监控脚本。

## Python 环境

在 `realwonder-gen` 中逐项新增固定版本 imageio 2.37.4、einops 0.8.2、omegaconf 2.3.0、peft 0.10.0 和 antlr4-python3-runtime 4.9.3。受保护的 torch/torchvision/torchaudio、numpy、diffusers、transformers、tokenizers、accelerate、sentencepiece 版本未改变；最终 `pip check` 通过。
