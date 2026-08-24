# 未解决问题

## P0：官方 81 帧 final_sim 未完成

- 证据：chain PID `89794` 不存在；`exit_code.txt=1`；`current_stage.txt=03_assemble_final_sim`。
- 根因：`scripts/assemble_final_sim_from_trajectory.py` 调用 `torchvision.io.write_video`，生成环境未安装 PyAV。
- 已完成部分：81 帧 Genesis PBD、`_MinimalSVR` RGB/flow、材料点轨迹及 report。
- 禁止：不审查就重启整条 chain；不要覆盖 partial output。
- 最小人工决策：优先让 assembler 使用已有 imageio 兼容写出，或经批准补 PyAV；从现有 `simulation_source` 继续。

## P0：trio 尚未启动

`04_smoke/OFFICIAL_SANTA_81F_TRIO_20260805_050719` 仅有准备文件，scheduler 从未启动。只有 official final_sim 完整校验后才可人工启动，并保持同一 GPU 顺序 Baseline→Correct→Shuffled。

## P1：原始官方输入缺失

原始 `right_s1_21f` 无损 PNG 目录未找到。现有 21 帧 proxy 不可替代 future GT。需要从可信来源找回或从 official simulation 合法重建；来源不确定的大规模下载需用户批准。

## P1：视觉验收未完成

现有五方法视频和 contact sheet 尚未人工观看。定量结果混合，且 seed 波动显著；方法迭代前必须先做视觉故障分类。

## P1：方法结论不足

Correct 仅在部分 proxy 指标及 Wan VAE 信息携带测试中优于 Shuffled；Flow 的 RAFT mean EPE 更好。没有证据支持“优于 RealWonder/Flow”。

## P2：环境与兼容风险

- compute-only visualizer 修改了 Genesis 私有 `_visualizer` 生命周期方法，需版本锁定和 review。
- PyAV 缺失阻塞 assembler，但本次交接禁止继续安装。
- 视频兼容 fallback 在两个 utils 文件重复，后续宜统一。
- Torch `torch.load(weights_only=False)`、distutils 和 RAFT deprecated warnings 尚未处理，不是本次失败根因。

## P2：尚未完成的科研范围

机器人可变形物体 case、跨对象泛化、规模化评测、Flow 扩展、最多四轮有证据迭代和论文级总结均未完成。Santa 是风力布料案例，不是机器人操作案例。
