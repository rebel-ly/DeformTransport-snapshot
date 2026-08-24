# DeformTransport 新会话/人工续跑提示词

接管 `/mnt/sdbd/home/liuyu_qyh/DeformTransport`，RUN_ID 为 `20260804_234925_autonomous_deformtransport`。首先只读查看：

1. `server_runs/20260804_234925_autonomous_deformtransport/human_handoff/HUMAN_HANDOFF.md`
2. 同目录的 `CURRENT_STATE.json`、`ACTIVE_PROCESSES.txt`、`OPEN_ISSUES.md`、`RESULT_INDEX.md`、`CODE_CHANGE_INDEX.md`、`NEXT_COMMANDS.sh`
3. `04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719/{current_stage.txt,exit_code.txt,stderr.log}`

项目目标：在材料点身份保持 transport 下，对完整 RealWonder 做严格 Baseline/Correct/Shuffled/Flow 对照，再扩展到 official 与机器人可变形物体案例。

已验证：transport GPU parity；21 帧 Wan VAE proxy 闭环；Santa proxy 完整 generator 的 Baseline/Correct/Shuffled/Flow/Blend、seed0 确定性复跑和 seed1；transport 运行时注入；SDPA 数值等价；25/25 单测；official Santa 4 帧 Genesis+`_MinimalSVR` smoke。现有 proxy 结果不能视为 future GT，视觉尚未人工确认。

当前环境：容器 `deformtransport-dev`；生成环境 `/workspace/tools/miniforge3/envs/realwonder-gen`；仿真 venv `/workspace/tools/venvs/deformtransport-sim`；GL overlay `/workspace/tools/conda-libs/deformtransport-gl`；Genesis commit `3aa206cd84729bc7cc14fb4007aeb95a0bead7aa`。分支 `feature/trajectory-probe`，HEAD `c98b2724563a64d96186d1bf4e6f9b8952ed9f48`，工作树脏，禁止清理或自动提交。

当前进程：唯一 watcher PID `291997` 存活且保持原状。旧安装 PID `86928/86942` 不存在。81 帧 chain PID `89794` 已退出，exit 1，停在 `03_assemble_final_sim`；81 帧物理仿真/flow/trajectory 成功，因 assembler 使用 torchvision 写视频但缺 PyAV 失败。trio scheduler 仅准备、从未启动。

下一优先级：人工审查并复用已有 `simulation_source`，修复或绕开视频写出，生成新的唯一 final_sim 目录；完成契约校验后才运行 RAFT/noise、transport，再在同一 GPU 顺序执行 official Baseline→Correct→Shuffled。先人工观看已有五方法视频，再决定方法迭代。不要重复 Santa proxy 已通过实验。

安全边界：不要启动未审查 GPU 任务；不要停止/重启 watcher；不要干预其他用户进程；不要安装/下载而未获授权；不要 `git reset/clean/restore/pull`；不要删除/覆盖资产；不要声称优于 RealWonder/Flow、完成机器人案例或完成项目。
