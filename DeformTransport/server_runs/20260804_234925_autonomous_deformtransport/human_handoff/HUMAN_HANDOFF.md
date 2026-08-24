# DeformTransport 人工交接

固化日期：2026-08-05（Asia/Shanghai）  
RUN_ID：`20260804_234925_autonomous_deformtransport`

## 1. 项目当前总体阶段

Santa 工程 proxy 的完整 RealWonder 生成对照已经完成；官方 Santa 的 Genesis + `_MinimalSVR` 最小 4 帧物理 smoke 已通过。官方 81 帧 chain 已自然失败并退出，物理仿真阶段成功，失败发生在 final_sim 组装阶段。项目整体尚未完成。

## 2. 已完成的关键事实

- material-point transport 真实 GPU 重算与历史 artifact 一致，exit 0。
- 21 帧 Wan VAE 编码→Correct/Shuffled transport→解码 proxy 闭环 exit 0。
- Santa proxy 的完整 RealWonder generator 已完成 Baseline、Correct、Shuffled、Flow、Blend α=0.5、Baseline seed0 复跑及 seed1。
- 同 seed Baseline 复跑视频 SHA256 完全一致。
- transport 运行时探针已证明 Correct/Shuffled payload 进入完整 generator 输入，Baseline 路径不加载 payload。
- SDPA 兼容修复通过 GPU 数值等价性验证；相关 CPU 单元测试累计 25/25 通过。
- 隔离仿真环境中官方 Santa 4 帧 Genesis PBD、`_MinimalSVR` RGB/flow/trajectory smoke exit 0。
- 官方 81 帧 Genesis PBD、渲染、flow 和轨迹导出已经完成；组装视频时因 PyAV 缺失退出，后续 RAFT/noise、transport、Wan VAE 未执行。

## 3. 当前尚未完成的主任务

- 人工审查后修复或绕开 81 帧 final_sim 组装的视频写出依赖，并从已有无损帧继续；不得盲目重跑物理仿真。
- 完成 official 81 帧 final_sim、RAFT/noise、transport artifact 的契约校验。
- `OFFICIAL_SANTA_81F_TRIO_20260805_050719` 仅准备，scheduler 从未启动；后续同一 GPU 顺序运行 Baseline→Correct→Shuffled。
- 人工视觉确认现有视频；机器人可变形物体案例；跨对象泛化、规模化评测及论文级结论。

## 4. 当前代码分支、HEAD和Git状态

- 分支：`feature/trajectory-probe`
- HEAD：`c98b2724563a64d96186d1bf4e6f9b8952ed9f48`
- 工作树有 8 个已跟踪修改文件及大量未跟踪科研文件。详见 `GIT_STATUS.txt`、`GIT_DIFF.patch`、`GIT_DIFF_STAT.txt`、`UNTRACKED_FILES.txt`。
- 未 commit、未 push、未 reset/clean/restore。

## 5. 当前有效环境及固定版本

- 生成环境：`/workspace/tools/miniforge3/envs/realwonder-gen`
- 隔离仿真 venv：`/workspace/tools/venvs/deformtransport-sim`
- 用户态 OpenGL：`/workspace/tools/conda-libs/deformtransport-gl`
- 仿真固定变量：`SETUPTOOLS_USE_DISTUTILS=stdlib`；`LD_LIBRARY_PATH=/workspace/tools/conda-libs/deformtransport-gl/lib:${LD_LIBRARY_PATH:-}`
- Torch `2.5.1+cu121`；CUDA runtime `12.1`；OpenCV headless `4.9.0.80`；cv2 `4.9.0`；trimesh `4.11.1`
- Genesis commit：`3aa206cd84729bc7cc14fb4007aeb95a0bead7aa`；PyTorch3D import 已验证。
- `realwonder-gen` 和隔离仿真 venv 的 `pip check` 均已通过。

## 6. 当前容器、宿主机和项目路径

- 宿主机：`gpu3`；用户：`liuyu_qyh`；CentOS 7 / kernel `3.10.0`
- 容器：`deformtransport-dev`；冻结时容器 ID `1638d03f7992...`，状态 Up
- 镜像：`image.ac.com:5000/gpu/admin/codeserver/vscode-pytorch:pytorch1.14-py3.8-cuda11.8`
- 宿主机项目：`/mnt/sdbd/home/liuyu_qyh/DeformTransport`
- 容器项目：`/workspace/DeformTransport`

## 7. 当前运行或残留 PID

- PID `291997`：唯一 GPU watcher，核验时 `/proc/291997` 存在，状态 `S (sleeping)`，命令为 `bash server_runs/20260804_234925_autonomous_deformtransport/gpu_scheduler/gpu_watcher.sh`，保持原状。
- PID `89794`：81 帧 chain，核验时不存在；`exit_code.txt=1`。
- PID `86928`、`86942`：核验时均不存在，旧安装已自然结束。
- 未发现其他当前 RUN_ID 活动任务；不得操作任何其他用户进程。

## 8. 当前 watcher 状态

唯一 watcher PID `291997` 仍存活，cwd 为项目根目录。没有创建第二个 watcher，没有停止或重启。

## 9. 当前隔离前端环境恢复状态

`_MinimalSVR` 位于 `demo_web/simulation_engine.py`，不走 SAM3D/MoGe/Flux。4 帧官方 smoke 首次因 distutils 断言失败，retry1 因无头 EGL context 失败，retry2 通过 compute-only visualizer 开关保留 Genesis 物理与 `_MinimalSVR` 渲染。原始 `right_s1_21f` 无损 PNG 资产在服务器上未找到；现有 21 帧输入仍是有损工程 proxy。

## 10. 当前正在安装或已经安装的包

没有仍运行的安装进程，本次交接未继续安装。安装日志位于 `01_environment/SIM_RUNTIME_RECOVERY_20260805_043358/`。完整包版本见 `REALWONDER_GEN_PIP_FREEZE.txt` 与 `SIM_OVERLAY_PIP_FREEZE.txt`。OpenCV headless 和用户态 OpenGL 已配置。

## 11. 所有失败过的关键分支及根因

- 初版 21 帧 Wan VAE 闭环 exit 1：proxy 的 `initial_rgb` 错指向 832×480 视频帧；按 RealWonder resize/crop 契约生成 512×512 v2 后修复。
- 官方 4 帧 smoke 初次失败：distutils 兼容断言。
- 官方 4 帧 retry1：无头环境 EGL context 创建失败。
- 官方 4 帧 retry2：仅禁用未使用的 Genesis visualizer 生命周期后 exit 0。
- 官方 81 帧 chain：PID `89794` 已退出，`current_stage.txt=03_assemble_final_sim`，`exit_code.txt=1`。81 帧仿真 report 成功；`torchvision.io.write_video` 因未安装 PyAV 抛出 `ImportError`。final_sim 仅为部分组装，不能视为 contract-complete；RAFT/noise 及之后阶段未运行。

## 12. 当前源码修改及其目的

详见 `CODE_CHANGE_INDEX.md`。主要修改为材料点轨迹导出、transport 模式注入、SDPA fallback 正确性、无头 compute-only Genesis、视频写出兼容层和实验/校验脚本。所有修改均未提交，必须人工审查。

## 13. 当前实验结果及结论边界

详见 `RESULT_INDEX.md`。Santa Baseline/Correct/Shuffled/Flow/Blend 是“完整 RealWonder generator + 工程 proxy 输入”的真实生成结果，但 proxy 不是 future GT。Correct 对 Shuffled 有部分指标优势；Flow 的 proxy RAFT mean EPE 最佳；seed 变化可大于若干方法差异。视觉材料尚未人工视觉确认。

不得宣称方法已优于 RealWonder、已优于 Flow、已完成机器人案例、Santa 是机器人操作案例、proxy 是 future GT 或项目已经完成。

## 14. 下一步人工操作顺序

1. 查看本目录与 `OPEN_ISSUES.md`，确认 watcher/chain 状态。
2. 审查 81 帧 `simulation_source/report.json`、81 张源帧及部分 final_sim；优先复用已有物理结果。
3. 决定使用现有 imageio 兼容写出还是在隔离环境补 PyAV；任何环境修改先人工确认。
4. 补齐 final_sim 并做 shape/dtype/finite/SHA256 校验，再运行 RAFT/noise 与 transport。
5. 资源检查通过后，人工取消 `NEXT_COMMANDS.sh` 中对应注释，同卡顺序运行 official Baseline→Correct→Shuffled。
6. 人工观看现有五组视频，再决定机器人 case 和有证据的第二轮方法改动。

## 15. 不得执行的危险操作

不得 `git reset/clean/restore/pull`，不得删除或覆盖资产，禁止自动 commit/push；禁止停止、暂停、renice 或修改其他用户进程；禁止把 proxy 当作 future GT；禁止未经输入、环境和资源核验就重启 chain、启动 trio 或运行 GPU 重任务。
