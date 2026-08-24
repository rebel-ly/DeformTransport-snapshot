# 代码修改索引

Git 基线：`feature/trajectory-probe` @ `c98b2724563a64d96186d1bf4e6f9b8952ed9f48`。完整未提交 diff 见 `GIT_DIFF.patch`。以下均未提交。

| 文件 | 修改目的 | 类型 | 已测试 | 证据 | 建议 |
|---|---|---|---|---|---|
| `.gitignore` | 忽略本地 venv 与 `artifacts/` | 工程配置 | 不适用 | diff | 人工确认后保留 |
| `case_simulation.py` | 按配置导出 flow 与材料点轨迹 | 正确性/科研数据链 | 是 | 轨迹探针、official 4f/81f 仿真 | 建议保留，复核兼容性 |
| `demo_web/simulation/utils.py` | torchvision 视频写出缺失时回退 imageio | 环境兼容 | 部分 | compile/import；81f assembler 未使用此函数而仍因 PyAV 失败 | 建议统一写出路径后保留 |
| `demo_web/simulation_engine.py` | 接入 `PointTrajectoryRecorder`；增加 compute-only visualizer | 正确性与无头兼容 | 是 | official 4f exit0；81f simulation exit0 | 建议保留，需审查私有 visualizer API |
| `infer_sim.py` | 增加 transport artifact 和 correct/shuffled/flow/blend 模式 | 实验性方法接入 | 是 | runtime injection audit；完整 proxy generator 五方法 | 建议保留并补官方 case 回归 |
| `simulation/genesis_simulator.py` | 原批处理路径记录材料点身份轨迹 | 正确性/科研数据链 | 部分 | compile、相关测试；主要成功证据来自 interactive 路径 | 建议保留，补原生 case_simulation 回归 |
| `simulation/utils.py` | imageio 视频写出兼容 fallback | 环境兼容 | 部分 | compile/import | 建议与 demo_web 实现去重后保留 |
| `wan/modules/attention.py` | FlashAttention 不可用时 SDPA fallback；补 q/k length、window、causal、scale 和 dtype | 正确性修复 | 是 | 两个 SDPA 审计目录；GPU 数值等价；25/25 单测 | 建议保留 |
| `deform_transport/trajectory.py` | 持续材料点身份及投影轨迹记录 | 新方法基础设施 | 是 | transport 单测与 official trajectory report | 建议保留 |
| `deform_transport/pipeline_integration.py` | transport artifact 模式映射与 shape/dtype 校验 | 正确性/集成 | 是 | `tests/test_pipeline_integration.py`；runtime audit | 建议保留 |
| `scripts/run_realwonder_trajectory_probe.py` | official case 的 Genesis + `_MinimalSVR` 轨迹/flow 探针 | 审计脚本 | 是 | official 4f、81f simulation | 建议保留 |
| `scripts/run_precomputed_demo_simulation.py` | 预计算 demo final_sim 路径 | 实验脚本 | 仅 compile | 尚无成功完整运行证据 | 保留但标记未验收 |
| `scripts/assemble_final_sim_from_trajectory.py` | 由无损轨迹帧组装 final_sim | 实验脚本 | 失败到达 | 81f chain 在视频写出因 PyAV 失败 | 修复写出兼容后再验收 |
| `scripts/export_transport_ready.py` | 从轨迹结果生成 transport payload | 实验脚本 | proxy/探针路径已测 | transport artifacts | 建议保留 |
| `scripts/run_wan_vae_transport_probe.py` | Wan VAE 编码→transport→解码闭环 | 实验脚本 | 是 | `WAN_VAE_E2E_21F_20260805_022223` | 建议保留 |
| `tests/test_pipeline_integration.py` 及 `tests/` | transport/模式/契约回归 | 测试 | 是 | post_full_generator 25/25 | 建议保留 |
| `04_smoke/final_sim_noise_reconstruction_queued/generate_noise.py` | 将硬编码 21 帧改为由 config 推导帧数 | RUN 专用实验修复 | 未在 81f 执行到 | 81f chain 止于 stage03 | 保留现场，人工审查后再跑 |

注意：未跟踪目录很多，`UNTRACKED_FILES.txt` 是完整清单；本索引只列科研相关源码，不代表所有未跟踪文件均应提交。
