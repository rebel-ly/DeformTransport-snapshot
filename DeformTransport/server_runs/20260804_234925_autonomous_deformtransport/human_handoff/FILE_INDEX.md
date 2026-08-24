# 交接文件索引

## 必需的 16 个文件

1. `HUMAN_HANDOFF.md` — 总体阶段、环境、进程、失败、代码、结果、下一步和危险边界。
2. `CURRENT_STATE.json` — 机器可读状态；81 帧 chain/trio/watcher 状态。
3. `ACTIVE_PROCESSES.txt` — 原始进程快照及最终 PID 核验追加段。
4. `GPU_STATE.txt` — 2026-08-05 05:11 的四卡、compute PID 和 MemAvailable 冻结快照。
5. `ENVIRONMENT_SNAPSHOT.txt` — 宿主、容器、Python、环境、磁盘、软链接等。
6. `GIT_STATUS.txt` — 冻结 Git status。
7. `GIT_DIFF.patch` — 已跟踪源码的完整未提交 diff。
8. `GIT_DIFF_STAT.txt` — diff 统计。
9. `UNTRACKED_FILES.txt` — 完整未跟踪文件清单。
10. `CODE_CHANGE_INDEX.md` — 源码修改目的、类型、测试和保留建议。
11. `RESULT_INDEX.md` — 关键实验的命令入口、输出、metrics、日志、exit、SHA 和结论边界。
12. `ASSET_INDEX.md` — checkpoint、Wan、proxy、official case 和环境资产。
13. `OPEN_ISSUES.md` — 当前阻塞、证据和最小人工决策。
14. `NEXT_COMMANDS.sh` — 仅供人工审查的续跑命令，GPU/环境修改命令全部注释。
15. `HUMAN_RESUME_PROMPT.md` — 新会话最小完整续跑提示词。
16. `FILE_INDEX.md` — 本文件。

## 辅助原始快照

- `GIT_IDENTITY_AND_CHECK.txt` — HEAD 与旧版 Git 命令输出；准确分支以 `GIT_STATUS.txt`/`CURRENT_STATE.json` 为准。
- `REALWONDER_GEN_PIP_FREEZE.txt` — 生成环境 freeze。
- `SIM_OVERLAY_PIP_FREEZE.txt` — 仿真 venv freeze。
- `RESULT_RAW_INVENTORY.txt` — 关键结果的原始文件清单。

## 权威性顺序

进程/chain 取 `CURRENT_STATE.json` 与 `ACTIVE_PROCESSES.txt` 的 `FINAL_HANDOFF_VERIFICATION`；完整命令取各实验目录 `command.sh`；退出码取各目录 `exit_code.txt`；代码内容取 `GIT_DIFF.patch`；结果结论取 `RESULT_INDEX.md` 与 metrics 原文件。
