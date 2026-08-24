# 决策日志

## 2026-08-04 23:50 +08:00

- 保留现有 dirty worktree，先执行只读审计。
- 四卡均有活动计算负载时不启动 GPU smoke。
- 未认证访问 OpenAI API 根路径返回 HTTP 421，只作为代理、TCP 和 TLS 可达证据，不解释为 API 授权成功。

## 2026-08-04 23:58 +08:00

- 五个初始 GPU 进程均分类为 A：其他用户活动任务，与 DeformTransport 无关。
- 可用 GPU 数为 0，继续 CPU 安全任务并在任何 CUDA 分配前重新门禁。

## 2026-08-05 00:00—00:20 +08:00

- 依赖按缺失导入逐项、固定版本、带 constraints 安装；每一步保留 dry-run、freeze 和 pip check，不升级保护包。
- 现有模型文件哈希正确，不重复下载。
- Baseline 必须使用官方默认 `generator`，不擅自切换 EMA；三组对照必须固定 checkpoint、seed、输入、时序、mask 和 GPU。

## 2026-08-05 00:23—00:45 +08:00

- 用户报告 GPU3 空闲后，实际门禁发现新出现的其他用户 PID 257153；依据不共享条款放弃抢占。
- 禁止用 sleep、空转循环或纯显存分配抢卡；Baseline 因 `final_sim` 缺失时，选择真实 Wan VAE + transport artifact 三路解码作为下一项可执行 GPU smoke。
- 当前实现先完成 provenance 和完整生成验证，再决定是否引入 softmax splatting 或逐去噪步 transport。
