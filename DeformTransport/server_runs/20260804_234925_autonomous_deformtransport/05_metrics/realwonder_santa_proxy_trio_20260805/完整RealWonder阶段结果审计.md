# 完整 RealWonder 阶段结果审计

## 验收范围

本阶段首次完成了 Santa 21 帧输入在原生 `infer_sim.py` 完整 generator 中的 Baseline、Correct、Shuffled、Dense Flow、Correct-Blend、seed1 和 EMA 对照。首次 Baseline→Correct→Shuffled 严格在同一物理 GPU2 上、相同 checkpoint、seed=0、structured noise、prompt、config、分辨率和输出帧数下顺序执行，均 exit 0。

输入 `santa_21f_final_sim_proxy_v1` 已通过 shape/dtype/finite/帧数/配置/transport 契约校验，但它是由已有 Santa 模拟帧和历史有损视频资产重建的工程 proxy，不是官方原始 `final_sim`，也不是 future GT。下述指标只能说明对 coarse 引导的相对一致性和方法间差异，不能直接证明真实生成质量、物理真实性或跨对象泛化。

## 完整生成输出

| 变体 | seed | transport | 状态 | 视频 SHA256 前缀 |
|---|---:|---|---|---|
| Baseline | 0 | 无 | exit 0 | `41bd8a36aab2` |
| Correct | 0 | material-point identity | exit 0 | `0dba5fb9601a` |
| Shuffled | 0 | shuffled identity control | exit 0 | `4305628bf2e0` |
| Dense Flow | 0 | RAFT target→initial latent warp | exit 0 | 已记录于 `flow_video_sha256.txt` |
| Correct-Blend | 0 | 0.5 Correct + 0.5 coarse，仅 mask 内 | exit 0 | 已记录于 `blend_video_sha256.txt` |
| Baseline 重复 | 0 | 无 | exit 0，逐字节等于首次 Baseline | `41bd8a36aab2` |
| Baseline seed1 | 1 | 无 | exit 0 | 已记录于 `baseline_seed1_sha256.txt` |
| Baseline EMA | 0 | 无，`--use_ema` | exit 0 | `12062a0ea6f0` |

各完整生成视频均为 21 帧、480×832、10 fps。

## Proxy 像素与时序指标

| 方法 | proxy PSNR↑ | proxy SSIM↑ | proxy 时序差分 L1↓ |
|---|---:|---:|---:|
| Baseline seed0 | 19.2271 | 0.76913 | 0.020981 |
| Correct | 18.9868 | 0.77861 | 0.021076 |
| Shuffled | 18.7701 | **0.79779** | 0.021556 |
| Dense Flow | 19.0136 | 0.77432 | 0.021103 |
| Correct-Blend α=0.5 | 19.1389 | 0.77304 | **0.020781** |
| Baseline seed1 | **19.8096** | 0.79617 | 0.021274 |
| Baseline EMA | 19.2830 | 0.82829 | 0.020141 |

这些数值中的“参考”是 coarse proxy，不是 GT，所以加粗仅表示该 proxy 指标的数值最优，不代表感知质量或方法最终优胜。

## RAFT 运动一致性

所有方法使用同一 torchvision RAFT-Large C_T_SKHT_V2，在 240×416 上与重建 coarse flow 比较。

| 方法 | EPE mean↓ | EPE median↓ | EPE p95↓ | cosine mean↑ |
|---|---:|---:|---:|---:|
| Baseline seed0 | 0.28005 | 0.05367 | **1.61631** | 0.33868 |
| Correct | 0.28279 | 0.04214 | 1.76988 | 0.41093 |
| Shuffled | 0.29010 | 0.03693 | 1.88610 | 0.44805 |
| Dense Flow | **0.27093** | 0.04575 | 1.64792 | 0.36306 |
| Correct-Blend α=0.5 | 0.28535 | 0.04517 | 1.75278 | 0.40588 |
| Baseline seed1 | 0.30979 | **0.02589** | 1.96106 | **0.50947** |

指标不同维度排序不一致，说明当前 proxy 上不存在单一方法全面占优。

## 关键审计结论

1. **transport 确实进入完整 generator。** Correct、Shuffled、Flow、Blend 的 stdout 分别记录了对应 payload 的加载、shape `[1,6,16,60,104]`、随后 6 个去噪 timestep 和视频保存；不是只做 VAE 或静态检查。
2. **持续材料点身份在完整模型中造成稳定、可测的输出变化。** 相对 Shuffled，Correct 的 proxy PSNR、时序差分 L1、RAFT mean EPE 和 p95 EPE更好；但 proxy SSIM、median EPE 和 cosine 更差，因此不能宣称全面胜出。
3. **Baseline 仍然强，Dense Flow 具有竞争力。** Baseline 的 RAFT p95 最好；Dense Flow 的 mean EPE 最好。当前没有证据证明 material-point transport 在完整 RealWonder 上稳定优于 Baseline 或 Flow。
4. **同 seed 路径确定性成立。** 两次 Baseline seed0 视频逐字节相同，SSIM=1、时序差=0。Baseline seed0→seed1 的组间差异（PSNR 20.983、SSIM 0.8424）明显大于 Baseline→Correct（PSNR 28.816、SSIM 0.9336），所以公平实验必须固定 seed。
5. **generator 与 EMA 不可混用。** 983/983 张量均有差异，参数 RMS 差约 `1.34e-5`；完整 EMA 视频也与默认 generator 不同（两者 PSNR 27.348、SSIM 0.8603）。
6. **第1轮方法迭代只有局部改善。** α=0.5 Blend 相对 full Correct 提高 proxy PSNR并降低时序差，但没有同步改善 SSIM 或 RAFT mean EPE。由于 proxy 非 GT，继续针对它调 α 会产生过拟合风险，因此没有盲目启动第2轮。

## 资源结论

- 完整任务单进程采样显存峰值最高 31,844 MiB，最低空闲 13,540 MiB。
- CPU `VmHWM` 约 46.1 GiB/任务。
- 两项完整模型并发时系统 `MemAvailable` 最低 23.188 GiB，虽未越过 20 GiB 安全线，但余量很窄；当前安全并发上限应为 2 个完整模型，不应仅因四卡空闲就启动 4 个完整模型。
- 无 CUDA OOM、Xid、ECC 或我方危险温度事件；未操作其他用户进程。

## 代码与回归

- `wan/modules/attention.py`：当 FlashAttention 不可用时，直接调用点回退到 PyTorch SDPA；补齐 q/k 长度、因果与局部窗口 mask。GPU 数值回归中 plain 与 `k_lens` 最大差均为 0。
- `infer_sim.py` 与 `deform_transport/pipeline_integration.py`：增加严格 shape/dtype/finite/mask 契约下的 Correct、Shuffled、Flow、Blend 注入。
- 修改后 unittest 25/25 通过，`pip check` 通过，compileall 与 `git diff --check` 通过。

## 视觉材料

- 六路对照视频：`methods_3x2.mp4`
- 六路 5 时刻联系表：`methods_contact_sheet.jpg`
- 原始三路/四路指标 JSON 与逐帧 CSV 均位于本目录。

当前自动图像查看器因宿主 CentOS 7 不支持 bwrap 用户命名空间而无法打开联系表，因此没有伪造“人工视觉已通过”的结论；视觉材料已完整生成，仍需人工审阅。

## 仍未完成与不可越界结论

- 未获得官方原始 Santa `final_sim` 或 future GT；当前完整结果仍是工程 proxy 证据。
- 其他官方案例与机器人可变形物体案例没有 contract-complete simulation 输出，不能公平运行完整 Baseline/Correct/Shuffled。
- 尚未证明跨对象泛化、真实机器人收益、统计显著性、LPIPS/FVD 改善或 material-point transport 对 Baseline/Flow 的稳定优势。
