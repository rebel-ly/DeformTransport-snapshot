# Phase 0D-4A matched VBench screen — gated unresolved

## 1. 阶段目标

对已有 Santa 视频执行同一官方 VBench 的 matched-case imaging/aesthetic comparison。

## 2. 审计问题

现有 Wan-Move/DeformTransport 视频是否在同一 Santa case 下有相对 RW 的视觉质量优势。

## 3. 使用的数据

冻结的 RW0、DT-FULL seed0--4、FRAG0、GRID1000、WM00，共九条候选路径已定位。

## 4. 使用的方法

使用 OpenCV 确定性解码对每条视频检查帧数、尺寸、fps、codec metadata 和 SHA256；检索官方 VBench 本地实现与权重。未使用非官方替代分数。

## 5. 关键命令/脚本

本次报告目录下的隔离 `isolated_vbench/` 是官方仓库 clone 尝试；该 checkout 没有有效 `HEAD`，故不可作为 VBench 实现运行。

## 6. 关键结果

DT/FRAG/GRID/WM0 均为 81 帧、832×464、16 fps。RW0 为 81 帧、**832×480、10 fps**（SHA `1a70268e9a872f24c4945b6bae41ef82b58fe2248b948cbe8e06b516e781cfe7`）。这违反 A 的必要 matched spatial/frame contract。官方 VBench 本地不存在；隔离 clone 未形成有效 checkout；未发现本地官方权重。

## 7. PASS/FAIL/UNRESOLVED 判断

`A_VIDEO_COUNT=9`，但 `A_ALL_CORRECTED_FORMAL_LINEAGE=FAIL` for matched A because RW0 decoded geometry/fps differs. `CODEC_STANDARDIZATION=FAIL`：没有可用 ffmpeg CLI，且把 480 高度 RW 直接变为 464 会是未预注册的空间变换，不能静默实施。VBench scores 未计算。`RW_PAPER_CONSISTENCY_EXACT_DIMENSION=UNRESOLVED`。

## 8. 对后续实验影响

不得据此给出 DT/RW visual-quality ranking。须先恢复一个与 832×464 Wan-Move evaluation domain 匹配、可审计的 RW Santa output，或由用户明确授权固定的共同 geometry transform；随后再在可用官方 VBench 环境中执行所有九条标准化输入。

## 9. 遗留问题

官方 VBench checkout/权重与共同 codec pipeline 均不可用；未进行重新编码或视觉打分。WM0 仅是 `UNCONDITIONED_TRAJECTORY_CONTROL_DIAGNOSTIC`。
