# Phase 0D-2F / F2-R Summary

## 1. 阶段目标

在任何新的 GPU/diffusion generation 前，审计 DT-FULL 正子组、RW coarse 支持孔洞、condition TC-ME 表示对称性和历史 Tree 时间线。

## 2. 审计问题

R1 检查 zero-switch 是否被 zero-visible 混杂；R2 检查 coarse TC-MAR 是否主要由 raster holes 造成；R3 检查 RW RGB 与 DT edited_y 是否可作对称 conditioning video；R4 检查 Tree 是否存在 Santa 式固定时间偏移。

## 3. 使用的数据

使用冻结 F2 corrected-v2 per-sample arrays、N=1257 material-ID join、canonical Santa RW 81-frame coarse simulation 及其 `flow_source_point_indices[0:81]`、F2 VAE-only edited_y reconstruction、Tree 的 raw trajectories/aligned transport/visibility/bridge sidecars 和历史 Tree motion report。

## 4. 使用的方法

R1 仅合并冻结分组。R2 以权威 point-ID occupancy（ID>=0）作为支持，不以 RGB 黑色判断；空间使用原有 512→832、y=176:656 几何，categorical support 以 nearest lookup 映射至 frozen 8x8 patch sample locations。R3 只审计表示。R4 在可见支持上对 selected bridge tracks 与 raw/aligned states 做 exact equality，并检验 offsets -2..+2；不可见帧的 exporter forward-fill 坐标不被当作几何状态。

## 5. 关键命令/脚本

`scripts/audit_r1_and_frag.py`、`scripts/audit_r2_rw_support.py`、`scripts/audit_r3_r4.py`、`scripts/audit_r4_visible_state.py`。所有执行为本地 deterministic Python/既有 helper/artifact inspection；未运行 transformer、diffusion 或视频生成。

## 6. 关键结果

R1: zero-switch 总数 62，zero-visible 为 0，positive-visible 为 62（全为 always-visible）；其 DT-RW MAR gap = -9.3879105103，ME gap = +0.0844265373，故正子组仍存在。Q4 stable N=13、MAR/ME gap = +15.5764840216/+0.5912901352；Q4 fragmented N=301、= +8.7771372835/+0.3102430976，分类为 high motion remains bad when stable。

R2: future canvas occupancy = 0.1576653270；9,766/9,766 sampled instances 满足 ANY，8,436/9,766（0.8638132296）满足 FULL。ALL/ANY/FULL carrier Lab means 分别为 65.7134452009、65.7134452009、66.2863770393，因此 holes 不能解释该高值。保留一个限制：F2 condition script 的 RGB 在 `to_common` 前后各除以 255 一次，故绝对 condition TC-MAR 数值受限，但支持子集比较严格复现冻结 ALL。

R3: RW 是 81 个显式 rerasterized RGB states；DT 是 `16x21x60x104` latent，其中 9,031/124,800 future cells（7.236378%）被写入、92.763622% 未触及，随后 VAE 解码为 RGB。二者 support 与 temporal semantics 不可比，condition TC-ME 不能用于因果“RW intrinsically better”结论。

R4: Tree bridge 在可见 support 上与 raw initial+future 和 aligned transport 完全一致：36,143/36,143 exact，最大残差 0；最佳 offset 0。±1 的 mean residual 为 1.9094756483 px，±2 为 3.8701688528 px。Tree timeline PASS，但历史数值仍仅 legacy/directional。

## 7. PASS/FAIL/UNRESOLVED 判断

F2-R 为 `PASS_WITH_LIMITATIONS`。R1/R3/R4 PASS；R2 完成且 holes 解释为 `NOT_SUPPORTED`，但 F2 condition absolute TC-MAR 的 double-normalization 使 appearance-alignment hypothesis 为 `UNRESOLVED`。未改变任何 formal evaluator、metric、track、visibility 或 evaluation support。

## 8. 对后续实验影响

仅依据 R1，预注册首轮候选保持 `WM-0`、`DT-FRAG-PRUNE`、`DT-GRID100-CENTER`。FRAG-PRUNE 冻结规则为 retain `switch_count < 3`，不额外移除 zero-future-visible。后续所有 arm 必须在 full N=1257 corrected-v2 support 及冻结子组上报告。

## 9. 遗留问题

R1 是关联性分层，不证明机制或因果；Q4 stable 仅 N=13。R2 的绝对 condition Lab 值需要在未来单独、明确版本化的诊断修复后才可重新解释，不能替换冻结 F2 数值。Tree timeline 兼容性不自动将历史 Tree scores 提升为 corrected-v2 formal evidence。
