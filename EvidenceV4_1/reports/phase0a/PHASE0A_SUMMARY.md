# DeformTransport Evidence V4.1 — Phase0A Summary

## 1. 阶段目标

Phase0A 的目标是确认后续正式实验所使用的 Santa material-track
资产是否具有可追溯、时间一致、坐标一致的 geometry/visibility contract，
并进一步判断保存的 visibility 是否可以被解释为稳定的视觉可见性语义。

本阶段不评价最终生成视频质量，而是验证进入后续因果实验之前的数据与协议基础。

---

## 2. 审计问题

本阶段依次回答：

1. material point、trajectory、raster visibility、aligned visibility
   是否来自可追溯的同一资产链；
2. source state 与 future trajectory 的时间索引是否正确；
3. corrected bridge 是否在 geometry 与 visibility 上使用同一时间轴；
4. carrier 是否按照 true source Wan-VAE 8×8 cell 选择；
5. bridge visibility 是否能逐元素追溯到 authoritative raster contract；
6. raster-defined visibility 是否能够进一步解释为稳定 perceptual visibility。

---

## 3. 使用的数据

### Raw simulation chain

`OFFICIAL_SANTA_81F_CHAIN_20260805_050719`

原始时间契约：

- source / `frame_initial.png` = simulation step 0；
- old future `frame_0000..0080` = steps 10..810。

总 material point 数：28264。

### Transport-ready asset

`official_santa_81f_transport_ready_continuous_probe_20260805_182205`

source-visible point 数：19126。

### Canonical aligned contract

`official_santa_81f_aligned_contract_20260806_192643`

正式时间轴：

- frame 0 = step0 / true source；
- frame 1 = step10；
- ...
- frame 80 = step800。

builder 内部 27/27 checks PASS。

### Rejected bridge

`20260811_024330__santa_corrected_physical_visibility`

N=1277。

### Formal corrected-v2 bridge

`20260811_224005__santa_corrected_v2_aligned_timeline`

N=1257。

---

## 4. 使用的方法

### 4.1 Asset/shape/hash audit

检查：

- tensor shape；
- material point ID 唯一性；
- source file SHA256；
- frame IDs；
- simulation steps；
- coordinate-system metadata。

### 4.2 Boolean contract reconstruction

从 `flow_source_point_indices.npy` 每帧提取 unique non-negative
frontmost material IDs，并与 projection-valid mask 相交，重新构造
`aligned_visible`。

随后验证：

`source_and_aligned_visible = source_visible AND aligned_visible`

并验证 corrected bridge visibility 是否等于 contract 中 selected material IDs
对应的 visibility。

### 4.3 Temporal alignment audit

比较：

- source coordinates；
- old future frame0；
- aligned frame0；
- corrected bridge frame0；
- full 81-frame timeline。

对 visible slots 做 exact-equality test，并测试 +1-frame shift 假设。

### 4.4 True-source VAE-cell audit

在 true source step0 的 832×480 video domain 中，以 8×8 VAE cell 为单位，
按照“距离 cell center 最近，再以最低 material point ID tie-break”的规则，
重新选择 persistent material carriers。

### 4.5 Semantic visibility audit

在 corrected-v2 上、在看图之前冻结 7 个 temporal/geometric cases：

- stable visible；
- persistent loss；
- long loss + reappearance；
- rapid switching；
- moderate switching；
- source silhouette boundary；
- boundary + switching。

在 512×512 simulator/render native space 中叠加：

- projected material point；
- frontmost raster support；
- contract visibility；
- local zoom；
- projected location front-ID/depth diagnostic。

最终人工进行 PASS / FAIL / UNRESOLVED 裁决。

---

## 5. 关键脚本/命令

关键实现包括：

- `deform_transport/transport_ready.py`
- `scripts/build_aligned_transport_visibility_contract.py`
- `scripts/export_santa_material_tracks_to_wan_move_visibility_corrected.py`
- Evidence V4.1 corrected-v2 builder
- Phase0A semantic overlay renderer

Phase0A 的关键特点是优先进行 exact reconstruction / exact equality，
而不是只比较统计比例。

---

## 6. 关键结果

### 6.1 Visibility contract lineage

`source_and_aligned_visible`：

- 总 observation：2,289,384；
- mismatch：0；
- exact equality：PASS。

Raster → `aligned_visible`：

- mismatch：0 / 2,289,384；
- exact reconstruction：PASS。

Old corrected bridge subset：

- 103,437 observations；
- bridge vs contract mismatch：0。

因此 visibility Boolean 数据本身拥有严格 lineage。

### 6.2 Source visibility

28264 material points 中：

- source-visible：19126；
- source-visible ratio ≈ 0.67669。

`source_visible_point_ids` 与
`source_raster_visible_point_ids` exact equal。

本 Santa 资产中 `source_valid` 全部为 True。

因此实际 source visibility 等价于：
source raster 中的 frontmost material-ID membership。

### 6.3 发现旧 corrected bridge 的 +1-frame 时间错位

旧 bridge：

- geometry：steps10..810；
- visibility：steps0..800。

测试：

`PLUS_ONE_SHIFT_VISIBLE_EXACT_EQUAL_t0_79 = True`

且：

`CURRENT_T80_EQ_OLD_FUTURE80_ON_VISIBLE = True`

因此 +1-frame 错位被逐元素证明。

正确 timeline 下 visible coordinate error：

- mean L2 ≈ 3.279 px；
- P95 ≈ 7.817 px；
- max ≈ 10.589 px。

source → old frame0：

- 1215/1277 selected points 非零移动；
- 451/1277（35.317%）改变 Wan-VAE cell。

旧 bridge 因此被正式拒绝。

### 6.4 True-source carrier selection

真正 step0 source：

- source-visible candidates：19126；
- occupied true-source VAE cells：1257；
- 正确 selected carriers：1257。

错误旧 bridge：

- selected：1277；
- 映射回 true source 后仅 1144 个 unique cells；
- missing source cells：113；
- duplicate excess：133；
- 与正确 selected ID set 的 Jaccard ≈ 0.0733。

### 6.5 Corrected-v2

formal corrected-v2：

- N=1257；
- one point per true source cell：PASS；
- frame0 exact source：PASS；
- visible raw coordinates exact authoritative geometry：PASS；
- visible positions unchanged by filling：PASS；
- global visibility fraction ≈ 0.4049422；
- frame0 visible=1257；
- frame40 visible=479；
- frame80 visible=328。

### 6.6 Semantic audit

7 frozen cases：

- PASS：4；
- FAIL：2；
- UNRESOLVED：1。

PASS：
- A stable visible；
- B persistent loss；
- C long loss + reappearance；
- F source silhouette boundary。

FAIL：
- D rapid switching；
- G boundary switching。

UNRESOLVED：
- E moderate switching。

主要发现：

在高 switching / silhouette boundary 区域，
Boolean visibility 的变化更符合 material-point frontmost raster ownership
切换，而不一定对应稳定的 perceptual visibility 变化。

---

## 7. PASS / FAIL / UNRESOLVED

### PASS

- asset lineage；
- source/raster visibility lineage；
- exact raster reconstruction；
- canonical step0..800 aligned contract；
- corrected-v2 carrier sampling；
- corrected-v2 geometry/visibility temporal alignment；
- operational raster/frontmost-support visibility definition。

### FAIL

- old 1277-point corrected bridge temporal contract；
- old bridge true-source-cell sampling contract；
- 将 per-material frontmost-support visibility 解释为
  “稳定 perceptual visibility ground truth”的强语义主张。

### UNRESOLVED

- simulator visibility 是否可以直接代表真实 RGB / real-world
  visibility semantics。

---

## 8. 对后续实验的影响

Phase0B 及之后：

1. 只允许使用 corrected-v2 1257-point bridge；
2. 禁止重新使用 rejected 1277-point bridge；
3. geometry timeline 必须是 step0..800；
4. visibility 应称为：
   `raster-defined visibility` /
   `frontmost-support visibility` /
   `operational visibility contract`；
5. 不应称为 stable perceptual visibility ground truth；
6. high-switch / boundary points 后续需要单独分析其对 causal comparison
   的影响；
7. Phase0B 必须证明 Correct vs Identity-Shuffled 唯一改变的是指定的
   identity/correspondence variable。

---

## 9. 遗留问题

Phase0A 本身不再新增实验。

仍保留两个跨阶段问题：

1. repository 内是否还有其它 downstream scripts 绕过
   `aligned_transport_ready.pt`，直接重新读取 raw
   `point_trajectories.pt`；
2. raster/frontmost visibility flicker 是否会对后续
   Correct vs Shuffled 差异产生可测影响。

第一个问题进入 repository-wide source audit；
第二个问题将在 Phase0B/后续 mechanism audit 中处理。

---

## Phase0A 最终结论

Phase0A 科学分析已结束。

**工程/数据契约层：PASS。**

经过审计和修复后，corrected-v2 提供了可追溯、
true-source aligned、geometry/visibility 同轴的正式 material-track bridge。

**稳定 perceptual visibility 强语义层：FAIL。**

当前 Boolean visibility 应被解释为精确的
frontmost-raster support contract，而不是稳定的人类感知可见性真值。

Phase0A 状态：

`CLOSED_WITH_LIMITATION`

允许进入：

`Phase0B — Correct vs Identity-Shuffled Causal Contract Audit`
