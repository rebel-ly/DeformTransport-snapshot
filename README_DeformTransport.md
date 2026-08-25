# DeformTransport

**面向可变形物体的材料身份感知物理条件视频生成方法**

DeformTransport 的整体流程如下：

```text
首帧图像 + 外部动作
        ↓
物理仿真与物理状态获取
        ↓
物理状态对齐与可见性构建
        ↓
结构预览条件 + 材料轨迹条件
        ↓
材料身份感知的潜特征传输
        ↓
视频扩散生成
        ↓
输出视频
```

---

## 1. 论文模块与代码对应关系

> 本节只给出论文模块、主要实现位置和作用。具体算法细节以对应源码中的注释为准。

| 论文模块 | 主要实现文件 / 函数 | 作用 |
|---|---|---|
| **物理仿真与物理状态获取** | `DeformTransport/scripts/run_realwonder_trajectory_probe.py`；`InteractiveSimulator` | 根据场景与外部动作推进布料物理仿真，记录具有**持续材料身份**的材料点时序状态 |
| **物理状态对齐与可见性构建** | `DeformTransport/scripts/export_transport_ready.py`；`build_transport_ready()`；`build_aligned_transport_visibility_contract.py` | 将材料点状态统一到正式 81 帧时间轴、图像坐标和潜空间坐标，并建立物理可见性约束 |
| **材料轨迹条件构建** | `DeformTransport/scripts/export_santa_material_tracks_to_wan_move_visibility_corrected.py` | 将物理材料点状态转换为视频生成模型使用的**材料轨迹条件**，并筛选持续材料点 |
| **深度条件构建** | `DeformTransport/scripts/run_phase0b4_functional_conditioning_audit.py::main()` | 根据已选材料点索引，从对齐后的物理状态中提取相应深度，用于潜空间冲突仲裁 |
| **结构预览条件构建** | C2 preview pipeline；`Wan-Move/wan/wan_move.py::WanMove.generate()` | 将物理仿真的粗粒度未来结构预览编码到潜空间，并通过 **SDEdit** 初始化扩散采样 |
| **材料身份感知的潜特征传输** | `Wan-Move/wan/modules/trajectory.py::create_pos_feature_map()`；`_dt_load_sidecars()`；`_dt_bilinear_source_features()`；`replace_feature()` | 将首帧源潜特征与**持续材料身份**绑定，并沿未来材料轨迹传播；结合可见性筛选与深度仲裁 |
| **视频生成** | `Wan-Move/generate.py`；`Wan-Move/wan/wan_move.py::WanMove.generate()`；`WanModel.forward()` | 将结构预览条件和材料轨迹条件送入视频扩散模型，完成多步去噪并解码得到未来视频 |

---

## 2. 各模块输入、输出与调用关系

### 2.1 物理仿真与物理状态获取

**主要入口**

```text
DeformTransport/scripts/run_realwonder_trajectory_probe.py
```

**主要调用**

```text
InteractiveSimulator
→ physics step
→ render / projection
→ save point trajectories
```

**输入**

- 初始场景
- 外部动作 / 外力
- 仿真参数

**输出**

```text
point_trajectories.pt
flows.npy
flow_source_point_indices.npy
frame_initial.png
frame_*.png
```

其中 `point_trajectories.pt` 保存后续条件构建所需的材料点时序状态。

**下一步**

```text
point_trajectories.pt
        ↓
物理状态对齐与可见性构建
```

---

### 2.2 物理状态对齐与可见性构建

**主要实现**

```text
DeformTransport/scripts/export_transport_ready.py
DeformTransport/scripts/build_aligned_transport_visibility_contract.py
```

`export_transport_ready.py` 主要调用：

```text
build_transport_ready()
save_transport_ready()
```

**输入**

```text
point_trajectories.pt
flow_source_point_indices.npy
首帧图像
粗粒度仿真帧
```

**输出**

```text
transport_ready.pt
aligned_transport_ready.pt
aligned_visibility_contract.pt
```

该阶段负责统一：

- 81 帧时间轴；
- 图像坐标；
- 视频潜空间坐标；
- 物理可见性。

**下一步**

```text
aligned_transport_ready.pt
aligned_visibility_contract.pt
        ↓
材料轨迹条件构建 / 深度条件构建
```

---

### 2.3 材料轨迹条件构建

**主要实现**

```text
DeformTransport/scripts/export_santa_material_tracks_to_wan_move_visibility_corrected.py
```

**功能**

把 RealWonder / 物理仿真得到的材料点状态转换为视频生成模型可直接使用的**材料轨迹条件**。

主要过程：

```text
物理材料点轨迹
→ 坐标变换到 480×832 视频坐标
→ 物理可见性筛选
→ 首帧每个被占据的 8×8 VAE 空间单元选择一个持续材料点
→ 输出正式材料轨迹条件
```

**输入来自**

- `aligned_transport_ready.pt`：对齐后的材料点位置等物理状态；
- `aligned_visibility_contract.pt`：物理可见性；
- 持续材料点索引：由物理仿真中的材料点身份继承。

**输出**

```text
santa_material_tracks_correct.npy
santa_material_visibility_correct.npy
santa_material_point_ids.npy
```

正式布料场景的数据规模为：

```text
tracks       : [1, 81, 1257, 2]
visibility   : [1, 81, 1257]
material IDs : [1257]
```

**下一步**

```text
tracks.npy
visibility.npy
material_ids.npy
        ↓
Wan-Move V3D 材料身份感知潜特征传输
```

---

### 2.4 深度条件构建

**主要实现**

```text
DeformTransport/scripts/run_phase0b4_functional_conditioning_audit.py::main()
```

**输入来自**

```text
aligned_transport_ready.pt
    └── depth[81, 28264]

santa_material_point_ids.npy
    └── 已选择的 1257 个持续材料点
```

**主要处理**

```python
depth_selected = depth_all[:, selected_ids]
```

即按照正式材料点索引，从全部对齐材料点深度中提取对应深度。

**输出**

```text
santa_authoritative_depth_81x1257.npy
```

数据规模：

```text
depth : [81, 1257]
dtype : float32
```

**下一步**

```text
depth.npy
        ↓
replace_feature(v3d)
        ↓
多个材料点竞争同一目标潜单元时进行深度仲裁
```

---

### 2.5 结构预览条件构建

论文完整方法 C2 同时使用**结构预览条件**和**材料轨迹条件**。

结构预览路径：

```text
物理仿真未来状态
        ↓
粗粒度结构预览
        ↓
Wan VAE Encode
        ↓
preview latent
        ↓
SDEdit 初始化
```

正式 C2 使用：

```text
WAN_FORMAL_PREVIEW_LATENT_58x104.npy
R3_SHARED_EPSILON_58x104.npy
start_index = 15
```

其中：

```text
preview latent   : [16, 21, 58, 104], float32
initial epsilon  : [16, 21, 58, 104], float32
```

这部分最终在：

```text
Wan-Move/wan/wan_move.py::WanMove.generate()
```

中进入扩散采样。

---

### 2.6 材料身份感知的潜特征传输

**核心文件**

```text
Wan-Move/wan/modules/trajectory.py
```

**主要函数**

```text
create_pos_feature_map()
        ↓
_dt_load_sidecars()
        ↓
_dt_bilinear_source_features()
        ↓
replace_feature()
```

#### `create_pos_feature_map()`

将二维材料轨迹映射到视频潜空间，并保存：

```text
tracks
visibility
depth
material IDs
```

#### `_dt_load_sidecars()`

读取：

```text
DT_TRACK_IDS_PATH
DT_TRACK_DEPTH_PATH
```

#### `_dt_bilinear_source_features()`

根据材料点在首帧中的连续坐标，对首帧 VAE 条件特征进行双线性采样，实现：

```text
持续材料身份
    ↕
首帧源潜特征
```

#### `replace_feature()`

正式方法使用：

```text
DT_TRANSPORT_VARIANT=v3d
```

执行：

```text
首帧源特征采样
→ 与持续材料身份绑定
→ 沿材料点未来轨迹传播
→ 可见性筛选
→ 深度冲突仲裁
→ 写入未来潜空间位置
```

输出：

```text
edited_y
```

**下一步**

```text
edited_y
        ↓
Wan-Move conditional branch
        ↓
WanModel.forward()
```

---

### 2.7 视频生成

**正式入口**

```text
Wan-Move/generate.py
```

完整调用链：

```text
generate.py
    ↓
WanMove.generate()
    ↓
create_pos_feature_map()
    ↓
replace_feature(v3d)
    ↓
edited_y
    ↓
y_cond
    ↓
WanModel.forward()
    ↓
Diffusion Sampling
    ↓
Wan VAE Decode
    ↓
output.mp4
```

最终输出：

```text
81 frames
480 × 832
MP4 video
```

---

## 3. 论文布料场景正式输入

论文正式演示使用 **Santa Cloth** 布料场景。

<details>
<summary><b>点击展开正式输入路径</b></summary>

### 首帧图像

```text
/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png
```

### 材料轨迹

```text
/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy
```

### 材料可见性

```text
/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy
```

### 持续材料身份

```text
/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy
```

### 材料深度

```text
/workspace/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy
```

</details>

---

## 4. 完整论文方法运行：C2

C2 是论文完整方法配置，同时启用：

```text
结构预览-SDEdit
+
正确材料轨迹传输
+
持续材料身份
+
可见性筛选
+
深度仲裁
```

冻结运行脚本：

```text
/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/run_c2.sh
```

### 4.1 进入 Docker

```bash
docker exec -it \
  --user 10011:10011 \
  --workdir /workspace \
  -e HOME=/workspace \
  deformtransport-dev \
  bash
```

### 4.2 查看 GPU

```bash
nvidia-smi
```

### 4.3 创建输出目录

```bash
mkdir -p /workspace/DeformTransport_demo_output
```

### 4.4 运行完整方法

例如使用 GPU 0：

```bash
bash \
/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/run_c2.sh \
/workspace/DeformTransport_demo_output \
0
```

参数：

```text
参数 1：输出目录
参数 2：GPU 编号
```

输出视频：

```text
/workspace/DeformTransport_demo_output/c2_provisional_correct_v3d_seed000.mp4
```

---

## 5. C2 内部实际调用

`run_c2.sh` 主要完成：

```text
设置 CUDA_VISIBLE_DEVICES
设置 DT_TRANSPORT_VARIANT=v3d
设置 DT_TRACK_IDS_PATH
设置 DT_TRACK_DEPTH_PATH
        ↓
调用 frozen overlay generate.py
        ↓
加载首帧图像
加载 tracks / visibility
加载 material IDs / depth
加载 preview latent / initial epsilon
设置 start_index=15
        ↓
WanMove.generate()
        ↓
DeformTransport V3D
        ↓
视频扩散生成
```

实际 Python 入口：

```text
/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay/generate.py
```

正式生成配置：

```text
DT_TRANSPORT_VARIANT = v3d
frame_num            = 81
resolution           = 480×832
seed                 = 0
sample_steps         = 40
sample_shift         = 3.0
dtype                = bf16
start_index           = 15
```

---

## 6. 材料轨迹核心模块运行：DT-FULL

如果只需单独演示论文中的**材料身份感知潜特征传输**，可运行冻结的 DT-FULL：

```text
/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/scripts/run_dtfull_container_exact.sh
```

```bash
mkdir -p /workspace/DeformTransport_demo_dtfull

bash \
/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/scripts/run_dtfull_container_exact.sh \
0 \
/workspace/DeformTransport_demo_dtfull
```

参数：

```text
参数 1：随机种子
参数 2：输出目录
```

输出：

```text
/workspace/DeformTransport_demo_dtfull/santa_correct_v3d_seed000.mp4
```

DT-FULL 使用与 C2 相同的：

```text
首帧图像
材料轨迹
材料可见性
持续材料身份
材料深度
V3D 潜特征传输
```

区别是：

```text
C2      = 结构预览-SDEdit + 材料轨迹传输
DT-FULL = 仅材料轨迹传输
```

因此：

- **C2**：用于演示论文完整方法；
- **DT-FULL**：用于单独演示材料身份感知潜特征传输模块。
