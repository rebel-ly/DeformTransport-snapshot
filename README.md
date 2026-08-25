# DeformTransport

DeformTransport 是一个面向可变形物体视频生成的物理引导方法。整体流程为：

```text
物理仿真
→ 物理状态对齐
→ 结构预览条件 + 材料轨迹条件
→ 材料身份感知的潜特征传输
→ 视频扩散生成
→ 输出视频
```

## 1. 论文模块与代码对应关系

| 论文模块             | 主要实现文件 / 函数                                                                                                                               | 功能                                                   | 输入                                                                                 | 输出                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **物理仿真与物理状态获取**  | `DeformTransport/scripts/run_realwonder_trajectory_probe.py`；`InteractiveSimulator`                                                       | 根据场景与外部动作推进布料物理仿真，记录具有持续材料身份的材料点时序状态                 | 初始场景、动作/外力、仿真参数                                                                    | `point_trajectories.pt`、`flows.npy`、`flow_source_point_indices.npy`、粗粒度渲染帧                                 |
| **物理状态对齐与可见性构建** | `DeformTransport/scripts/export_transport_ready.py`；`build_transport_ready()`；`build_aligned_transport_visibility_contract.py`            | 将材料点状态统一到正式 81 帧时间轴、图像坐标和潜空间坐标，并建立物理可见性约束            | `point_trajectories.pt`、raster point IDs、首帧及仿真帧                                    | `aligned_transport_ready.pt`、`aligned_visibility_contract.pt`                                              |
| **材料轨迹条件构建**     | `DeformTransport/scripts/export_santa_material_tracks_to_wan_move_visibility_corrected.py`                                                | 将物理材料点状态转换为视频模型可使用的材料轨迹，并在首帧每个被占据的 VAE 空间单元选择一个持续材料点 | 对齐后的二维材料点轨迹、可见性、材料点索引                                                              | `santa_material_tracks_correct.npy`、`santa_material_visibility_correct.npy`、`santa_material_point_ids.npy` |
| **深度条件构建**       | `DeformTransport/scripts/run_phase0b4_functional_conditioning_audit.py::main()`                                                           | 根据正式选择的材料点索引，从对齐物理状态中提取对应深度，用于潜空间冲突仲裁                | `aligned_transport_ready.pt` 中的 `depth[81,28264]` + `santa_material_point_ids.npy` | `santa_authoritative_depth_81x1257.npy`                                                                    |
| **结构预览条件构建**     | C2 preview pipeline；`Wan-Move/wan/wan_move.py::WanMove.generate()`                                                                        | 将物理仿真的粗粒度结构预览编码为视频潜表示，并通过 SDEdit 方式初始化后续扩散采样         | 结构预览帧、Wan VAE、共享随机噪声                                                               | `WAN_FORMAL_PREVIEW_LATENT_58x104.npy`、`R3_SHARED_EPSILON_58x104.npy`、`start_index=15`                     |
| **材料身份感知的潜特征传输** | `Wan-Move/wan/modules/trajectory.py::create_pos_feature_map()`、`_dt_load_sidecars()`、`_dt_bilinear_source_features()`、`replace_feature()` | 将首帧源潜特征与持续材料身份绑定，并沿未来材料轨迹传播；利用可见性筛选和深度仲裁处理遮挡及冲突      | 首帧 VAE 特征、tracks、visibility、material IDs、depth                                     | `edited_y`                                                                                                 |
| **视频生成**         | `Wan-Move/generate.py`；`Wan-Move/wan/wan_move.py::WanMove.generate()`；`WanModel.forward()`                                                | 将首帧图像、结构预览条件和材料轨迹条件共同用于扩散去噪，并通过 VAE 解码得到未来视频         | 图像、Prompt、`edited_y`、preview latent、noise                                          | 81 帧、480×832 MP4 视频                                                                                        |

## 2. 模块数据流

```text
场景 + 外部动作
        ↓
RealWonder / Genesis 物理仿真
        ↓
point_trajectories.pt
        │
        ├──────────────────────────────┐
        ↓                              ↓
物理状态与可见性对齐                  粗粒度结构预览
        ↓                              ↓
aligned_transport_ready.pt        Wan VAE Encode
aligned_visibility_contract.pt         ↓
        ↓                       preview_latent
持续材料点选择                         +
        ↓                        shared epsilon
tracks.npy                             │
visibility.npy                         │
material_ids.npy                       │
        ↓                              │
按相同 material IDs 提取 depth         │
        ↓                              │
depth.npy                              │
        │                              │
        └──────────────┬───────────────┘
                       ↓
             DeformTransport C2
                       ↓
        材料身份感知的潜特征传输
                       ↓
                  edited_y
                       ↓
                WanModel.forward
                       ↓
                Diffusion Sampling
                       ↓
                  Wan VAE Decode
                       ↓
                    MP4
```

## 3. 论文布料场景正式输入

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

正式材料轨迹规模：

```text
tracks       : [1, 81, 1257, 2]
visibility   : [1, 81, 1257]
material IDs : [1257]
depth        : [81, 1257]
```

## 4. 完整论文方法运行：C2

C2 为论文完整方法配置，同时启用：

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

### 进入 Docker

```bash
docker exec -it \
  --user 10011:10011 \
  --workdir /workspace \
  -e HOME=/workspace \
  deformtransport-dev \
  bash
```

### 查看 GPU

```bash
nvidia-smi
```

### 创建演示输出目录

```bash
mkdir -p /workspace/DeformTransport_demo_output
```

### 运行完整方法

例如使用 GPU 0：

```bash
bash \
/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/run_c2.sh \
/workspace/DeformTransport_demo_output \
0
```

参数含义：

```text
参数 1：输出目录
参数 2：GPU 编号
```

输出视频：

```text
/workspace/DeformTransport_demo_output/c2_provisional_correct_v3d_seed000.mp4
```

查看结果：

```bash
ls -lh /workspace/DeformTransport_demo_output/
```

### C2 实际使用的主要输入

`run_c2.sh` 内部调用冻结的：

```text
/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay/generate.py
```

并输入：

```text
首帧图像
├── resized_input_image.png

材料轨迹条件
├── santa_material_tracks_correct.npy
├── santa_material_visibility_correct.npy
├── santa_material_point_ids.npy
└── santa_authoritative_depth_81x1257.npy

结构预览条件
├── WAN_FORMAL_PREVIEW_LATENT_58x104.npy
├── R3_SHARED_EPSILON_58x104.npy
└── start_index = 15
```

生成配置：

```text
DT_TRANSPORT_VARIANT = v3d
frame_num            = 81
resolution           = 480×832
seed                 = 0
sample_steps         = 40
sample_shift         = 3.0
dtype                = bf16
```

## 5. 材料轨迹核心模块运行：DT-FULL

如果只需演示论文中**材料身份感知的潜特征传输**，可使用冻结的 DT-FULL runner。

```text
/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/scripts/run_dtfull_container_exact.sh
```

运行：

```bash
mkdir -p /workspace/DeformTransport_demo_dtfull

bash \
/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/scripts/run_dtfull_container_exact.sh \
0 \
/workspace/DeformTransport_demo_dtfull
```

参数含义：

```text
参数 1：随机种子
参数 2：输出目录
```

输出：

```text
/workspace/DeformTransport_demo_dtfull/santa_correct_v3d_seed000.mp4
```

DT-FULL 与完整 C2 使用相同的：

```text
首帧图像
材料轨迹
材料可见性
持续材料身份
材料深度
V3D 潜特征传输
```

区别是 DT-FULL **不启用结构预览-SDEdit**，主要用于单独验证和展示材料轨迹传输模块。

## 6. 核心生成调用链

```text
generate.py
    ↓
WanMove.generate()
    ↓
create_pos_feature_map()
    ↓
_dt_load_sidecars()
    ↓
_dt_bilinear_source_features()
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
