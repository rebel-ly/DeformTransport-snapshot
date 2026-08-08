# DeformTransport 项目地图

## 1. 当前主线

当前开发案例：

- Santa cloth
- 81像素帧
- 21个Wan VAE latent时刻
- 风场物理作用
- 无真实未来视频GT
- 仅作为开发和机制验证案例

当前目标：

1. 保持RealWonder Baseline不变；
2. 将Hard Point Transport升级为Soft Point Transport；
3. 优先解决nearest-cell、遮挡冲突和硬替换造成的模糊；
4. 方法稳定后接入RealWonder官方多案例；
5. 最后完成正式对比和消融。

---

## 2. 最重要的代码入口

### infer_sim.py

完整RealWonder视频生成入口。

作用：

- 加载final_sim；
- 编码粗模拟RGB；
- 加载transport artifact；
- 选择correct、shuffled、flow或blend；
- 调用RealWonder generator；
- 输出未来视频。

当前限制：

transport在完整推理前替换sim_latent，尚未实现去噪过程中动态注入。

### deform_transport/hard_transport.py

当前Hard Point Transport实现。

包括：

- Correct材料点身份运输；
- Shuffled身份消融；
- nearest-cell source采样；
- nearest-cell target forward splat；
- 多点冲突平均；
- transport mask与count生成。

这是固定旧方法和消融，不应直接覆盖。

### deform_transport/transport_ready.py

材料点轨迹输入契约。

负责：

- point_id；
- object_id；
- source visibility；
- source/future投影；
- latent坐标；
- 3D材料点轨迹；
- 点—粒子绑定；
- 保存与验证transport_ready.pt。

### deform_transport/trajectory.py

几何轨迹和坐标映射工具。

重点包括：

- 图像坐标；
- RealWonder裁剪；
- latent坐标；
- 轨迹形状和时间对齐。

### deform_transport/pipeline_integration.py

将预计算transport artifact加载到RealWonder。

支持：

- correct
- shuffled
- flow
- blend

它只读取和校验artifact，不现场计算运输。

### deform_transport/wan_vae_codec.py

Wan VAE编码和解码封装。

### deform_transport/transport_payloads.py

RealWonder图像、点支持区域和运输输入辅助函数。

### deform_transport/transport_metrics.py

Correct、Shuffled及其他transport变体的proxy指标。

---

## 3. 关键脚本

### scripts/export_transport_ready.py

从材料点轨迹和source raster生成transport_ready.pt。

### scripts/run_checkpoint_free_transport_probe.py

不使用Wan VAE完整生成器的运输机制探针。

### scripts/run_wan_vae_transport_probe.py

Wan VAE级Correct和Shuffled实验。

### scripts/build_flow_latent_artifact.py

构建2D RAFT Flow Latent Transport artifact。

### scripts/assemble_final_sim_from_trajectory.py

组装RealWonder final_sim，输出RGB帧和视频。

### scripts/run_santa_validation_suite.py

Santa验证入口。

---

## 4. 当前正式结果入口

路径：

server_runs/20260804_234925_autonomous_deformtransport/current/santa

主要名称：

- 01_realwonder_baseline.mp4
- 02_point_hard_correct.mp4
- 03_point_identity_shuffled.mp4
- 04_point_blend_a050.mp4
- 05_flow_transport.mp4
- final_sim
- transport_ready.pt
- latent_artifact_hard.pt
- latent_artifact_blend_a050.pt
- latent_artifact_flow.pt

这些均为符号链接，原始实验目录保持不变。

---

## 5. 五种方法名称

### realwonder_baseline

原始RealWonder粗RGB latent和flow条件生成。

### point_hard_correct

正确材料点身份，nearest-cell硬运输。

### point_identity_shuffled

只打乱材料点与source latent特征身份对应。

### point_blend_a050

Correct transported latent与RealWonder coarse latent按alpha=0.5融合。

### flow_transport

使用二维RAFT flow运输source latent。

---

## 6. 下一步新代码

新文件统一命名：

deform_transport/soft_transport.py

新测试：

tests/test_soft_transport.py

第一阶段方法名：

point_soft_bilinear

第一阶段仅实现：

- source双线性采样；
- target四邻域forward splatting；
- normalized weight aggregation；
- Correct与Shuffled公平性；
- weight mask和weight sum契约。

暂不同时加入：

- depth；
- target visibility；
- learned module；
- denoising-time injection。

---

## 7. 命名规则

未来实验目录统一：

YYYYMMDD_HHMMSS__case__method__purpose

例如：

20260805_190000__santa__point_soft_bilinear__artifact

20260805_193000__santa__point_soft_bilinear__generator

20260805_200000__santa__point_soft_shuffled__generator
