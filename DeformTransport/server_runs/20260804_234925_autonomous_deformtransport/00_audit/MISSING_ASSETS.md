# 缺失资产

## 阻塞完整生成

- 任一案例的完整 final_sim 目录。
- Santa 21 帧原始 coarse RGB PNG 序列、精确 resized_input_image.png、匹配 noises.npy、config.yaml 和 prompt.txt。
- 原始 21 帧 flow、source raster point indices 与外部 point_trajectories.pt；transport_ready.pt 内虽含自包含轨迹，但旧外部路径不可复核。
- 公平定量评测所需的未来真实 GT 视频。

## 阻塞新案例制作但不阻塞已有 final_sim 生成

- SAM 3D Objects 源码与权重。
- SAM2 源码与权重。
- Genesis 固定 commit 3aa206cd84729bc7cc14fb4007aeb95a0bead7aa。
- PyTorch3D、Kaolin、OpenCV、scikit-image 等前端依赖。

## 机器人案例

- 没有包含初始 RGB、真实机器人动作、未来 GT、模拟轨迹、相机、mask、flow 和 coarse RGB 的完整机器人—可变形物体案例。
- cases/xml/franka_emika_panda 只是机器人描述资产，不能单独构成实验案例。
- sand_house 的配置提示包含机械臂，但本地没有完整模拟或真实 GT，不能作为已验证机器人案例。
