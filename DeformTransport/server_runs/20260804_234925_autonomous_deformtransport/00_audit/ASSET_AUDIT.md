# 资产完整性审计

## 审计结论

当前生成端模型权重完整且容器内可读，但缺少任何可直接交给 infer_sim.py 的完整 final_sim 输入目录。Santa 的 transport_ready.pt 和 VAE transport artifact 自身可读、数值有限并通过结构校验；其中记录的外部源路径全部指向旧机器 /home/a/DeformTransport，当前服务器不存在对应 21 帧原始文件。因此现有证据只能支持代理与 VAE-only 结论，不能直接运行完整 RealWonder Baseline/Correct/Shuffled。

## 模型

- RealWonder distilled checkpoint：存在，18,774,247,788 字节，SHA-256 为 3a60efeea42e2f533945a867001e5d4a0f297cfee8c246f1ed8342ba4f0e6f85。
- Wan VAE：存在，507,609,880 字节，SHA-256 为 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981；与既有官方来源记录一致。
- Wan diffusion safetensors：存在，3,128,957,992 字节。
- CLIP：存在，4,772,359,047 字节。
- UMT5：存在，11,361,920,418 字节。
- tokenizer、config、license：存在且可读。
- 仓库 ckpts 和 wan_models 是面向容器的绝对软链接。宿主机视角显示断链是预期现象；在 deformtransport-dev 内解析到 /workspace/model_staging 并可读。
- 未发现第二份同尺寸模型副本；当前链接共享同一份 model_staging 文件，不属于重复占盘。

## Santa transport_ready.pt

- 路径：artifacts/transport_validation/santa_cloth_21f/transport_ready.pt。
- 大小：34,229,269 字节。
- SHA-256：2bb3d0fe4beff3a6ef7e6d66fa8f5438f5bb0172264dfb561cb7254368fa6664。
- validate_transport_ready：通过。
- 轨迹：21 x 28,264 x 3，float32，全部有限。
- binding：28,264 x 5，int64。
- 首帧可见点：23,921。
- 投影：render 512 x 512，video 480 x 832，latent 60 x 104。
- camera：K 为 21 x 1 x 4 x 4，R 为 21 x 1 x 3 x 3，T 为 21 x 1 x 3。
- 所有 paths 项均指向旧机器 /home/a/DeformTransport；本机不存在对应 21 帧轨迹、原始 PNG、flow 或 source raster 文件。
- 大型张量内容没有写入审计记录，仅记录形状、类型和有限性。

## VAE transport artifact

- 路径：artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt。
- 包含 source、target、Correct、Shuffled、fused latent、共享 mask/count 和 permutation。
- Correct/Shuffled fused latent 均为 1 x 6 x 16 x 60 x 104。
- mask/count 均为 6 x 1 x 60 x 104。
- 所有浮点 tensor 有限。
- Correct 在 6/6 latent 时刻和 21/21 解码帧优于 Shuffled；这仍是 coarse-RGB/VAE 代理证据，不是完整生成证据。

## infer_sim.py 真实输入需求

必需：config.yaml、noises.npy、resized_input_image.png、frames/frame_*.png、prompt.txt、RealWonder checkpoint，以及 Wan VAE/T5/CLIP/tokenizer/config。

按配置可选：points_masks_downsampled.pt 仅在 mask_dropin_step > 0 时使用；mesh_masks_downsampled.pt 在存在且非空时使用。flow 不由 infer_sim.py 直接加载，但 noises.npy 的构造来自同一模拟 flow/coarse RGB。point_trajectories.pt 和 camera 是 transport 制备与审计需要，不是 infer_sim.py 直接参数。

当前没有任何 final_sim 目录。现有 2 帧/4 帧 Santa 残片不满足完整生成输入。不能从压缩 MP4 反推原始 PNG 后用于公平对照，因为会令 Baseline 重新编码的 coarse latent 与已保存 transport artifact 的 mask 外 latent 不一致。

## 官方案例

- lamp：静态 cases 完整；demo_data 含相机、点云、mesh、mask；无 final_sim、无未来 GT；刚体。
- persimmon：同上；三刚体；无 final_sim、无未来 GT。
- sand_house：只有输入图、背景图、配置；无 demo_data/final_sim；MPM sand；配置文本包含机械臂动作，但不是已完成真实机器人案例。
- santa_cloth：静态 cases 与 demo_data 前端资产存在；无 final_sim；风驱 PBD cloth，不是机器人案例。
- tree：静态 cases 与 demo_data 前端资产存在；无 final_sim；MPM elastic。
- two_duck：只有静态 cases；无 demo_data/final_sim；刚体。
- 当前无可直接执行三组比较的共同输入 bundle，也无未来真实 GT 视频。

## 完整前端分类

- A 当前生成必需：不需要 SAM 3D Objects、SAM2、Genesis、PyTorch3D 或 Kaolin；只需已有 final_sim。
- B 当前生成不需要：上述前端组件。
- C 新 case 制作需要：SAM 3D Objects、SAM2、Genesis、PyTorch3D、Kaolin。
- D 当前完全不需要：新大型视频模型和无关数据集。

仓库 submodule 目录未安装；realwonder-full 环境存在但 sam2、genesis、pytorch3d、kaolin、cv2、imageio、skimage、lpips 均缺失。当前不盲目安装完整前端。

## 评测资产

- realwonder-gen：imageio 2.37.4 与 imageio-ffmpeg 0.6.0 可用。
- imageio-ffmpeg 自带 ffmpeg 7.0.2；系统 PATH 中无 ffmpeg/ffprobe。
- OpenCV、scikit-image、LPIPS 未安装。
- 项目已有 masked L1/MSE/PSNR 与 transport coverage 代码；尚无完整 SSIM、LPIPS、FVD 实现。
- 在没有未来 GT 视频前，不安装仅供最终评测的额外依赖。

## 下载判断

现阶段无需重复下载任何模型。官方仓库未提供完整 final_sim 下载链接。下一步若要生成新 final_sim，需要恢复前端或获得官方预生成 bundle；这应在 GPU 和任务优先级允许时单独处理。
