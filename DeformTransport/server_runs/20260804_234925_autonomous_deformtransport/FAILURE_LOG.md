# 失败与异常日志

## F001：本地沙箱启动失败

CentOS 7 / Linux 3.10 不支持 bwrap 创建用户命名空间。经授权改用宿主执行；所有写入限定在项目 `server_runs`，未扩大目录范围。

## F002：旧版 Docker formatter 不兼容

当前 Docker 版本的 `docker ps --format` 不暴露 `.State`。改用 `docker inspect` 只读确认容器正在运行。

## F003：严格 shell 下 Conda 激活失败

`set -u` 时 Conda binutils 激活脚本读取未定义 `ADDR2LINE`。后续仅在 Conda 激活段禁用 nounset，任务脚本本身保持失败即退出。

## F004：无 GPU 时 infer_sim 导入失败

`infer_sim.py` 导入阶段的 `vidgen/memory.py` 调用 `torch.cuda.current_device()`，所以显式隐藏 CUDA 时连 `--help` 也不能完成。这不是依赖缺失；GPU 入口验证排队等待独占卡。

## F005：完整 Santa final_sim 缺失

现有 `transport_ready.pt` 的路径指向旧机器；本机没有对应 21 帧原始 PNG、flows 和完整 `final_sim`。未从压缩 MP4 伪造公平输入，完整生成保持阻塞。

## F006：GPU3 抢占门禁失败

2026-08-05 00:26 与 00:28 检查时 GPU3 已出现其他用户 PID 257153。未创建我方锁、未启动共享任务、未干预该进程。

## F007：新 smoke 脚本首次 CPU dry-run 导入失败

深层运行目录脚本首次未将仓库根加入 `sys.path`，导致 `deform_transport` 无法导入。已增加基于脚本绝对路径的仓库根注入；随后 py_compile 和 `--help` 均通过。显式隐藏 CUDA 的完整前置检查已通过 checkpoint/artifact 哈希、shape、有限性和 mask/count 门禁，并按预期只在 `CUDA 不可用` 处停止。

## F008：Wan VAE 启动归档哈希读取失败

首次 GPU2 启动尝试 `WAN_VAE_E2E_21F_20260805_020119` 在创建 GPU 锁和进入 CUDA 前，因宿主无法解析容器内 checkpoint 软链接而退出。已改为在 `deformtransport-dev` 内计算 SHA256；失败目录保留。

## F009：video proxy v1 初始帧尺寸违反 RealWonder 契约

`WAN_VAE_E2E_21F_20260805_020232` 在 GPU2 加载模型后因 `initial_rgb` 为 832×480 而退出码 1；无 CUDA/OOM/共享冲突。v1 原样保留。精确历史 `frame_initial.png` 经官方预处理与历史源图逐像素一致（MAE=0、最大差=0），据此生成 v2。

## F010：v2 仅修正 initial_rgb 仍不足以完整执行

v2 除 `paths.initial_rgb` 外所有 transport 字段完全不变，但 21 个 `coarse_rgb_frames` 同样是 832×480，而探针统一要求 512×512。未浪费 GPU 重试 v2；另建 v3，把视频代理帧嵌入方形画布后转成 512×512，官方预处理往返平均 MAE 1.649/255，并明确标为有损工程 proxy。

## F011：探针整卡显存辅助统计读取了错误物理卡

`_gpu_used_memory_mib()` 固定执行 `nvidia-smi --id=0`；`CUDA_VISIBLE_DEVICES=2` 不会重映射 nvidia-smi 的物理编号，因此报告的 15738MiB 是 GPU0。资源结论改用 PyTorch 峰值和 watcher 的 GPU2/进程级采样；计算结果不受影响。

## F012：本机不存在官方完整 final_sim

在当前用户范围内对 `final_sim`、`noises.npy`、simulation 输出及时间戳目录做完整搜索，未发现可直接运行的官方 Santa `final_sim`。官方 Hugging Face `example_data` 只有首帧、prompt 和 noise，不包含本地案例的 simulation frames/config，不能冒充 Santa 公平输入。现用目录明确标注为有损工程 proxy，不是 future GT。

## F013：RAFT/noise 重建的两个独立前置失败

首次因深层脚本未注入仓库根导致项目导入失败；修正后第二次因 `rp.resize_images` 隐式依赖未安装的 OpenCV 失败。未增加依赖修复次数，改用 Torch area 做等尺寸 0.5× 预缩放并令 NoiseWarper 不再重复缩放；第三次 exit 0。三次根因不同，失败现场均保留。

## F014：完整 Baseline 暴露 FlashAttention 硬依赖与不完整 SDPA 回退

首次完整 Baseline 已进入 generator，但 CLIP 直接调用 `flash_attention(..., version=2)`，环境无 flash-attn，触发硬断言。加入仓库已有 SDPA 回退后，GPU 数值探针进一步证明旧回退忽略 `k_lens`，最大差 1.7734375，因此主动停止仅属于本项目的无效重试。补齐 q/k 长度、因果和局部窗口 mask 后，无 mask 与 `k_lens` 数值回归最大差均为 0；随后完整 Baseline exit 0。源码修改前后与 diff 均归档在 `02_code_audit`。

## F015：I2V VAE 条件探针参数类型错误

独立 GPU3 探针首次错误传入 bfloat16，而 `infer_sim.py` 原生 `processor_dtype` 为 float32，导致输入与 bias dtype 不同。按源码改回 float32 后一次重试通过；GPU2 完整任务不受影响。

## F016：seed1 启动目录命名字段丢失

复制 Baseline 启动器后用 Perl 替换含 shell 变量的字符串，Perl 提前解释了 `${upper}`/`${stamp}`，导致目录名成为 `REALWONDER_SANTA__SEED1_`，输出名成为 `santa__seed1.mp4`。实际命令明确为 `--seed 1`，输入、checkpoint、日志、PID、exit 0 和视频均完整，因此仅是归档命名缺陷，不影响科研结果；原目录保留，不重跑。

## F017：watcher 后台恢复被父命令环境回收

原 watcher PID 417602 已退出。首次用 nohup 恢复的 PID 283876 被命令执行环境在父命令结束时回收，没有产生第二实例。随后改为一个持续受管会话运行同一原脚本，唯一 watcher PID 更新为 291997，继续写入原 CSV；未创建并行 watcher。
