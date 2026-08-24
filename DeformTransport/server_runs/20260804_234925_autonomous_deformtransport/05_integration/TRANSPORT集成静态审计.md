# Transport 集成静态审计

## 已确认

1. infer_sim.py 读取 transport_latent_path 与 transport_mode。
2. 未提供 transport_latent_path 时不调用 loader，Baseline 原路径保持不变。
3. Correct/Shuffled 从同一 artifact 选择不同 fused latent。
4. loader 检查 5 维 shape 与 freshly encoded sim_latent 完全一致。
5. loader 检查浮点与有限值，并转换到 reference device/dtype。
6. transport_mask 与 contribution_count 共用，且验证 mask == count > 0。
7. 当前 artifact 的 Correct/Shuffled tensor 不同，mask/count 共享且非空。
8. 替换后的 sim_latent 被传入 CausalInferencePipelineSDEdit。
9. SDEdit 在首个 denoising timestep 对 sim_latent 加噪，并以此启动去噪，所以 transport 静态上确实进入生成器。
10. 模型预测参数化为 flow prediction，并转换为 x0；当前方法不是直接修改预测 noise 或 velocity。

## 当前方法的精确语义

它在生成器外预先构造 masked-replace fused coarse latent，然后替换 RealWonder freshly encoded sim_latent。生成器内部不会在每个去噪步重新运输材料点，也没有可调 lambda。transport 的影响来自 SDEdit 初始 latent 和可选 mask/franka drop-in，而不是 CamProbe 式逐步 denoising warp。

## 未通过或未验证

- 尚未有完整 final_sim，无法实测 loader shape、调用次数、注入前后 norm 或输出差异。
- artifact 不含 case 名、输入 hash、checkpoint hash或坐标元数据；loader 只验证 shape，不能识别同 shape 的错误案例。现有注释中防止不同 case 的表述强于实际检查。
- loader 不验证 freshly encoded baseline latent 与 artifact 构造时的 target latent 数值一致；若从压缩 MP4 重建 frames，会形成不公平对照。
- 没有运行期 mask coverage、count、Correct/Shuffled差异、注入前后差异和 transport 调用次数日志。
- 没有 lambda 或 timestep-dependent 融合强度。
- 没有完整生成结果，因此不能声称优于 Baseline 或 Shuffled。

## 后续最小修改候选

只有在完整输入和首次 Baseline smoke 后再修改源码。首要候选是给 artifact 增加 case、输入文件 hash、latent shape/时序与 target latent hash，并加入轻量运行日志；不先加入 soft splatting 或复杂 refiner。
