# 科研结果索引

路径均相对于 `server_runs/20260804_234925_autonomous_deformtransport/`。每项的完整命令以运行目录内 `command.sh` 为权威原样记录；`manifest.yaml`、`stdout.log`、`stderr.log`、`exit_code.txt` 均保留，不在此复制以避免转录误差。

## 完整 RealWonder generator + Santa 工程 proxy

| 结果 | 运行目录 | 输出视频与 SHA256 | metrics | exit |
|---|---|---|---|---|
| Santa Baseline seed0 | `04_smoke/REALWONDER_SANTA_BASELINE_20260805_032928` | `santa_baseline_seed0.mp4` — `41bd8a36aab27a2c6f78137e836d1ada96ee574ce20f1d83d7c3fb42edc3bcd8` | `05_metrics/realwonder_santa_proxy_trio_20260805/` | 0 |
| Correct seed0 | `04_smoke/REALWONDER_SANTA_CORRECT_20260805_033343` | `santa_correct_seed0.mp4` — `0dba5fb9601a48dfe03348a7bb2b0ce5c47f19e268296ebd7ea4c8c7a85ed394` | 同上 | 0 |
| Shuffled seed0 | `04_smoke/REALWONDER_SANTA_SHUFFLED_20260805_033730` | `santa_shuffled_seed0.mp4` — `4305628bf2e0e42c68d44d1a969eba5929e802ddc1ee9e336da241c93780eaa6` | 同上 | 0 |
| Flow seed0 | `04_smoke/REALWONDER_SANTA_FLOW_20260805_035223` | `santa_flow_seed0.mp4` — `8f5bb30101bf331f120074ac68a204c556aa245b43ae69ce0651f8270619e45c` | 同上 | 0 |
| Blend α=0.5 seed0 | `04_smoke/REALWONDER_SANTA_BLEND_20260805_040147` | `santa_blend_seed0.mp4` — `e85cc28968d83fbb19fda3b728b7b5caff8b8ca6fe8d9c04bb011fd782fc2e75` | 同上 | 0 |
| Baseline seed0 复跑 | `04_smoke/REALWONDER_SANTA_BASELINE_20260805_034305` | `santa_baseline_seed0.mp4` — 与首次同 SHA | `baseline_repeat*` | 0 |
| Baseline seed1 | `04_smoke/REALWONDER_SANTA__SEED1_` | `santa__seed1.mp4` — `f3e86df58f1bbaa30d70fcdeb0fccbfee2ab13d72a07d25b1afaced4ab5f7871` | `baseline_seed1*` | 0 |

命名说明：seed1 目录中的双下划线是历史命名缺陷，实际 `command.sh` 使用 seed 1。上述均是完整 generator 的实际视频，但输入为有损工程 proxy，不是 official future GT。

主要 proxy 指标（PSNR/SSIM/temporal L1）：Baseline `19.2271/0.7691/0.02098`；Correct `18.9868/0.7786/0.02108`；Shuffled `18.7701/0.7978/0.02156`；Flow `19.0136/0.7743/0.02110`；Blend `19.1389/0.7730/0.02078`；seed1 `19.8096/0.7962/0.02127`。Flow 的 RAFT proxy mean EPE 最低。完整 CSV/JSON 与对比见 metrics 目录和 `完整RealWonder阶段结果审计.md`。

视觉材料：`methods_3x2.mp4` SHA `f6d24c...ab6bd3`，`methods_contact_sheet.jpg` SHA `f5069f...6fc52`，`trio_2x2.mp4` SHA `ad1733...3bc1`，`contact_sheet.jpg` SHA `18574d...b220`；尚未人工视觉确认。

## 其他必须保留的验证

### 8. Wan VAE 21 帧闭环

- 目录：`04_smoke/WAN_VAE_E2E_21F_20260805_022223`
- 完整命令：`command.sh`；manifest/log/exit：同目录对应文件；exit 0。
- 输出：`outputs/fused_correct.mp4` SHA `1f06fdcb6c1ddbce35867769cbe213d142a72d1968ffef350cc3a274c678aeaa`；`outputs/fused_shuffled.mp4` SHA `e8831a0fee45a52bb6abef811b8b9a43a40a36acb43b2831ba8fe57398c15e1f`；`outputs/vae_latent_outputs.pt` SHA `72c1eac59ba130ca88c395d546272b5d95a5b62ea5e281834cb05ad7804e8f31`。
- 实测：44.715 s，peak allocated 4320.06 MiB，reserved 6190 MiB。属于 proxy 证据，不是完整 generator 视频结论。

### 9. transport GPU smoke

- 目录：`04_smoke/DT_TRANSPORT_GPU_PARITY_20260805_015718`
- 命令/manifest/log/exit：同目录；exit 0。
- `outputs/结果.json` SHA `b7e69cb9c4b5f9440d16a95073bc622a24cbcb9148840937c972713fc21e9dff`。
- 28,264 点×6 latent 时刻；Correct/Shuffled 对历史 artifact 最大绝对差均为 0；总计 1.4178 s；Torch peak allocated 26.36 MiB、reserved 30 MiB。

### 10. final_sim/noise 重建

- 目录：`04_smoke/FINAL_SIM_NOISE_RECON_20260805_031213`
- 完整命令及日志：同目录；exit 0。
- 输入/输出：`prepared_inputs/santa_21f_final_sim_proxy_v1`；`noises.npy` SHA `75be54392fae1695bd072fe26c6b417a614b87f4dd4c1768719856d5105c8779`；`flows.npy` SHA `c91528226238519a0252f0adbefd385d1083953146130f96b9da9cd8787263f5`。
- 这是 RAFT/noise-warp 的真实执行，但输入仍是 proxy。

### 11. generator/checkpoint 运行时审计

- generator 加载：`04_smoke/REALWONDER_GENERATOR_LOAD_20260805_030721`，exit 0；完整命令/manifest/log 在目录。只证明 checkpoint 和模型可加载，不是 Baseline 视频。CPU maxRSS 45.60 GiB。
- transport 运行时：`04_smoke/REALWONDER_BASELINE_TRANSPORT_RUNTIME_AUDIT_20260805_032032`，exit 0；证明实际 VAE encode shape `[1,6,16,60,104]`、Baseline 不变、Correct/Shuffled 被 pipeline 接受；不执行去噪生成。
- generator/EMA：`04_smoke/REALWONDER_CHECKPOINT_GENERATOR_EMA_AUDIT_20260805_033029`，exit 0；983/983 tensors 存在差异，global max `1.584e-4`，RMS `1.34044e-5`。

### 12. SDPA 兼容修复与数值等价性

- 审计目录：`02_code_audit/flash_attention_sdpa_fallback_20260805_0324` 与 `02_code_audit/sdpa_padding_mask_fix_20260805_0328`。
- 最终快照 SHA：`939697d25b6cdba718859e7040be8358cd93dbea5e5dc0e15ea6bc8a0147ce71`。
- GPU plain/k_lens 数值差最大值 0；CLIP exact smoke 通过；`02_validation/post_full_generator/unittest.log` 记录 25/25 通过。

## 官方 81 帧 chain：失败但可复用的现场

- 目录：`04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719`
- PID `89794` 已退出；exit 1；最后阶段 `03_assemble_final_sim`。
- `simulation_source/report.json` SHA `8df232018cd8e00e4f3766f9d67cb3269ac42cc829a56a886deb6e14551b1f6b`；`point_trajectories.pt` SHA `ab2160857b63e2b1634513be218a05ff9bc4049a42c8bbed3ce7309e088f5226`。
- 81 帧仿真/flow/轨迹成功；失败根因仅为 assembler 调用 torchvision 写 MP4 时缺少 PyAV。partial final_sim 不能作为完成输入。

## 结论边界

已实际验证、proxy 证据、完整 generator 结果已经在上文区分。尚未人工视觉确认；未证明优于 RealWonder 或 Flow；未完成机器人案例；Santa 不是机器人操作案例；proxy 不是 future GT；项目未完成。
