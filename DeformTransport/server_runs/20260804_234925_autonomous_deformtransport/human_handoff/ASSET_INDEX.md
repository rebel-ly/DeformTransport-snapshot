# 资产索引

## 模型与软链接

- 项目 `ckpts` → `/workspace/model_staging/ckpts`（宿主 `/mnt/sdbd/home/liuyu_qyh/model_staging/ckpts`）。
- RealWonder checkpoint：`ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt`；SHA256 `3a60efeea42e2f533945a867001e5d4a0f297cfee8c246f1ed8342ba4f0e6f85`。
- 项目 `wan_models` → `/workspace/model_staging/wan_models`。
- Wan VAE：`wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth`；SHA256 `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`。
- diffusion weights SHA256 `4ec199076538b946935ebcb3ba808d3c427e638f29519a3c3c98d31d821e5eed`。
- CLIP SHA256 `628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38`；T5 SHA256 `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d`。

## Santa proxy 资产

- contract-complete proxy final_sim：`prepared_inputs/santa_21f_final_sim_proxy_v1/`。
- transport 代理：`prepared_inputs/santa_21f_videoproxy_transport_ready/`；错误旧资产保留，修复 v2/v3 不覆盖历史。
- `noises.npy` SHA `75be5439...c8779`；`flows.npy` SHA `c9152822...63f5`。
- Wan VAE artifacts：`artifacts/transport_validation/santa_cloth_21f/wan_vae/` 及 `04_smoke/WAN_VAE_E2E_21F_20260805_022223/outputs/`。
- 资产性质：有损工程 proxy，不是原始 RealWonder final_sim，不是 future GT。

## 官方 Santa 资产

- demo 根：`demo_web/demo_data/santa_cloth/`。
- `_MinimalSVR`：`demo_web/simulation_engine.py`。
- 成功最小官方 smoke：`04_smoke/OFFICIAL_SANTA_SIM_4F_20260805_050419_retry2/`。
- 81 帧可复用物理输出：`04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719/simulation_source/`，含 81 张 frame、flows、point trajectories 和 report。
- 81 帧 partial final_sim：同 chain 的 `final_sim/`；缺 `simulation.mp4`、noises 及完整校验，禁止当作 READY。
- 原始 `right_s1_21f` 无损 PNG 目录：未找到；不要把 proxy 冒充该资产。

## 环境资产

- Genesis 固定 checkout/commit：`3aa206cd84729bc7cc14fb4007aeb95a0bead7aa`。
- 隔离仿真 venv：`/workspace/tools/venvs/deformtransport-sim`。
- 用户态 GL overlay：`/workspace/tools/conda-libs/deformtransport-gl`。
- 生成环境：`/workspace/tools/miniforge3/envs/realwonder-gen`。
- 环境安装日志：`01_environment/SIM_RUNTIME_RECOVERY_20260805_043358/`。

## 台账与完整清单

- 模型哈希：`00_audit/model_sha256.txt`。
- 原始结果文件清单：`human_handoff/RESULT_RAW_INVENTORY.txt`。
- 全部未跟踪文件：`human_handoff/UNTRACKED_FILES.txt`。
