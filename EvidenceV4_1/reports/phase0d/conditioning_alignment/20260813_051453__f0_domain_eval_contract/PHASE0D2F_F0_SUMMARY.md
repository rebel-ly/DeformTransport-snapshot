# Phase0D-2F / F0 summary

## 1. 阶段目标

封存 0D-2R-E replay closure，并建立 generation/latent/track/output/evaluation domain contract。

## 2. 审计问题

解释 480x832 请求与 464x832 输出的关系，验证真实 latent HxW，以及 N=1257 evaluator 兼容性。

## 3. 使用的数据

冻结 0D-2R-E evidence、formal runner/logs、formal source image、corrected-v2 tracks/visibility/IDs、Wan-Move source 和 evaluator source。

## 4. 使用的方法

只读 SHA256/manifest/code-path/asset metadata 检查；一次允许的 VAE-only probe 尝试，未获得 tensor 结果。没有 generation 或 evaluator execution。

## 5. 关键命令/脚本

`sha256sum -c` 在原 replay evidence 目录成功；审计 `generate.py`、`wan_move.py`、`vae.py`、`utils.py`、`eval_v3.py` 与 `run_v3_joint_eval.sh`。

## 6. 关键结果

Replay closure PASS；canonical Replay A MP4 SHA 为 `08785a...f935`，decoded RGB SHA 为 `935d93...7cbc6`。请求、源图、轨迹均为 480x832；最终 RGB 为 464x832。轨迹正式映射为 x'=x*w/img_w, y'=y*h/img_h，当前为 identity。真实 VAE tensor 未捕获；480→464 机制未定位。现有 evaluator 硬编码 N=1277。

## 7. PASS/FAIL/UNRESOLVED 判断

`PHASE0D2RE_CLOSURE_AUDIT=PASS`。最终状态 `UNRESOLVED_EVALUATION_CONTRACT`；同时 real latent 和 output-cause 未闭合。

## 8. 对后续实验影响

Phase0D-3 execution 保持 DEFERRED，且 `PROCEED_TO_F1=False`。Phase0C structural causal evidence 不变；7.24% 不能作为 real-domain intervention fraction，直到真实 latent HxW 被捕获。

## 9. 遗留问题

需要在后续获授权阶段以不改历史资产的方法捕获真实 VAE y 形状、定位 480→464，并提供独立 corrected-v2 N=1257 evaluator wrapper/contract；本阶段不执行这些工作。
