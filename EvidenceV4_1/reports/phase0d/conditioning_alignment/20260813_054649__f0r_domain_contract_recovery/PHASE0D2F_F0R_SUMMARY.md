# Phase0D-2F / F0-R summary

## 1. 阶段目标

仅恢复 N1277 语义、真实 VAE/transport shape，以及 480→464 output trace。

## 2. 审计问题

判断旧 evaluator 的 N=1277 是否污染正式路径，并寻找可审计的真实 latent 与最终高度转换原因。

## 3. 使用的数据

使用前一 F0 archive、冻结 formal runner/source/logs、Replay A/B 文件清单、Wan-Move frozen source，以及 GPU1 preflight。

## 4. 使用的方法

只读源码/日志/文件清单；运行一次协议允许的 VAE-only probe。未加载 T5、CLIP、14B，未作 diffusion generation。

## 5. 关键命令/脚本

`nvidia-smi -i 1`、container 内 `grep/nl` 读取 generator/VAE/trajectory，和一次 `WanVAE.encode`/零 latent decode 形状探针。

## 6. 关键结果

两处 N1277 都是 legacy evaluator guards，且不消费 formal corrected-v2 asset，因此 formal contamination=False。VAE probe 未产出结构化 tensor，Replay 没有中间 tensor。静态公式为 formal H/W=480/832，但与 observed 464x832 最终输出的转换在冻结路径中不存在明确源行。

## 7. PASS/FAIL/UNRESOLVED 判断

N1277 blocker 已排除；真实 latent 与 464 cause 未闭合，最终为 `UNRESOLVED_REAL_LATENT`，`PROCEED_TO_F1=False`。

## 8. 对后续实验影响

不得进入 F1。Phase0B/0C artifact-level causal/structural/differential事实未被推翻；Phase0C 7.24% 仍不能转述为真实 generator-domain fraction。

## 9. 遗留问题

后续需获授权后捕获可靠 VAE tensor/output shape evidence，或发现既存 runtime intermediate；在此之前不应宣布 transport H/W 或 480→464 cause。
