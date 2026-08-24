# Phase0D-1 — Formal Seed / Generation / Evaluation Contract Audit

## 1. 阶段目标
在任何 formal GPU generation 前冻结 Correct corrected-v2 Santa 的 seed、input、generation、RNG、evaluation 与 cached-output contract。

## 2. 审计问题
确认正式 multi-seed generation 仅改变预注册 seed，且具有 corrected-v2 N=1257 generation entrypoint、generator-consumed conditioning invariance evidence 与 formal evaluation reference lineage。

## 3. 使用的数据
Phase0C status、corrected-v2 tracks/visibility/IDs、0B-4R authoritative depth sidecar、official aligned source image/prompt、frozen Wan-Move source及有界 historical output inventory。

## 4. 使用的方法
只读 SHA/shape/lineage audit、static RNG call-path audit、bounded runner/evaluator/cache search、GPU2 status preflight。未生成视频、未 instantiate WanMove 14B。

## 5. 关键命令/脚本
定位 `run_v3d_formal_validation.sh`、frozen `wan_move.py`/`trajectory.py`、historical texture evaluator。该 located runner 和 evaluator 仅作 disqualification evidence。

## 6. 关键结果
Previous Stage、formal input及 frozen source gate均 PASS。seeds frozen to [0,1,2,3,4]，determinism seed=0。Diffusion noise uses a per-seed `torch.Generator`; global Torch RNG is reseeded before trajectory. `track_feats` consumes one randperm but is not supplied to model args; static/algebraic V3D structure is seed invariant, although real-latent edited_y SHA is not functionally measured.

Critical blockers: the only located V3D Santa runner binds historical N=1277 tracks; no corrected-v2 N=1257 formal command/sidecar binding was found. The only located evaluator is a historical source-texture proxy and does not provide formal future-video reference/GT lineage. Cached Santa MP4s are N=1277 and `REUSABLE=False`.

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0D1_STATUS = UNRESOLVED`。这是 formal generation/evaluation contract 未冻结，不是 source/input drift。`PROCEED_TO_PHASE0D2=False`。

## 8. 对后续实验影响
在主对话冻结 corrected-v2 N=1257 generation runner和 formal reference/evaluator之前，不得启动 GPU multi-seed generation。固定 seeds 不得因质量改选。

## 9. 遗留问题
- corrected-v2 N=1257 exact generation entrypoint/sidecar paths unresolved;
- formal future-video reference/evaluation lineage unresolved; historical texture proxy is not GT;
- real-latent functional edited_y hash across seeds remains unmeasured;
- FVD single-clip seed floor is not applicable;
- historical N=1277 Santa remains legacy/rejected.
