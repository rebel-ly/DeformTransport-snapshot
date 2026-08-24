# Phase0B — Causal Contract Final Closure

## 1. 阶段目标

对 corrected-v2 Santa 的 Correct 与 Identity-Shuffled 输入完成预注册因果契约闭环：确认干预仅改变源帧位置/特征与持续未来材料轨迹之间的对应关系，而不引入生成器实际消费的未来几何、可见性、深度、材料 ID 或随机路径混杂。

## 2. 审计问题

审计 Correct 与 Identity-Shuffled 的 conditioning 差异是否严格来自 source material identity / source feature 到 future material target 的对应关系改变；是否存在 target geometry、visibility、depth、persistent material ID、RNG 或 velocity/delta/source-relative motion confound。

## 3. 使用的数据

- 正式 bridge：corrected-v2 Santa，N=1257、T=81、simulation steps 0,10,...,800。
- Correct tracks、visibility、persistent material IDs：`server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline`。
- Identity-Shuffled seed=0：`causal_contract/20260812_134250__santa_corrected_v2_identity_shuffle_seed0`。
- authoritative aligned transport depth/point-ID lineage：`aligned_transport_ready.pt`；0B-4R 选取并保存 81×1257 depth sidecar。
- 安装的 patched Wan-Move source：Git HEAD `80c58a7d2ad175fa82a4d57f79f2a1415317dcfa`。

历史 N=1277 Santa bridge 为 legacy/rejected，不作为正式证据。

## 4. 使用的方法

- 0B-1：seed=0 source-identity derangement，验证双射、零 fixed points、source coordinate/cell-set 保持，以及 t1..80 future states、visibility、IDs 精确不变。
- 0B-2：比较绝对状态与 transition algebra，单独记录完整 polyline 的 source→first-future segment 已变化。
- 0B-3：对冻结 patched Wan-Move 消费路径做静态审计，检查 source lookup、future placement 与可能的 delta/velocity/source-relative encoding。
- 0B-4/4R：在正常 package import 后，以 CPU-only synthetic VAE latent 做函数级 `create_pos_feature_map` / `replace_feature` paired audit，核对 RNG、context、write support 和 `edited_y` differential localization。

## 5. 关键命令/脚本

- 0B-1 evidence：`causal_contract/20260812_134250__santa_corrected_v2_identity_shuffle_seed0/report.json`。
- 0B-2 evidence：`intervention_algebra/20260812_134638__santa_seed0/report.json`。
- 0B-3 evidence：`wanmove_consumption/20260812_135158__installed_source_audit/phase0b3_status.json`。
- 0B-4 first attempt：`functional_conditioning/20260812_141951__santa_v3d_seed0/traceback.txt`。
- 0B-4R CPU functional harness/output：`functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/phase0b4_status.json` 和 `FINAL_STDOUT.txt`。

本闭环步骤仅读取上述 evidence 并生成本文件、状态 JSON 与 SHA256 清单；没有重跑 0B-1/2/3/4，也没有运行 GPU。

## 6. 关键结果

- 0B-1 PASS：N=1257、seed=0、derangement fixed points=0；t1..80 future coordinates、visibility、material IDs 精确相同；source coordinate multiset 与 source VAE-cell set 保持。
- 0B-2 QUALIFIED_PASS：source frame 与 first transition 不同；future frames 1..80 精确相同，transitions 1..79 精确相同。`FULL_TRACK_KINEMATIC_INVARIANCE=FAIL`。
- 0B-3 PASS：t0 source coordinate 用于 source feature lookup，future absolute coordinates 用于 future target placement；target-source delta、consecutive velocity、source-relative displacement 未检测到；paired RNG PASS。
- 0B-4 first attempt：`CPU_IMPORT_BLOCKED`。这是 normal package import 的工程 blocker，不是 scientific FAIL；该目录完整保留。
- 0B-4R PASS：normal package import PASS；functional compute device=CPU；authoritative depth lineage PASS；internal RNG permutation paired；future context track、visibility、depth、ID、future track_pos mismatch 均为 0；Correct/Shuffled write-support 均为 9031 cells，support mismatch=0；source slot mismatch=0；outside-support conditioning mismatch=0；inside-support conditioning differentials=27209；source coordinate mismatch count=1257。

## 7. PASS/FAIL/UNRESOLVED 判断

```text
PHASE0B_STATUS = PASS
CORRECT_VS_IDENTITY_SHUFFLED_CAUSAL_CONTRACT = PASS
INTERVENTION_ISOLATION = PASS
GENERATOR_CONSUMED_MOTION_CONFOUND = NOT_DETECTED
FUNCTIONAL_CONDITIONING_LOCALIZATION = PASS
FULL_TRACK_KINEMATIC_INVARIANCE = FAIL
```

`FULL_TRACK_KINEMATIC_INVARIANCE=FAIL` 不构成 Phase0B overall FAIL：Identity-Shuffled 改变 source coordinate assignment，故完整 polyline 的 source→first-future segment 数学上不同。patched Wan-Move 不消费该 segment 作为 velocity、delta 或 source-relative motion representation；功能审计进一步证明 future geometry/support、visibility、depth、IDs 与 RNG 相同，conditioning 差异只出现在 intended common target support 内。

正式中文表述：Identity-Shuffled 保持未来材料状态、未来目标几何、visibility、depth、persistent material identity、future latent write support 和随机计算路径不变，仅打乱源帧位置/特征与持续未来材料轨迹之间的对应关系。

Formal English statement: Identity-Shuffled preserves future material states, future target geometry, visibility, depth, persistent material identities, future latent write support, and paired random computation, while permuting the assignment between source locations/features and persistent future material trajectories.

不得表述为 “Correct and Shuffled use identical complete trajectories.”

## 8. 对后续实验影响

在完全相同的 formal protocol / seed 下，如未来观察到 `Correct > Identity-Shuffled`，可解释为 correct material correspondence contributes useful conditioning information。该结论不表示 correspondence alone guarantees better final video quality；视频质量须由后续正式性能实验独立验证。正式性能比较必须只使用 corrected-v2。

## 9. 遗留问题

- Phase0B only establishes causal-contract validity; it does not establish performance superiority.
- formal performance comparison must use corrected-v2 only.
- complete polyline kinematic invariance is false and must not be claimed.
- historical N=1277 Santa experiments remain legacy/rejected.
- 初始 0B-4 的 CPU import blocker 已保留为工程证据；4R 成功恢复不删除或改写该历史记录。
