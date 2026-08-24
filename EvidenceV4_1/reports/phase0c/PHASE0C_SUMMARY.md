# Phase0C — Operator Intervention Final Closure

## 1. 阶段目标

对 corrected-v2 Santa 冻结 V3D transport operator 完成结构、幅度、carrier funnel 与 Correct-vs-Identity-Shuffled differential decomposition 的机制级闭环归档。

## 2. 审计问题

确认 source-feature correspondence 干预是否保持 future target support、collision/winner structure、depth ordering、carrier contribution funnel 和 write equation 不变；并确认所有 observed conditioning differential 是否由相同 operator 对 source-feature correspondence differential 的传播精确解释。

## 3. 使用的数据

正式 case 为 corrected-v2 Santa：N=1257、T=81、20 future latent slots、latent H×W=60×104、V3D。使用 Phase0B-4R 保存的 Correct/Shuffled context、track_pos、edited_y、authoritative depth/IDs，及 0C-1 至 0C-4A 的冻结证据。历史 N=1277 Santa 为 legacy/rejected，未使用。

## 4. 使用的方法

- 0C-1：按冻结 V3D `(depth, material_id)` arbitration 重建 target support、collision、winner 与 carrier contribution。
- 0C-2：从 saved edited_y 和精确 synthetic harness y 量化 intervention coverage；synthetic amplitude 与 real latent amplitude 分开处理。
- 0C-3：逐 carrier 审计 visibility、target/depth validity、collision 与 winner funnel。
- 0C-3A-R：只读 CSV 确认 visible count 非常量；Spearman 因预注册 SciPy dependency unavailable 而不计算。
- 0C-4：以冻结 bilinear source lookup、winner mapping 与 direct replacement write equation 独立构造 predicted delta，并与 observed delta exact 比较。
- 0C-4A：解释常数 synthetic channel0 的 121 个非零差分为 bilinear float32 lookup numerics。

## 5. 关键命令/脚本

各子审计的 `run_phase0c1.py`、`run_phase0c2.py`、`run_phase0c3.py`、`run_phase0c4.py` 及其 JSON/CSV/summary 均已归档。本阶段仅验证已保存 status JSON 并写入本 closure 三个文件；没有重跑 0C-1/2/3/4、GPU 或 operator。

## 6. 关键结果

- 结构：9766 candidate assignments、9031 unique target writes、722 collision cells、1457 collision carriers、maximum multiplicity=3。Correct/Shuffled target structure、collision、winner IDs/depth、per-carrier contribution 与 zero set 的 mismatch 均为 0。
- 覆盖：future spatial cells=124800；support=9031/9031，fraction=0.07236378205128205；source slot 与 support 外修改均为 0；real Wan latent amplitude=`UNRESOLVED_NO_VALID_CACHED_LATENT`。
- Carrier funnel：1110 carriers 有 candidate，1102 有 contribution，155 为 zero contribution；147 是 sampled future visibility absence，8 是持续 collision loss；Correct/Shuffled funnel mismatch=0。
- 0C-3A-R：visible range=0–20、unique=21、population variance=40.58002758142311。原“zero variance”解释正式撤销；Spearman=`NOT_COMPUTED_DEPENDENCY_UNAVAILABLE`，为 non-gating descriptive statistic。
- Differential：V3D 使用 bilinear `grid_sample` source lookup、lexicographic min(depth, material_id) winner、并在 winning target cell direct full-channel replacement；没有 alpha blend、target interpolation、splatting weight、residual addition、normalization、patch/channel transform。Observed/predicted nonzero scalars=27209/27209，observed-only/predicted-only=0/0，exact equal=True，residual=0。
- 常数 channel0：source synthetic y exact all-one；lookup 后的非一值为 0.9999999403953552，来自 float32 bilinear numerics；20 source carriers 的差异经 winners 使用 121 次，精确对应 121 observed nonzeros，不是 operator confound。

## 7. PASS/FAIL/UNRESOLVED 判断

```text
PHASE0C_STATUS = PASS
OPERATOR_STRUCTURAL_EQUIVALENCE = PASS
INTERVENTION_LOCALIZATION = PASS
CORRECT_SHUFFLED_SUPPORT_EQUIVALENCE = PASS
COLLISION_AND_WINNER_EQUIVALENCE = PASS
CARRIER_FUNNEL_EQUIVALENCE = PASS
DIFFERENTIAL_DECOMPOSITION = PASS
UNEXPLAINED_OPERATOR_DIFFERENTIAL = NONE_DETECTED
REAL_LATENT_AMPLITUDE = UNRESOLVED_NO_VALID_CACHED_LATENT
PHASE0C3_SPEARMAN = NOT_COMPUTED_DEPENDENCY_UNAVAILABLE
CHANNEL0_NUMERICAL_ORIGIN = BILINEAR_FLOAT32_SOURCE_LOOKUP_NUMERICS
```

正式中文表述：在 corrected-v2 Santa 的冻结 V3D operator 下，Correct 与 Identity-Shuffled 具有完全相同的未来 target support、collision structure、winner material IDs、winning depths、carrier contribution funnel 以及 write equation。两者所有实际 conditioning differential 均可由 source feature correspondence 的改变通过相同 frozen V3D operator 精确解释，未检测到额外的 operator-side differential。

Formal English statement: Under the frozen V3D operator for corrected-v2 Santa, Correct and Identity-Shuffled share identical future target support, collision structure, winner material IDs, winning depths, carrier contribution funnels, and write operations. All observed conditioning differences are exactly explained by source-feature correspondence differences propagated through the same frozen V3D operator, with no unexplained operator-side differential detected.

## 8. 对后续实验影响

后续 formal Correct vs Identity-Shuffled performance comparison 可将差异解释为 source material correspondence conditioning information 的贡献。最终 video-quality superiority 仍必须由独立正式 video/performance experiment 证明。

## 9. 遗留问题

- real Wan latent amplitude remains unresolved.
- Phase0C establishes mechanism/operator validity, not final video superiority.
- Phase0C-3 Spearman was not computed because preregistered SciPy dependency was unavailable.
- historical N=1277 Santa remains legacy/rejected.
- frontmost-support visibility limitation from Phase0A remains unchanged.
- 不得据此宣称 correspondence guarantees better video quality、V3D superiority 已确立、所有 selected carriers contribute、collision 无影响、visibility 是 semantic ground truth，或 synthetic channels 是 real Wan semantic channels。
