# DeformTransport Evidence V4.1
# Repository-wide Temporal / Contract Audit

## 1. 阶段目标

在进入 Phase0B 前，对当前 DeformTransport 仓库及历史正式实验链进行
repository-wide contract audit，确认 raw simulation、aligned transport、
Wan-Move bridge、Correct/Shuffled 和 formal launcher 是否使用一致的
source/future 时间定义。

审阅源码版本：

- branch: code-review-ready
- commit: 9a70c7de75113281b27396272be3c33046c5f93b

---

## 2. 审计问题

1. 已经修复的 Santa source/frame0 问题为何会在 Wan-Move bridge 中重新出现；
2. 哪些 downstream scripts 绕过 canonical aligned contract；
3. 历史 Santa V3 系列是否使用错误 bridge；
4. Tree 是否存在相同问题；
5. SandHouse raw state0 是否等于 pristine source；
6. 历史 V1 latent Correct/Shuffled 是否被同一问题污染；
7. 后续 Phase0B 应使用哪个唯一 canonical input。

---

## 3. 使用的数据

### Santa

Raw:
OFFICIAL_SANTA_81F_CHAIN_20260805_050719/final_sim/point_trajectories.pt

Canonical aligned:
official_santa_81f_aligned_contract_20260806_192643

Rejected historical bridge:
20260809_010015__santa_correct_tracks

Rejected corrected-old bridge:
20260811_024330__santa_corrected_physical_visibility

Formal corrected-v2:
20260811_224005__santa_corrected_v2_aligned_timeline

### SandHouse

Raw:
sand_house_cached_sim_20260808_174356/raw_sim/point_trajectories.pt

Historical formal sidecar:
sandhouse_frame_ids_165_to_81.npy

---

## 4. 使用的方法

- GitHub source-code inspection；
- raw trajectory metadata inspection；
- exact source/frame0 tensor equality；
- authoritative alignment lineage comparison；
- launcher path audit；
- historical bridge report inspection；
- source/future intervention implementation inspection。

所有时间等价结论要求 exact equality；
未预注册 equivalence margin，因此小数值差异不解释为“等价”。

---

## 5. 关键源码发现

### Santa

Raw Santa trajectory：

- initial_points_* 为独立 source；
- points_* 为 simulation steps 10..810。

Canonical aligned builder 正确构造：

- aligned frame0 = source / step0；
- frames1..80 = old future0..79 / steps10..800。

但历史 Santa Wan-Move exporter 重新读取 raw point_trajectories.pt，
直接使用 points_uv[0] 进行 source-cell sampling，
绕过了 canonical aligned artifact。

因此此前已经修复的数据资产没有失效；
后续 exporter 发生了 pipeline-stage contract bypass。

### SandHouse recorder

simulation/genesis_simulator.py 的顺序为：

1. custom_simulation(sid)
2. scene.step()
3. 取得更新后的 object points
4. recorder.record(simulation_step=sid)

因此 recorder 保存的是 post-step physical state，
但 simulation_step metadata 使用 pre-step sid。

PointTrajectoryRecorder 另外保存真正的 initial_points_*。

---

## 6. 关键结果

### Santa raw

frame_ids = 0..80
simulation_steps = 10,20,...,810

initial_points_uv shape = (28264,2)
points_uv shape = (81,28264,2)

因此：

initial = true source
points_uv[0] = first future state

### Historical Santa bridge 20260809_010015

N = 1277

source =
raw final_sim/point_trajectories.pt

visibility global fraction = 1.0

因此：

- temporal contract FAIL；
- physical/raster visibility contract FAIL；
- true-source VAE-cell sampling FAIL。

Historical Santa V3S/V3B/V3C/V3D/V3E 及基于该 bridge 的
formal Correct/Shuffled 不再满足当前正式输入契约。

### Corrected-v2

N = 1257
step0..800 aligned timeline
true-source sampling PASS
geometry/visibility alignment PASS

它是后续 Santa formal audit 的唯一 canonical bridge。

### SandHouse

raw metadata：

frame_ids = 0..164
simulation_steps = 0,2,...,328

但 exact source check：

UV frame0 == initial: False
max abs diff = 0.44915771484375 px

XYZ frame0 == initial: False
max abs diff = 0.0005390280857682228

Depth frame0 == initial: False
max abs diff = 0.000102996826171875

Projection-valid frame0 == initial: True

因此 metadata step0 不是 pristine source state。

历史 165->81 raw state index selection：

0,2,...,160

sidecar 与实现 exact equal。

对应 metadata steps：

0,4,...,320

但 selected raw index0 仍然不是 pristine source。

### V1 latent transport

V1 使用：

source_points_2d_latent
vs
points_2d_latent[future indices]

source 和 future 在 transport-ready contract 中独立保存。

当前 repository audit 未发现 V1 使用 raw points_uv[0] 作为 source，
因此本次 frame0 bug 不构成对历史 V1 Correct>Shuffled 结论的否定证据。

---

## 7. PASS / FAIL / UNRESOLVED

### PASS

- Santa canonical aligned builder；
- Santa corrected-v2；
- Tree aligned exporter code contract；
- SandHouse 165->81 raw-index subsampling implementation；
- historical V1 source/future separation against this specific bug。

### FAIL

- Santa historical 1277 bridge；
- Santa corrected-old bridge；
- historical Santa Wan-Move V3 formal input contract；
- SandHouse claim that raw frame0 is pristine source；
- historical SandHouse formal source-time contract。

### UNRESOLVED

- numerical effect size of the small SandHouse source offset on generated-video metrics；
- methods whose historical formal input contract failed must be revalidated
  only if they are needed as final paper evidence。

---

## 8. 对后续实验的影响

Phase0B must:

1. use only Santa corrected-v2 N=1257；
2. use aligned step0..800 timeline；
3. never use historical 1277 Santa bridge；
4. explicitly preserve source state separately from future trajectory；
5. verify Correct vs Identity-Shuffled by exact equality of all
   non-intervened variables；
6. never infer source semantics from raw array index 0；
7. treat old Santa/SandHouse Wan-Move formal results as legacy exploratory
   evidence rather than formal paper results。

No old GPU experiment is rerun merely because it is legacy.
Only experiments required by the final formal method will be regenerated.

---

## 9. 遗留问题

The next question is Phase0B:

Does Correct vs Identity-Shuffled change ONLY the intended
source material identity/correspondence variable?

This must be demonstrated on the corrected-v2 authoritative input
before any performance comparison is accepted.

---

## Final Decision

REPOSITORY TEMPORAL / CONTRACT AUDIT = CLOSED

Formal Santa input:
corrected-v2, N=1257, step0..800.

Historical 1277 Santa Wan-Move evidence:
LEGACY / FORMAL REJECT.

Historical SandHouse V3D evidence:
LEGACY / FORMAL REJECT because raw frame0 is not exact pristine source.

Proceed to:
Phase0B — Correct vs Identity-Shuffled Causal Contract Audit.
