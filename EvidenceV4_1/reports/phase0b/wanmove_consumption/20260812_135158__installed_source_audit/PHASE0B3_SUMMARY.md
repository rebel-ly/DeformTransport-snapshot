# Phase0B-3 Wan-Move Consumption-Path Audit

## 1. 阶段目标

验证实际安装并用于 DeformTransport 实验的 patched Wan-Move
是否将 Identity-Shuffled 的 t0 坐标仅作为 source-feature lookup，
而不是作为 velocity、displacement 或 source-relative motion 输入。

## 2. 审计问题

Phase0B-2 已证明 Identity-Shuffled 会改变完整 polyline 的
frame0->frame1 隐含位移。

本阶段判断该差异是否真正进入 Wan-Move 的 generator-consumed
motion representation。

## 3. 使用的数据

Wan-Move git HEAD:

80c58a7d2ad175fa82a4d57f79f2a1415317dcfa

Installed modified files:

wan/wan_move.py
SHA256:
aca79f9cc4bf32ea363c4440ed2c7e7d90ef5aa763f3e96ae6c2b8eff35c1857

wan/modules/trajectory.py
SHA256:
0c6bc94d8ce1f885f0333314a9b201a650163cd209b2a3b3f95b4f3a35a49dae

## 4. 使用的方法

- exact installed-source archival；
- Git working-tree provenance；
- Python AST function extraction；
- all subtraction-expression extraction；
- randomness-call extraction；
- source/target/depth/ID data-flow inspection；
- motion-difference vocabulary audit。

## 5. 关键命令/脚本

Audited functions:

- WanMove.generate
- create_pos_feature_map
- replace_feature
- _dt_original_replace
- _dt_bilinear_source_features
- _dt_load_sidecars
- _dt_mode

## 6. 关键结果

Wan-Move conditioning path is:

t0 source coordinate
-> source VAE feature lookup

future absolute coordinates
-> future target-cell placement

For V3D, source feature sampling uses tracks_sampled[0].

Future target positions are consumed separately through track_pos.

No source-target displacement, consecutive-frame velocity,
source-relative displacement, torch.diff(track), or equivalent
motion-difference encoding was detected.

Subtractions in the audited critical functions are limited to:

- temporal tensor-size arithmetic；
- pixel-to-latent center conversion；
- normalized grid coordinates；
- future point distance to its target latent-cell center。

The latter is target-cell arbitration and does not subtract source
coordinates.

Depth, persistent material IDs, tracks, and visibility are permuted
together when Wan-Move performs its internal track reordering.

Trajectory-internal RNG is reset by torch.manual_seed(seed) before
trajectory processing.

Diffusion noise uses a separate explicitly seeded torch.Generator.

Correct and Identity-Shuffled therefore execute the same code path
and paired random sequence when N, tensor shapes, seed and variant
are fixed.

## 7. PASS/FAIL/UNRESOLVED 判断

PASS:

- source coordinate used as source-feature address；
- future coordinates used as target placement；
- visibility/depth/material-ID alignment；
- paired trajectory RNG；
- paired diffusion noise；
- no detected generator-consumed velocity/delta/source-relative
  motion confound。

Array-level full-track kinematic invariance remains FAIL because
the shuffled source coordinate changes the implied frame0->frame1
polyline segment.

This is not a generator-consumed motion confound in the audited
implementation.

Phase0B-3 overall:

PASS

## 8. 对后续实验影响

Identity-Shuffled may continue as the formal causal control.

The valid interpretation is:

Future material states, future visibility, and target geometry are
preserved exactly while the assignment between source locations
and persistent future material identities is permuted.

Do NOT claim that the complete 81-frame polyline trajectory is
identical between Correct and Identity-Shuffled.

## 9. 遗留问题

Phase0B-4 must perform a functional generator-condition audit and
verify that, with identical seed and source VAE condition, the
Correct/Shuffled difference enters edited_y only through the
intended source-feature-to-future-target correspondence pathway.

No full video generation is required unless the functional
conditioning audit cannot close this question.

## Final Decision

PHASE0B3_STATUS = PASS

WAN_MOVE_CONSUMPTION_CAUSAL_ISOLATION = PASS

GENERATOR_CONSUMED_MOTION_CONFOUND = NOT_DETECTED
