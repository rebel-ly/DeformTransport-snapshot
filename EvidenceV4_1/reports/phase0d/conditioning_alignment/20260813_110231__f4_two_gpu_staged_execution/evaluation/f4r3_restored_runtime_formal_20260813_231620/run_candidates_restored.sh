#!/usr/bin/env bash
# F4-R3 only: exact host runtime and evaluator; isolated symbolic bindings.
set -euo pipefail
R=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution/evaluation/f4r3_restored_runtime_formal_20260813_231620
E=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py
P=/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/envs/wan-move/bin/python
ROOT=/mnt/sdbd/home/liuyu_qyh/DeformTransport
declare -A V=(
 [wm0]=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution/outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4
 [frag]=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution/outputs/frag_container_seed0/santa_correct_v3d_seed000.mp4
 [grid100]=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution/outputs/grid100_container_seed0/santa_correct_v3d_seed000.mp4
)
for m in wm0 frag grid100; do
 d="$R/candidates/$m"; s="$d/suite"; o="$d/out"
 mkdir -p "$s/santa/dt_full" "$o"
 ln -sfn "${V[$m]}" "$s/santa/dt_full/santa_dt_full_correct_seed0.mp4"
 sha256sum "${V[$m]}" > "$o/bound_candidate_sha256.txt"
 CUDA_VISIBLE_DEVICES=1 "$P" "$E" --root "$ROOT" --suite "$s" --out "$o" --mode appearance > "$o/appearance.log" 2>&1
 CUDA_VISIBLE_DEVICES=1 "$P" "$E" --root "$ROOT" --suite "$s" --out "$o" --mode motion --case santa --batch 8 > "$o/motion.log" 2>&1
done
