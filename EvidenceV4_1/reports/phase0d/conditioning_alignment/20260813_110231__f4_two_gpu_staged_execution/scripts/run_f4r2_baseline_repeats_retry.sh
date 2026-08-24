#!/usr/bin/env bash
# F4-R2 retry after documented historical-suite binding failure; baseline paths only.
set -euo pipefail
W=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution
E=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py
P=/workspace/tools/miniforge3/envs/wan-move/bin/python
O="$W/evaluation/f4r2_baseline_only_repeats"
S="$O/baseline_suite"
mkdir -p "$S/santa/dt_full"
ln -sfn /workspace/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4 "$S/santa/dt_full/santa_dt_full_correct_seed0.mp4"
sha256sum "$E" > "$O/evaluator_sha256.txt"
for i in 1 2 3 4 5; do
  d="$O/retry_run$(printf '%02d' "$i")"
  mkdir -p "$d"
  date -Is > "$d/start_time.txt"
  "$P" "$E" --root /workspace/DeformTransport --suite "$S" --out "$d" --mode motion --case santa --batch 8 > "$d/motion.log" 2>&1
  date -Is > "$d/end_time.txt"
done
