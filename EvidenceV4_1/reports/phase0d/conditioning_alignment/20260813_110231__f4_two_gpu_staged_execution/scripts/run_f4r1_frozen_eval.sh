#!/usr/bin/env bash
# Exact-SHA F1-R4 evaluator runner. Candidate bindings are symlinks only.
set -euo pipefail
W=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution
E=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py
P=/workspace/tools/miniforge3/envs/wan-move/bin/python
ROOT=/workspace/DeformTransport
base="$W/evaluation/frozen_evaluator_runs"
mkdir -p "$base"
declare -A src=(
  [DT-FULL]=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4
  [WM-0]="$W/outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4"
  [DT-FRAG-PRUNE]="$W/outputs/frag_container_seed0/santa_correct_v3d_seed000.mp4"
  [DT-GRID100-CENTER]="$W/outputs/grid100_container_seed0/santa_correct_v3d_seed000.mp4"
)
for method in DT-FULL WM-0 DT-FRAG-PRUNE DT-GRID100-CENTER; do
  tag=$(printf '%s' "$method" | tr '[:upper:]-' '[:lower:]_')
  suite="$base/$tag/suite"
  out="$base/$tag/out"
  mkdir -p "$suite/santa/dt_full" "$out"
  ln -sfn "${src[$method]}" "$suite/santa/dt_full/santa_dt_full_correct_seed0.mp4"
  sha256sum "${src[$method]}" > "$out/bound_candidate_sha256.txt"
  "$P" "$E" --root "$ROOT" --suite "$suite" --out "$out" --mode appearance > "$out/appearance.log" 2>&1
  "$P" "$E" --root "$ROOT" --suite "$suite" --out "$out" --mode motion --case santa --batch 8 > "$out/motion.log" 2>&1
done
