#!/usr/bin/env bash
set -u
P=/workspace/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/nodepth_final_20260818T190000Z
O=$P/nodepth_gpu1_run_attempt2
PID=$(cat "$O/pid.txt")
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/closure_observed_end_utc.txt"
RC=$(cat "$O/exit_code.txt" 2>/dev/null || printf unknown)
printf 'attempt2_exit_code=%s\n' "$RC" >> "$O/closure.log"
if [ "$RC" != 0 ]; then
  if grep -qi 'OutOfMemoryError\|CUDA out of memory' "$O/stderr.log"; then printf 'ATTEMPT2_FAILURE_TYPE=CUDA_OOM\nNO_AUTOMATIC_ATTEMPT3=True\n' >> "$O/closure.log"; fi
  exit 0
fi
exec /bin/bash "$P/attempt2_post_generation.sh"
