#!/usr/bin/env bash
set -u
P=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/nodepth_final_20260818T190000Z
W=$P/gpu_opportunity_watcher
exec 9>"$W/evaluator_finalizer.lock"
flock -n 9 || exit 0
state(){ t=$W/evaluator_finalizer_state.json.tmp.$$; printf '%s\n' "$1" > "$t"; mv "$t" "$W/evaluator_finalizer_state.json"; }
state '{"state":"WAITING_FOR_VALIDATED_VIDEO","duplicate_finalizer_prevention":true}'
printf '%s WAITING_FOR_VALIDATED_VIDEO\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$W/evaluator_finalizer.log"
while :; do
  if grep -q 'FORMAL_EVALUATOR_PENDING_EXECUTION' "$W/companion_closure_state.json" 2>/dev/null; then state '{"state":"EVALUATOR_PENDING_AUTHORIZED_EXECUTOR"}'; printf '%s EVALUATOR_PENDING_AUTHORIZED_EXECUTOR\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$W/evaluator_finalizer.log"; exit 0; fi
  if grep -q 'BLOCKED_BY_' "$W/companion_closure_state.json" 2>/dev/null; then state '{"state":"BLOCKED_BY_VIDEO_OR_GENERATION_FAILURE"}'; exit 1; fi
  sleep 20
done
