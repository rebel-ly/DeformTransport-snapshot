#!/usr/bin/env bash
set -u
P=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/nodepth_final_20260818T190000Z
W=$P/gpu_opportunity_watcher
LOCK=$W/companion_closure.lock
STATE=$W/companion_closure_state.json
FINAL=$W/FINAL_STATUS.json
LOG=$W/companion_closure.log
exec 9>"$LOCK"
flock -n 9 || exit 0
write_state(){ tmp=$STATE.tmp.$$; printf '%s\n' "$1" > "$tmp"; mv "$tmp" "$STATE"; }
write_final(){ tmp=$FINAL.tmp.$$; printf '%s\n' "$1" > "$tmp"; mv "$tmp" "$FINAL"; }
log(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }
write_state '{"state":"WAITING_FOR_SUCCESSFUL_GENERATION","duplicate_companion_prevention":true}'
log WAITING_FOR_SUCCESSFUL_GENERATION
while :; do
  if grep -q '"state":"non_resource_failure"' "$W/watcher_state.json" 2>/dev/null; then write_state '{"state":"BLOCKED_BY_NONRESOURCE_FAILURE"}'; log BLOCKED_BY_NONRESOURCE_FAILURE; exit 1; fi
  if grep -q '"state":"generation_success"' "$W/watcher_state.json" 2>/dev/null; then
    last=$(tail -n 1 "$W/launch_history.csv"); out=$(printf '%s' "$last" | awk -F, '{print $8}'); gpu=$(printf '%s' "$last" | awk -F, '{print $2}'); vid=$out/nodepth_formal_correct_v3d_seed000.mp4
    if test -s "$vid" && test -f "$out/exit_code.txt" && test "$(cat "$out/exit_code.txt")" = 0 && ! grep -qi 'OutOfMemoryError\|Traceback\|RuntimeError\|Killed\|Segmentation fault' "$out/stderr.log"; then
      write_final "{\"state\":\"VIDEO_VALIDATING\",\"successful_attempt\":\"$out\",\"physical_gpu\":$gpu}"; write_state '{"state":"VIDEO_VALIDATING"}'; log VIDEO_VALIDATING
      exec /bin/bash "$P/post_generation_formal_closure.sh" "$out" "$gpu"
    fi
  fi
  sleep 20
done
