#!/usr/bin/env bash
set -u
P=/workspace/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/nodepth_final_20260818T190000Z
W=$P/gpu_opportunity_watcher
L=$W/watcher.log
S=$W/watcher_state.json
H=$W/launch_history.csv
LAUNCH=$P/run_nodepth_formal.sh
ATT2=$P/nodepth_gpu1_run_attempt2
mkdir -p "$W"
test -f "$H" || printf 'attempt,gpu,total_mib,used_mib,free_mib,reason,start_utc,output_dir,pid\n' > "$H"
log(){ printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$L"; }
while :; do
  if test -f "$ATT2/pid.txt" && kill -0 "$(cat "$ATT2/pid.txt")" 2>/dev/null; then printf '{"state":"attempt2_running"}\n' > "$S"; log ATTEMPT2_RUNNING; sleep 20; continue; fi
  snap=$(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null) || { log SNAPSHOT_FAILED; sleep 20; continue; }
  best_gpu=; best_free=-1; best_total=; best_used=; best_reason=
  while IFS=, read -r g total used free; do
    g=$(echo "$g"|tr -d ' '); total=$(echo "$total"|tr -d ' '); used=$(echo "$used"|tr -d ' '); free=$(echo "$free"|tr -d ' ')
    procs=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$' || true)
    reason=; test -z "$procs" && reason=NO_COMPUTE_PROCESS; test "$free" -ge 40000 && reason=${reason:+BOTH}${reason:-FREE_MEMORY_GE_40000}
    if test -n "$reason" && test "$free" -gt "$best_free"; then best_gpu=$g; best_total=$total; best_used=$used; best_free=$free; best_reason=$reason; fi
  done <<< "$snap"
  if test -z "$best_gpu"; then printf '{"state":"watching","snapshot":"%s"}\n' "$(echo "$snap"|tr '\n' ';')" > "$S"; log NO_ELIGIBLE_GPU; sleep 20; continue; fi
  ts=$(date -u +%Y%m%dT%H%M%SZ); n=$(($(wc -l < "$H"))); o=$P/nodepth_auto_attempt_${n}_gpu${best_gpu}_${ts}; mkdir -p "$o"
  printf '%s\n' "$snap" > "$o/gpu_snapshot_at_selection.txt"; printf '%s\n' "PYTORCH_CUDA_ALLOC_CONF_PRESENT=False" > "$o/runtime_policy.txt"; printf '%s\n' "/bin/bash $LAUNCH $o $best_gpu" > "$o/exact_command.txt"
  unset PYTORCH_CUDA_ALLOC_CONF
  nohup /bin/sh -c "unset PYTORCH_CUDA_ALLOC_CONF; /bin/bash '$LAUNCH' '$o' '$best_gpu'; rc=\$?; printf '%s\\n' \"\$rc\" > '$o/exit_code.txt'" > "$o/stdout.log" 2> "$o/stderr.log" < /dev/null & pid=$!
  printf '%s\n' "$pid" > "$o/pid.txt"; printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$n" "$best_gpu" "$best_total" "$best_used" "$best_free" "$best_reason" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$o" "$pid" >> "$H"
  printf '{"state":"launched","attempt":%s,"gpu":%s,"free_mib":%s,"pid":%s}\n' "$n" "$best_gpu" "$best_free" "$pid" > "$S"; log "LAUNCHED attempt=$n gpu=$best_gpu free=$best_free pid=$pid reason=$best_reason"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  rc=$(cat "$o/exit_code.txt" 2>/dev/null || printf unknown)
  if test "$rc" = 0 && test -s "$o/nodepth_formal_correct_v3d_seed000.mp4"; then printf '{"state":"generation_success","attempt":%s}\n' "$n" > "$S"; log GENERATION_SUCCESS; exit 0; fi
  if grep -qi 'OutOfMemoryError\|CUDA out of memory' "$o/stderr.log" && ! test -s "$o/nodepth_formal_correct_v3d_seed000.mp4"; then printf 'ATTEMPT_FAILURE_CLASS=RESOURCE_CAPACITY_OOM\n' > "$o/failure_class.txt"; log RESOURCE_OOM_REARM; continue; fi
  printf '{"state":"non_resource_failure","attempt":%s,"exit_code":"%s"}\n' "$n" "$rc" > "$S"; log NON_RESOURCE_FAILURE_STOP; exit 1
done
