#!/usr/bin/env bash
# Observation only: no generation launch, no mutation of output artifacts.
set -euo pipefail
R=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/runtime
PIDS='27680 27786 27994'
log="$R/wave1_observer_transitions.log"
printf '%s WATCHER_STARTED original_pids=%s\n' "$(date -Is)" "$PIDS" >> "$log"
while :; do
 alive=0
 for p in $PIDS; do
  if kill -0 "$p" 2>/dev/null; then alive=$((alive+1)); else test -e "$R/original_pid_${p}_observed_end.txt" || { date -Is > "$R/original_pid_${p}_observed_end.txt"; printf '%s ORIGINAL_PID_EXITED pid=%s\n' "$(date -Is)" "$p" >> "$log"; }; fi
 done
 [ "$alive" -eq 0 ] && break
 sleep 20
done
date -Is > "$R/wave1_all_original_pids_observed_ended.txt"
printf '%s ALL_ORIGINAL_PIDS_OBSERVED_ENDED\n' "$(date -Is)" >> "$log"
