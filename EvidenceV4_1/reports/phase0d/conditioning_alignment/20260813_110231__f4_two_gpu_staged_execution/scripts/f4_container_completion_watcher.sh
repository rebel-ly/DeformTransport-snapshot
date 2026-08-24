#!/usr/bin/env bash
# Recovery watcher for an already-running F4 pair. Scheduling observes only named process liveness.
set -euo pipefail
B=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution
W=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution
F=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze
R="$W/scripts/run_subset_container_v3d.sh"
rt="$B/runtime"
alive() { docker exec deformtransport-dev bash -lc "pgrep -f '$1' >/dev/null"; }
while alive 'wm0_tracks.npy' && alive 'frag_prune_tracks.npy'; do sleep 10; done
if ! alive 'wm0_tracks.npy'; then arm=WM-0; gpu=GPU1; cvd=1; else arm=DT-FRAG-PRUNE; gpu=GPU2; cvd=2; fi
printf '%s\n' "$arm" > "$rt/recovered_first_finished_arm.txt"
printf '%s\n' "$gpu" > "$rt/recovered_first_free_gpu.txt"
date -Is > "$rt/grid100_recovered_start_time.txt"
docker exec -e "CUDA_VISIBLE_DEVICES=$cvd" deformtransport-dev bash "$R" 0 \
  "$W/outputs/grid100_container_seed0" \
  "$F/artifacts/grid100_center_tracks.npy" "$F/artifacts/grid100_center_visibility.npy" \
  "$F/artifacts/grid100_center_ids.npy" "$F/artifacts/grid100_center_depth.npy" \
  > "$rt/grid100_recovered_stdout.log" 2> "$rt/grid100_recovered_stderr.log"
code=$?
date -Is > "$rt/grid100_recovered_end_time.txt"
printf '%s\n' "$code" > "$rt/grid100_recovered_exit_code.txt"
