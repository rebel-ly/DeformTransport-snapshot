#!/usr/bin/env bash
# Runs detached inside deformtransport-dev.  It never restarts the two initial arms.
set -euo pipefail
W=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution
F=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze
rt="$W/runtime"
log="$rt/persistent_watcher_transitions.log"
stamp() { printf '%s %s\n' "$(date -Is)" "$*" >> "$log"; }
alive() { kill -0 "$1" 2>/dev/null; }
wm_pid=211282
frag_pid=211297
stamp "WATCHER_STARTED wm_pid=$wm_pid frag_pid=$frag_pid"
while alive "$wm_pid" && alive "$frag_pid"; do sleep 10; done
if alive "$wm_pid"; then arm=DT-FRAG-PRUNE; gpu=GPU2; cvd=2; else arm=WM-0; gpu=GPU1; cvd=1; fi
# Atomically claim the one permitted launch. Existing output or marker is a no-launch state.
if ! mkdir "$rt/grid100_launch.lock" 2>/dev/null; then stamp "GRID100_NOT_LAUNCHED existing_lock"; exit 0; fi
grid="$W/outputs/grid100_container_seed0/santa_correct_v3d_seed000.mp4"
if [ -e "$grid" ] || [ -e "$rt/grid100_recovered_end_time.txt" ]; then stamp "GRID100_NOT_LAUNCHED existing_output_or_completion"; exit 0; fi
if ! nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(($cvd+1))p" | awk '{exit !($1 >= 20000)}'; then stamp "GRID100_NOT_LAUNCHED insufficient_free_memory gpu=$gpu"; exit 0; fi
printf '%s\n' "$arm" > "$rt/recovered_first_finished_arm.txt"
printf '%s\n' "$gpu" > "$rt/recovered_first_free_gpu.txt"
date -Is > "$rt/grid100_recovered_start_time.txt"
stamp "GRID100_LAUNCH arm_finished=$arm gpu=$gpu"
set +e
CUDA_VISIBLE_DEVICES="$cvd" bash "$W/scripts/run_subset_container_v3d.sh" 0 "$W/outputs/grid100_container_seed0" \
  "$F/artifacts/grid100_center_tracks.npy" "$F/artifacts/grid100_center_visibility.npy" \
  "$F/artifacts/grid100_center_ids.npy" "$F/artifacts/grid100_center_depth.npy" \
  > "$rt/grid100_recovered_stdout.log" 2> "$rt/grid100_recovered_stderr.log"
code=$?
set -e
date -Is > "$rt/grid100_recovered_end_time.txt"
printf '%s\n' "$code" > "$rt/grid100_recovered_exit_code.txt"
stamp "GRID100_FINISHED exit_code=$code"
if [ "$code" -ne 0 ]; then exit "$code"; fi
# Evaluation is preregistered and starts only after the third output has completed.
stamp "POSTRUN_STARTED"
python "$W/scripts/f4_postrun_container.py" > "$rt/overnight_completion_stdout.log" 2> "$rt/overnight_completion_stderr.log"
post=$?
printf '%s\n' "$post" > "$rt/overnight_completion_exit_code.txt"
date -Is > "$rt/overnight_completion_end_time.txt"
stamp "POSTRUN_FINISHED exit_code=$post"
exit "$post"
