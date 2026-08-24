#!/usr/bin/env bash
# Authorized F4 scheduler. It never inspects generated content or metrics.
set -euo pipefail
base_dir=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution
f3_dir=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze
runner="$base_dir/scripts/run_subset_frozen_v3d.sh"
mkdir -p "$base_dir/outputs/wm0_seed0" "$base_dir/outputs/frag_prune_seed0" "$base_dir/outputs/grid100_center_seed0"
date -Is > "$base_dir/runtime/wm0_start_time.txt"
CUDA_VISIBLE_DEVICES=1 bash "$runner" 0 "$base_dir/outputs/wm0_seed0" "$f3_dir/artifacts/wm0_tracks.npy" "$f3_dir/artifacts/wm0_visibility.npy" "$f3_dir/artifacts/wm0_ids.npy" "$f3_dir/artifacts/wm0_depth.npy" > "$base_dir/runtime/wm0_stdout.log" 2> "$base_dir/runtime/wm0_stderr.log" &
wm_pid=$!
date -Is > "$base_dir/runtime/frag_start_time.txt"
CUDA_VISIBLE_DEVICES=2 bash "$runner" 0 "$base_dir/outputs/frag_prune_seed0" "$f3_dir/artifacts/frag_prune_tracks.npy" "$f3_dir/artifacts/frag_prune_visibility.npy" "$f3_dir/artifacts/frag_prune_ids.npy" "$f3_dir/artifacts/frag_prune_depth.npy" > "$base_dir/runtime/frag_stdout.log" 2> "$base_dir/runtime/frag_stderr.log" &
frag_pid=$!
printf '%s\n' "$wm_pid" > "$base_dir/runtime/wm0_pid.txt"
printf '%s\n' "$frag_pid" > "$base_dir/runtime/frag_pid.txt"
first_arm=UNRESOLVED; first_gpu=UNRESOLVED
if wait "$wm_pid"; then wm_exit=0; else wm_exit=$?; fi
date -Is > "$base_dir/runtime/wm0_end_time.txt"; printf '%s\n' "$wm_exit" > "$base_dir/runtime/wm0_exit_code.txt"
first_arm=WM-0; first_gpu=GPU1
date -Is > "$base_dir/runtime/grid100_start_time.txt"
CUDA_VISIBLE_DEVICES=1 bash "$runner" 0 "$base_dir/outputs/grid100_center_seed0" "$f3_dir/artifacts/grid100_center_tracks.npy" "$f3_dir/artifacts/grid100_center_visibility.npy" "$f3_dir/artifacts/grid100_center_ids.npy" "$f3_dir/artifacts/grid100_center_depth.npy" > "$base_dir/runtime/grid100_stdout.log" 2> "$base_dir/runtime/grid100_stderr.log" &
grid_pid=$!
printf '%s\n' "$grid_pid" > "$base_dir/runtime/grid100_pid.txt"
if wait "$frag_pid"; then frag_exit=0; else frag_exit=$?; fi
date -Is > "$base_dir/runtime/frag_end_time.txt"; printf '%s\n' "$frag_exit" > "$base_dir/runtime/frag_exit_code.txt"
if wait "$grid_pid"; then grid_exit=0; else grid_exit=$?; fi
date -Is > "$base_dir/runtime/grid100_end_time.txt"; printf '%s\n' "$grid_exit" > "$base_dir/runtime/grid100_exit_code.txt"
printf '%s\n' "$first_arm" > "$base_dir/runtime/first_finished_arm.txt"
printf '%s\n' "$first_gpu" > "$base_dir/runtime/first_free_gpu.txt"
