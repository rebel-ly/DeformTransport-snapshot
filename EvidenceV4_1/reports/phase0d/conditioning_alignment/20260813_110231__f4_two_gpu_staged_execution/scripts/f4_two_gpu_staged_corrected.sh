#!/usr/bin/env bash
# Completion-only two-GPU scheduler; no video/metric inspection.
set -euo pipefail
b=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution
f=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze
r="$b/scripts/run_subset_frozen_v3d.sh"; mkdir -p "$b/outputs"/{wm0_seed0,frag_prune_seed0,grid100_center_seed0}
date -Is > "$b/runtime/wm0_start_time.txt"; CUDA_VISIBLE_DEVICES=1 bash "$r" 0 "$b/outputs/wm0_seed0" "$f/artifacts/wm0_tracks.npy" "$f/artifacts/wm0_visibility.npy" "$f/artifacts/wm0_ids.npy" "$f/artifacts/wm0_depth.npy" > "$b/runtime/wm0_stdout.log" 2> "$b/runtime/wm0_stderr.log" & wp=$!
date -Is > "$b/runtime/frag_start_time.txt"; CUDA_VISIBLE_DEVICES=2 bash "$r" 0 "$b/outputs/frag_prune_seed0" "$f/artifacts/frag_prune_tracks.npy" "$f/artifacts/frag_prune_visibility.npy" "$f/artifacts/frag_prune_ids.npy" "$f/artifacts/frag_prune_depth.npy" > "$b/runtime/frag_stdout.log" 2> "$b/runtime/frag_stderr.log" & fp=$!
printf '%s\n' "$wp" > "$b/runtime/wm0_pid.txt"; printf '%s\n' "$fp" > "$b/runtime/frag_pid.txt"
if wait -n -p done "$wp" "$fp"; then code=0; else code=$?; fi
if [[ "$done" == "$wp" ]]; then arm=WM-0; gpu=GPU1; we="$code"; date -Is > "$b/runtime/wm0_end_time.txt"; printf '%s\n' "$we" > "$b/runtime/wm0_exit_code.txt"; cvd=1
else arm=DT-FRAG-PRUNE; gpu=GPU2; fe="$code"; date -Is > "$b/runtime/frag_end_time.txt"; printf '%s\n' "$fe" > "$b/runtime/frag_exit_code.txt"; cvd=2; fi
printf '%s\n' "$arm" > "$b/runtime/first_finished_arm.txt"; printf '%s\n' "$gpu" > "$b/runtime/first_free_gpu.txt"
date -Is > "$b/runtime/grid100_start_time.txt"; CUDA_VISIBLE_DEVICES="$cvd" bash "$r" 0 "$b/outputs/grid100_center_seed0" "$f/artifacts/grid100_center_tracks.npy" "$f/artifacts/grid100_center_visibility.npy" "$f/artifacts/grid100_center_ids.npy" "$f/artifacts/grid100_center_depth.npy" > "$b/runtime/grid100_stdout.log" 2> "$b/runtime/grid100_stderr.log" & gp=$!; printf '%s\n' "$gp" > "$b/runtime/grid100_pid.txt"
if [[ "$arm" == WM-0 ]]; then if wait "$fp"; then fe=0; else fe=$?; fi; date -Is > "$b/runtime/frag_end_time.txt"; printf '%s\n' "$fe" > "$b/runtime/frag_exit_code.txt"; else if wait "$wp"; then we=0; else we=$?; fi; date -Is > "$b/runtime/wm0_end_time.txt"; printf '%s\n' "$we" > "$b/runtime/wm0_exit_code.txt"; fi
if wait "$gp"; then ge=0; else ge=$?; fi; date -Is > "$b/runtime/grid100_end_time.txt"; printf '%s\n' "$ge" > "$b/runtime/grid100_exit_code.txt"
