#!/usr/bin/env bash
set -u

GPU="${1:?usage: bash run.sh <gpu_id>}"
PY="/workspace/tools/miniforge3/envs/wan-move/bin/python"
WANENV="/workspace/tools/miniforge3/envs/wan-move"
WAN_ROOT="/workspace/Wan-Move"
DT_ROOT="/workspace/DeformTransport"
RUN="$DT_ROOT/server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0/correct"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME="$WANENV"
export CUDA_PATH="$WANENV"
export PATH="$WANENV/bin:$PATH"
export LD_LIBRARY_PATH="$WANENV/targets/x86_64-linux/lib:$WANENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

printf '%s\n' "$BASHPID" > "$RUN/pid.txt"
date -Iseconds > "$RUN/start_time.txt"
cd "$WAN_ROOT" || exit 1
PROMPT="$(cat "$DT_ROOT/server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0/prompt.txt")"

"$PY" generate.py \
  --task wan-move-i2v \
  --size '480*832' \
  --frame_num 81 \
  --ckpt_dir "$WAN_ROOT/Wan-Move-14B-480P" \
  --image "$DT_ROOT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/resized_input_image.png" \
  --track "$DT_ROOT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_tracks_correct.npy" \
  --track_visibility "$DT_ROOT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_visibility_correct.npy" \
  --prompt "$PROMPT" \
  --base_seed 0 \
  --sample_steps 40 \
  --sample_shift 3.0 \
  --t5_cpu \
  --offload_model True \
  --dtype bf16 \
  --save_file "$RUN/tree_formal_correct_seed0.mp4" \
  > "$RUN/stdout.log" 2> "$RUN/stderr.log"
EC=$?

printf '%s\n' "$EC" > "$RUN/exit_code.txt"
date -Iseconds > "$RUN/end_time.txt"
if [ -f "$RUN/tree_formal_correct_seed0.mp4" ]; then
  sha256sum "$RUN/tree_formal_correct_seed0.mp4" > "$RUN/output_sha256.txt"
fi
exit "$EC"
