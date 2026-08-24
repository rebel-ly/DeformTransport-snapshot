#!/usr/bin/env bash
set -u

GPU="${1:?usage: run.sh GPU_ID}"
DT="/mnt/sdbd/home/liuyu_qyh/DeformTransport"
WAN="/mnt/sdbd/home/liuyu_qyh/Wan-Move"
ENV="/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/envs/wan-move"
PY="$ENV/bin/python"
RUN="$DT/server_runs/wan_move_formal/20260810_072309__tree_correct_vs_identity_shuffled_seed0/correct"
IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/resized_input_image.png"
TRACK="$DT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_tracks_correct.npy"
VISIBILITY="$DT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_visibility_correct.npy"
PROMPT_FILE="$DT/server_runs/wan_move_formal/20260810_072309__tree_correct_vs_identity_shuffled_seed0/prompt.txt"
OUTPUT="$RUN/tree_formal_correct_seed0.mp4"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME="$ENV"
export CUDA_PATH="$ENV"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

date -Iseconds > "$RUN/start_time.txt"
printf '%s\n' "$BASHPID" > "$RUN/pid.txt"
PROMPT="$(<"$PROMPT_FILE")"
cd "$WAN" || exit 1

"$PY" generate.py \
  --task wan-move-i2v \
  --size '480*832' \
  --frame_num 81 \
  --ckpt_dir "$WAN/Wan-Move-14B-480P" \
  --image "$IMAGE" \
  --track "$TRACK" \
  --track_visibility "$VISIBILITY" \
  --prompt "$PROMPT" \
  --base_seed 0 \
  --sample_steps 40 \
  --sample_shift 3.0 \
  --t5_cpu \
  --offload_model True \
  --dtype bf16 \
  --save_file "$OUTPUT"
EC=$?

date -Iseconds > "$RUN/end_time.txt"
printf '%s\n' "$EC" > "$RUN/exit_code.txt"
if [ -f "$OUTPUT" ]; then
  sha256sum "$OUTPUT" > "$RUN/output_sha256.txt"
fi
exit "$EC"
