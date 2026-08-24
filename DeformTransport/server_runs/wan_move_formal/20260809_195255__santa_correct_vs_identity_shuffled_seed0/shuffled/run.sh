#!/usr/bin/env bash
set -u

PY="/workspace/tools/miniforge3/envs/wan-move/bin/python"
WANENV="/workspace/tools/miniforge3/envs/wan-move"
RUN="/workspace/DeformTransport/server_runs/wan_move_formal/20260809_195255__santa_correct_vs_identity_shuffled_seed0/shuffled"

export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME="/workspace/tools/miniforge3/envs/wan-move"
export CUDA_PATH="/workspace/tools/miniforge3/envs/wan-move"
export PATH="/workspace/tools/miniforge3/envs/wan-move/bin:$PATH"
export LD_LIBRARY_PATH="/workspace/tools/miniforge3/envs/wan-move/targets/x86_64-linux/lib:/workspace/tools/miniforge3/envs/wan-move/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

cd /workspace/Wan-Move || exit 1

PROMPT="$(cat "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt")"

date -Iseconds > "$RUN/start_time.txt"

"$PY" generate.py   --task wan-move-i2v   --size '480*832'   --frame_num 81   --ckpt_dir "/workspace/Wan-Move/Wan-Move-14B-480P"   --image "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"   --track "/workspace/DeformTransport/server_runs/wan_move_bridge/20260809_195037__santa_identity_shuffled_binding_seed0/santa_material_tracks_identity_shuffled.npy"   --track_visibility "/workspace/DeformTransport/server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_visibility_correct.npy"   --prompt "$PROMPT"   --base_seed 0   --sample_steps 40   --sample_shift 3.0   --t5_cpu   --offload_model True   --dtype bf16   --save_file "$RUN/santa_formal_identity_shuffled_seed0.mp4"   > "$RUN/stdout.log"   2> "$RUN/stderr.log"

EC=$?

echo "$EC" > "$RUN/exit_code.txt"
date -Iseconds > "$RUN/end_time.txt"

if [ -f "$RUN/santa_formal_identity_shuffled_seed0.mp4" ]; then
    sha256sum "$RUN/santa_formal_identity_shuffled_seed0.mp4"       > "$RUN/output_sha256.txt"
fi

exit "$EC"
