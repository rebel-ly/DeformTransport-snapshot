#!/usr/bin/env bash
# Runs inside existing deformtransport-dev; arguments are already /workspace paths.
set -euo pipefail
seed="${1:?seed}"; out="${2:?out}"; track="${3:?track}"; vis="${4:?visibility}"; ids="${5:?ids}"; depth="${6:?depth}"
case "$seed" in 0) ;; *) exit 64;; esac
WAN=/workspace/Wan-Move; PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
IMAGE=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png
PROMPT_FILE=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt
export DT_TRANSPORT_VARIANT=v3d DT_TRACK_IDS_PATH="$ids" DT_TRACK_DEPTH_PATH="$depth" PYTHONPATH="$WAN:${PYTHONPATH:-}"
mkdir -p "$out"; cd "$WAN"
exec "$PY" generate.py --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir /workspace/Wan-Move/Wan-Move-14B-480P --image "$IMAGE" --track "$track" --track_visibility "$vis" --prompt "$(cat "$PROMPT_FILE")" --base_seed 0 --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --save_file "$out/santa_correct_v3d_seed000.mp4"
