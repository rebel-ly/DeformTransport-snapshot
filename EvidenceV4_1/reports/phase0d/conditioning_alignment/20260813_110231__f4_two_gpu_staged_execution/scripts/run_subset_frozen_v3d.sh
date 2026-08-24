#!/usr/bin/env bash
# F4 execution adapter: identical frozen generate.py arguments, parameterized only by F3 sidecar rows.
set -euo pipefail
seed="${1:?seed}"; out="${2:?out}"; track="${3:?track}"; vis="${4:?visibility}"; ids="${5:?ids}"; depth="${6:?depth}"
case "$seed" in 0) ;; *) echo 'only frozen seed 0' >&2; exit 64;; esac
WAN=/mnt/sdbd/home/liuyu_qyh/Wan-Move
PY=/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/envs/wan-move/bin/python
IMAGE=/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png
PROMPT_FILE=/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt
CKPT=/mnt/sdbd/home/liuyu_qyh/Wan-Move/Wan-Move-14B-480P
export DT_TRANSPORT_VARIANT=v3d DT_TRACK_IDS_PATH="$ids" DT_TRACK_DEPTH_PATH="$depth" PYTHONPATH="$WAN:${PYTHONPATH:-}"
cmd=("$PY" generate.py --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$CKPT" --image "$IMAGE" --track "$track" --track_visibility "$vis" --prompt "$(cat "$PROMPT_FILE")" --base_seed "$seed" --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --save_file "$out/santa_correct_v3d_seed000.mp4")
if [[ "${F4_DRY_RUN:-0}" == 1 ]]; then printf 'COMMAND:'; printf ' %q' "${cmd[@]}"; printf '\n'; exit 0; fi
mkdir -p "$out"; cd "$WAN"; exec "${cmd[@]}"
