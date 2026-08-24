#!/usr/bin/env bash
# Frozen Phase0D corrected-v2 N=1257 V3D runner.  It is inert until invoked.
set -euo pipefail
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
SEED="${1:?usage: $0 [--dry-run] SEED OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: $0 [--dry-run] SEED OUTPUT_DIR}"
case "$SEED" in 0|1|2|3|4) ;; *) echo "SEED must be one of 0 1 2 3 4" >&2; exit 64;; esac
WAN="/mnt/sdbd/home/liuyu_qyh/Wan-Move"
PY="/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/envs/wan-move/bin/python"
TRACK="/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy"
VISIBILITY="/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy"
IDS="/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy"
DEPTH="/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy"
IMAGE="/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"
PROMPT_FILE="/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt"
CHECKPOINT="/mnt/sdbd/home/liuyu_qyh/Wan-Move/Wan-Move-14B-480P"
export DT_TRANSPORT_VARIANT="v3d"
export DT_TRACK_IDS_PATH="$IDS"
export DT_TRACK_DEPTH_PATH="$DEPTH"
export PYTHONPATH="$WAN:${PYTHONPATH:-}"
CMD=("$PY" generate.py --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$CHECKPOINT" --image "$IMAGE" --track "$TRACK" --track_visibility "$VISIBILITY" --prompt "$(cat "$PROMPT_FILE")" --base_seed "$SEED" --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --save_file "$OUTPUT_DIR/santa_correct_v3d_seed$(printf '%03d' "$SEED").mp4")
if [[ "$DRY" == 1 ]]; then
  printf 'DT_TRANSPORT_VARIANT=%s\nDT_TRACK_IDS_PATH=%s\nDT_TRACK_DEPTH_PATH=%s\n' "$DT_TRANSPORT_VARIANT" "$DT_TRACK_IDS_PATH" "$DT_TRACK_DEPTH_PATH"
  printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
  printf 'COMMAND:'; printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
cd "$WAN"
exec "${CMD[@]}"
