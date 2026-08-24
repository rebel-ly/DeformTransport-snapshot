#!/usr/bin/env bash
# Container-only formal binding; inert until invoked.  --dry-run never calls generate.py.
set -euo pipefail

FORMAL_WANMOVE_PYTHON="/workspace/tools/miniforge3/envs/wan-move/bin/python"
WAN="/workspace/Wan-Move"
TRACK="/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy"
VISIBILITY="/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy"
IDS="/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy"
DEPTH="/workspace/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy"
IMAGE="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"
PROMPT_FILE="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt"
CHECKPOINT="/workspace/Wan-Move/Wan-Move-14B-480P"

dry=0
if [[ "${1:-}" == "--dry-run" ]]; then dry=1; shift; fi
seed="${1:?usage: $0 [--dry-run] SEED OUTPUT_DIR}"
out="${2:?usage: $0 [--dry-run] SEED OUTPUT_DIR}"
case "$seed" in 0|1|2|3|4) ;; *) echo "SEED must be one of 0 1 2 3 4" >&2; exit 64 ;; esac
export DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$IDS"
export DT_TRACK_DEPTH_PATH="$DEPTH"
export PYTHONPATH="$WAN:${PYTHONPATH:-}"
cmd=("$FORMAL_WANMOVE_PYTHON" generate.py --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$CHECKPOINT" --image "$IMAGE" --track "$TRACK" --track_visibility "$VISIBILITY" --prompt "$(cat "$PROMPT_FILE")" --base_seed "$seed" --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --save_file "$out/santa_correct_v3d_seed$(printf '%03d' "$seed").mp4")
if [[ "$dry" == 1 ]]; then
  printf 'ACTUAL_PYTHON=%s\n' "$FORMAL_WANMOVE_PYTHON"
  printf 'DT_TRANSPORT_VARIANT=%s\nDT_TRACK_IDS_PATH=%s\nDT_TRACK_DEPTH_PATH=%s\n' "$DT_TRANSPORT_VARIANT" "$DT_TRACK_IDS_PATH" "$DT_TRACK_DEPTH_PATH"
  printf 'OUTPUT_DIR=%s\nCOMMAND:' "$out"; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi
mkdir -p "$out"
cd "$WAN"
exec "${cmd[@]}"
