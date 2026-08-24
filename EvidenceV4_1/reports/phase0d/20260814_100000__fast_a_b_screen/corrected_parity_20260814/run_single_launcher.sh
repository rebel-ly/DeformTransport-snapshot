#!/usr/bin/env bash
# Authoritative corrected A2/B2 parity launcher.  SOURCE_MODE is the sole arm switch.
set -euo pipefail
mode="${1:?SOURCE_MODE original|overlay}"; out="${2:?output directory}"
case "$mode" in original|overlay) ;; *) echo 'SOURCE_MODE must be original or overlay' >&2; exit 64;; esac
W=/workspace
WAN="$W/Wan-Move"
OVERLAY="$W/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay"
PY="$W/tools/miniforge3/envs/wan-move/bin/python"
TRACK="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy"
VIS="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy"
IDS="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy"
DEPTH="$W/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy"
IMAGE="$W/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"
PROMPT='Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.'
if [ "$mode" = original ]; then SRC="$WAN"; else SRC="$OVERLAY"; fi
export DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$IDS"
export DT_TRACK_DEPTH_PATH="$DEPTH"
if [ "$mode" = original ]; then
  export PYTHONPATH="$WAN:${PYTHONPATH:-}"
else
  export PYTHONPATH="$OVERLAY:$WAN:${PYTHONPATH:-}"
fi
mkdir -p "$out"
cd "$WAN"
exec "$PY" "$SRC/generate.py" --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$WAN/Wan-Move-14B-480P" --image "$IMAGE" --track "$TRACK" --track_visibility "$VIS" --prompt "$PROMPT" --base_seed 0 --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --save_file "$out/santa_correct_v3d_seed000.mp4"
