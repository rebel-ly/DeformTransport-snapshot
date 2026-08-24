#!/usr/bin/env bash
set -euo pipefail
ARM="${1:?arm required}"
GPU="${2:?gpu required}"
DRY_RUN="${DRY_RUN:-0}"
R=/workspace/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation
W=/workspace
WAN="$W/Wan-Move"
case "$ARM" in
  SC) OVER="$W/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay"; TRACK="$R/conditions/SC_static_correct_tracks_seed0.npy"; VIS="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy"; NOVIS=0; NODEPTH=0 ;;
  SS) OVER="$W/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay"; TRACK="$R/conditions/SS_static_shuffled_tracks_seed0.npy"; VIS="$W/DeformTransport_EvidenceV4_1/reports/phase0b/causal_contract/20260812_134250__santa_corrected_v2_identity_shuffle_seed0/santa_material_visibility_identity_shuffled_seed0.npy"; NOVIS=0; NODEPTH=0 ;;
  C2-NOVIS) OVER="$R/overlay_a1_frozen_r3"; TRACK="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy"; VIS="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy"; NOVIS=1; NODEPTH=0 ;;
  C2-NODEPTH) OVER="$R/overlay_a1_frozen_r3"; TRACK="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy"; VIS="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy"; NOVIS=0; NODEPTH=1 ;;
  *) exit 64 ;;
esac
O="$R/outputs/$ARM"; mkdir -p "$O"
export CUDA_VISIBLE_DEVICES="$GPU" DT_TRANSPORT_VARIANT=v3d DT_TRACK_IDS_PATH="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy" DT_TRACK_DEPTH_PATH="$W/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy" DT_A1_DISABLE_VISIBILITY="$NOVIS" DT_A1_DISABLE_DEPTH_ARBITRATION="$NODEPTH" PYTHONPATH="$OVER:$WAN:${PYTHONPATH:-}"
if [ "$DRY_RUN" = 1 ]; then
  printf 'ARM=%s
OVER=%s
TRACK=%s
VIS=%s
NOVIS=%s
NODEPTH=%s
' "$ARM" "$OVER" "$TRACK" "$VIS" "$NOVIS" "$NODEPTH"
  test -r "$TRACK"; test -r "$VIS"; test -r "$DT_TRACK_IDS_PATH"; test -r "$DT_TRACK_DEPTH_PATH"; exit 0
fi
cd "$WAN"
exec "$W/tools/miniforge3/envs/wan-move/bin/python" "$OVER/generate.py" --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$WAN/Wan-Move-14B-480P" --image "$W/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png" --track "$TRACK" --track_visibility "$VIS" --prompt 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.' --base_seed 0 --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --preview_latent "$W/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/WAN_FORMAL_PREVIEW_LATENT_58x104.npy" --initial_epsilon "$W/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/R3_SHARED_EPSILON_58x104.npy" --start_index 15 --save_file "$O/$ARM"_seed000.mp4
