#!/usr/bin/env bash
# External wrapper only: performs no source changes and uses formal overlay verbatim.
set -euo pipefail
OUT="${1:?output directory}"; GPU="${2:?host GPU ordinal}"
W=/workspace; WAN="$W/Wan-Move"; OVERLAY="$W/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay"
R="$W/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_000000__phase0d_4d_r2_contract_correction"
OLD="$W/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_200000__phase0d_4d_recovery"
export CUDA_VISIBLE_DEVICES="$GPU"
export DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy"
export DT_TRACK_DEPTH_PATH="$W/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy"
export PYTHONPATH="$OVERLAY:$WAN:${PYTHONPATH:-}"
mkdir -p "$OUT"
cd "$WAN"
exec "$W/tools/miniforge3/envs/wan-move/bin/python" "$OVERLAY/generate.py" --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$WAN/Wan-Move-14B-480P" --image "$W/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png" --track "$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy" --track_visibility "$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy" --prompt 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.' --base_seed 0 --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --preview_latent "$W/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/preview_wan_vae_latent_e1_832x480.npy" --initial_epsilon "$OLD/FINAL_C_SHARED_EPSILON.npy" --start_index 0 --save_file "$OUT/begin0_correct_v3d_seed000.mp4"
