#!/usr/bin/env bash
set -u
R=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_040000__phase0d_4d_r3g_epsilon_bridge
O="$R/e0_gpu2"
mkdir -p "$O"
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/start_time_utc.txt"
W=/workspace
WAN="$W/Wan-Move"
OVER="$W/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay"
export CUDA_VISIBLE_DEVICES=2 DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy"
export DT_TRACK_DEPTH_PATH="$W/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy"
export PYTHONPATH="$OVER:$WAN:${PYTHONPATH:-}"
cd "$WAN"
"$W/tools/miniforge3/envs/wan-move/bin/python" "$OVER/generate.py" --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$WAN/Wan-Move-14B-480P" --image "$W/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png" --track "$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy" --track_visibility "$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy" --prompt 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.' --base_seed 0 --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --initial_epsilon "$W/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/R3_SHARED_EPSILON_58x104.npy" --save_file "$O/e0_epsilon_only_correct_v3d_seed000.mp4" > "$O/stdout.log" 2> "$O/stderr.log"
rc=$?
echo "$rc" > "$O/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/end_time_utc.txt"
[ "$rc" -eq 0 ] && touch "$O/completion.marker"
exit "$rc"
