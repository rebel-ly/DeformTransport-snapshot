#!/usr/bin/env bash
set -u
R=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution
O="$R/cs_seed1_gpu0"
W=/workspace
WAN="$W/Wan-Move"
OVER="$W/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay"
R3="$W/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation"
SH="$W/DeformTransport_EvidenceV4_1/reports/phase0b/causal_contract/20260812_134250__santa_corrected_v2_identity_shuffle_seed0"
CORRECT="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline"
DEPTH="$W/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy"
EPS="$W/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_100000__phase0d_4d_r3m_seed1_wave1/EPSILON_SEED1_58x104.npy"
mkdir -p "$O"
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/start_time_utc.txt"
printf '%s\n' "$$" > "$O/wrapper_pid.txt"
export CUDA_VISIBLE_DEVICES=0
export DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$CORRECT/santa_material_point_ids.npy"
export DT_TRACK_DEPTH_PATH="$DEPTH"
export PYTHONPATH="$OVER:$WAN:${PYTHONPATH:-}"
cd "$WAN"
"$W/tools/miniforge3/envs/wan-move/bin/python" "$OVER/generate.py" --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$WAN/Wan-Move-14B-480P" --image "$W/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png" --track "$SH/santa_material_tracks_identity_shuffled_seed0.npy" --track_visibility "$SH/santa_material_visibility_identity_shuffled_seed0.npy" --prompt 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.' --base_seed 1 --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --preview_latent "$R3/WAN_FORMAL_PREVIEW_LATENT_58x104.npy" --initial_epsilon "$EPS" --start_index 15 --save_file "$O/cs_preview_shuffled_v3d_seed001.mp4" > "$O/stdout.log" 2> "$O/stderr.log" &
child=$!
printf '%s\n' "$child" > "$O/python_pid.txt"
wait "$child"
rc=$?
printf '%s\n' "$rc" > "$O/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/end_time_utc.txt"
[ "$rc" -eq 0 ] && touch "$O/completion.marker"
exit "$rc"
