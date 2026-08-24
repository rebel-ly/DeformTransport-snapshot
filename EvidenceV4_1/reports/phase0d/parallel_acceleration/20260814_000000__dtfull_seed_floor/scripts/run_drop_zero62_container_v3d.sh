#!/usr/bin/env bash
# Frozen DROP-ZERO62 runner: only the prebuilt 1195-carrier input differs from DT-FULL.
set -euo pipefail
out="${1:?output directory required}"
W=/workspace
A="$W/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/drop_zero62/20260814_081746"
export DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$A/drop_zero62_ids.npy"
export DT_TRACK_DEPTH_PATH="$A/drop_zero62_depth.npy"
export PYTHONPATH="$W/Wan-Move:${PYTHONPATH:-}"
mkdir -p "$out"
cd "$W/Wan-Move"
exec "$W/tools/miniforge3/envs/wan-move/bin/python" generate.py --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$W/Wan-Move/Wan-Move-14B-480P" --image "$W/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png" --track "$A/drop_zero62_tracks.npy" --track_visibility "$A/drop_zero62_visibility.npy" --prompt 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.' --base_seed 0 --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --save_file "$out/santa_correct_v3d_seed000.mp4"
