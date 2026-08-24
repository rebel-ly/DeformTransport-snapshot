#!/usr/bin/env bash
# Container-path equivalent of the frozen canonical v3d runner; seed/output only vary.
set -euo pipefail
seed="${1:?seed}"; out="${2:?output}"
case "$seed" in 0|1|2|3|4) ;; *) exit 64;; esac
W=/workspace
export DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy"
export DT_TRACK_DEPTH_PATH="$W/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy"
export PYTHONPATH="$W/Wan-Move:${PYTHONPATH:-}"
mkdir -p "$out"; cd "$W/Wan-Move"
exec "$W/tools/miniforge3/envs/wan-move/bin/python" generate.py --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir "$W/Wan-Move/Wan-Move-14B-480P" --image "$W/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png" --track "$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy" --track_visibility "$W/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy" --prompt 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.' --base_seed "$seed" --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --save_file "$out/santa_correct_v3d_seed$(printf '%03d' "$seed").mp4"
