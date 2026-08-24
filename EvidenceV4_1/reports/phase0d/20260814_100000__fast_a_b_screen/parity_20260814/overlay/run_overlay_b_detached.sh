#!/usr/bin/env bash
set -euo pipefail
OUT=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/parity_20260814/overlay
OVERLAY=/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
mkdir -p "$OUT"
date -Is > "$OUT/start_time.txt"
printf '%s\n' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" > "$OUT/environment.txt"
PYTHONPATH="$OVERLAY:/workspace/Wan-Move" "$PY" -c 'import wan,wan.wan_move; from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler; print(wan.__file__); print(wan.wan_move.__file__); print(FlowUniPCMultistepScheduler.__module__)' > "$OUT/source_provenance.txt" 2>&1
cd "$OVERLAY"
set +e
PYTHONPATH="$OVERLAY:/workspace/Wan-Move" "$PY" generate.py --task wan-move-i2v --size '480*832' --frame_num 81 --ckpt_dir /workspace/Wan-Move/Wan-Move-14B-480P --image /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png --track /workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy --track_visibility /workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy --prompt 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.' --base_seed 0 --sample_steps 40 --sample_shift 3.0 --t5_cpu --offload_model True --dtype bf16 --save_file "$OUT/santa_correct_v3d_seed000.mp4" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" > "$OUT/exit_code.txt"
date -Is > "$OUT/end_time.txt"
if [ "$rc" -eq 0 ]; then printf 'COMPLETE\n' > "$OUT/completion.marker"; else printf 'FAILED\n' > "$OUT/completion.marker"; fi
exit "$rc"
