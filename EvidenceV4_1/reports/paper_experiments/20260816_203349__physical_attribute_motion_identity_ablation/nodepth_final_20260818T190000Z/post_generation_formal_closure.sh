#!/usr/bin/env bash
set -u
P=/workspace/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/nodepth_final_20260818T190000Z
O=${1:?successful_attempt_dir}
G=${2:?physical_gpu}
W=$P/gpu_opportunity_watcher
V=$O/nodepth_formal_correct_v3d_seed000.mp4
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration -of json "$V" > "$O/video_validation_ffprobe.json"
sha256sum "$V" > "$P/VIDEO_SHA256SUMS.txt"
printf '{"state":"EVALUATING","physical_gpu":%s,"nodepth_video_path":"%s"}\n' "$G" "$V" > "$W/FINAL_STATUS.json"
printf '%s\n' 'FORMAL_EVALUATOR_PENDING_EXECUTION' > "$W/companion_closure_state.json"
