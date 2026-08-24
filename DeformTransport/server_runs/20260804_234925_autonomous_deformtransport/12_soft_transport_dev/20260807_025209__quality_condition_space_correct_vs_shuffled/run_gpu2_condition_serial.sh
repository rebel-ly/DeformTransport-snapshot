#!/usr/bin/env bash

set -uo pipefail

cd /workspace/DeformTransport

PY="/workspace/tools/miniforge3/envs/realwonder-gen/bin/python"
GPU_ID=2

ALIGNED_FINAL_SIM="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410"
CHECKPOINT="/workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"
QUALITY_ARTIFACT="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_232607__full_generation_compatible_candidates/quality_ramp4_full_generation.pt"

COND_ROOT="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_025209__quality_condition_space_correct_vs_shuffled"
CORRECT_DIR="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_025209__quality_condition_space_correct_vs_shuffled/quality_condition_correct"
SHUFFLED_DIR="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_025209__quality_condition_space_correct_vs_shuffled/quality_condition_shuffled"

CORRECT_VIDEO="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_025209__quality_condition_space_correct_vs_shuffled/quality_condition_correct/aligned_santa_quality_condition_correct_seed0.mp4"
SHUFFLED_VIDEO="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_025209__quality_condition_space_correct_vs_shuffled/quality_condition_shuffled/aligned_santa_quality_condition_shuffled_seed0.mp4"

run_case() {
    local name="$1"
    local mode="$2"
    local output_dir="$3"
    local output_video="$4"

    echo "running_$name"       > "$COND_ROOT/status.txt"

    date --iso-8601=seconds       > "$output_dir/start_time.txt"

    CUDA_VISIBLE_DEVICES="$GPU_ID"     PYTHONPATH=/workspace/DeformTransport     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     timeout --signal=TERM --kill-after=60s 7200s     "$PY" -u infer_sim.py       --checkpoint_path "$CHECKPOINT"       --sim_data_path "$ALIGNED_FINAL_SIM"       --output_path "$output_video"       --seed 0       --eval_degradation 0.5       --local_attn_size 21       --transport_latent_path "$QUALITY_ARTIFACT"       --transport_mode "$mode"       --transport_injection_mode condition_residual       --transport_injection_scale 1.0       --transport_injection_step 0       > "$output_dir/stdout.log"       2> "$output_dir/stderr.log"

    local rc=$?

    echo "$rc"       > "$output_dir/exit_code.txt"

    date --iso-8601=seconds       > "$output_dir/end_time.txt"

    if [ "$rc" -ne 0 ]; then
        echo "failed_${name}_exit_$rc"           > "$COND_ROOT/status.txt"

        return "$rc"
    fi

    if [ ! -s "$output_video" ]; then
        echo "failed_${name}_missing_video"           > "$COND_ROOT/status.txt"

        return 91
    fi

    local condition_count
    local inter_step_count

    condition_count=$(
      grep -c         "Applied condition-space transport"         "$output_dir/stdout.log"       || true
    )

    inter_step_count=$(
      grep -c         "Applying inter-step transport residual"         "$output_dir/stdout.log"       || true
    )

    echo "$condition_count"       > "$output_dir/condition_application_count.txt"

    echo "$inter_step_count"       > "$output_dir/inter_step_application_count.txt"

    if [ "$condition_count" -ne 1 ]; then
        echo "failed_${name}_condition_count_$condition_count"           > "$COND_ROOT/status.txt"

        return 92
    fi

    if [ "$inter_step_count" -ne 0 ]; then
        echo "failed_${name}_unexpected_inter_step_$inter_step_count"           > "$COND_ROOT/status.txt"

        return 93
    fi

    echo "$name success"
}

echo "running_quality_condition_correct"   > "$COND_ROOT/status.txt"

run_case   "quality_condition_correct"   "correct"   "$CORRECT_DIR"   "$CORRECT_VIDEO"

CORRECT_RC=$?

echo "$CORRECT_RC"   > "$COND_ROOT/correct_exit_code.txt"

if [ "$CORRECT_RC" -ne 0 ]; then
    exit "$CORRECT_RC"
fi

sleep 30

run_case   "quality_condition_shuffled"   "shuffled"   "$SHUFFLED_DIR"   "$SHUFFLED_VIDEO"

SHUFFLED_RC=$?

echo "$SHUFFLED_RC"   > "$COND_ROOT/shuffled_exit_code.txt"

if [ "$SHUFFLED_RC" -ne 0 ]; then
    exit "$SHUFFLED_RC"
fi

echo "success"   > "$COND_ROOT/status.txt"

date --iso-8601=seconds   > "$COND_ROOT/end_time.txt"

echo "GPU 2 condition-space serial run succeeded"
echo "CORRECT_VIDEO=$CORRECT_VIDEO"
echo "SHUFFLED_VIDEO=$SHUFFLED_VIDEO"
