#!/usr/bin/env bash

set -uo pipefail

cd /workspace/DeformTransport

PY="/workspace/tools/miniforge3/envs/realwonder-gen/bin/python"

RUN_ROOT="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport"

ALIGNED_FINAL_SIM="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410"

BALANCED_ARTIFACT="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_232607__full_generation_compatible_candidates/balanced_ramp4_full_generation.pt"

QUALITY_ARTIFACT="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_232607__full_generation_compatible_candidates/quality_ramp4_full_generation.pt"

CHECKPOINT="/workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"

GPU_ID="2"

SECOND_WAVE_ROOT="/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_004033__gpu2_serial_balanced_shuffled_then_quality_correct"

SHUFFLED_DIR="$SECOND_WAVE_ROOT/balanced_shuffled"

QUALITY_DIR="$SECOND_WAVE_ROOT/quality_correct"

SHUFFLED_VIDEO="$SHUFFLED_DIR/aligned_santa_balanced_ramp4_shuffled_seed0.mp4"

QUALITY_VIDEO="$QUALITY_DIR/aligned_santa_quality_ramp4_correct_seed0.mp4"

mkdir -p "$SHUFFLED_DIR" "$QUALITY_DIR"

echo "启动时间=$(date --iso-8601=seconds)"
echo "GPU_ID=$GPU_ID"
echo "SECOND_WAVE_ROOT=$SECOND_WAVE_ROOT"

check_gpu_free() {
    local used

    used=$(
        nvidia-smi           -i "$GPU_ID"           --query-gpu=memory.used           --format=csv,noheader,nounits         | tr -d ' '
    )

    echo "GPU_${GPU_ID}_USED_MIB=$used"

    if [ "$used" -gt 1024 ]; then
        return 1
    fi

    return 0
}

wait_for_gpu() {
    local attempt

    for attempt in $(seq 1 30)
    do
        if check_gpu_free; then
            echo "GPU $GPU_ID 已空闲"
            return 0
        fi

        echo "GPU $GPU_ID 尚未释放，等待20秒：$attempt/30"
        sleep 20
    done

    echo "GPU $GPU_ID 等待10分钟后仍未空闲"
    return 1
}

run_inference() {
    local task_name="$1"
    local output_dir="$2"
    local output_video="$3"
    local artifact="$4"
    local mode="$5"

    mkdir -p "$output_dir"

    echo "running_$task_name"       > "$SECOND_WAVE_ROOT/status.txt"

    date --iso-8601=seconds       > "$output_dir/start_time.txt"

    nvidia-smi -i "$GPU_ID"       > "$output_dir/nvidia_smi_before.txt"

    free -h       > "$output_dir/memory_before.txt"

    sha256sum       "$CHECKPOINT"       "$ALIGNED_FINAL_SIM/config.yaml"       "$ALIGNED_FINAL_SIM/prompt.txt"       "$ALIGNED_FINAL_SIM/resized_input_image.png"       "$ALIGNED_FINAL_SIM/flows.npy"       "$ALIGNED_FINAL_SIM/noises.npy"       "$artifact"       infer_sim.py       deform_transport/pipeline_integration.py       vidgen/pipeline_sdedit.py       > "$output_dir/input_sha256.txt"

    printf '%q '       env       CUDA_VISIBLE_DEVICES="$GPU_ID"       PYTHONPATH=/workspace/DeformTransport       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True       "$PY" -u infer_sim.py       --checkpoint_path "$CHECKPOINT"       --sim_data_path "$ALIGNED_FINAL_SIM"       --output_path "$output_video"       --seed 0       --eval_degradation 0.5       --local_attn_size 21       --transport_latent_path "$artifact"       --transport_mode "$mode"       --transport_injection_mode inter_step_residual       --transport_injection_scale 1.0       --transport_injection_step 0       > "$output_dir/inference_command.txt"

    printf '\n'       >> "$output_dir/inference_command.txt"

    CUDA_VISIBLE_DEVICES="$GPU_ID"     PYTHONPATH=/workspace/DeformTransport     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     timeout --signal=TERM --kill-after=60s 7200s     "$PY" -u infer_sim.py       --checkpoint_path "$CHECKPOINT"       --sim_data_path "$ALIGNED_FINAL_SIM"       --output_path "$output_video"       --seed 0       --eval_degradation 0.5       --local_attn_size 21       --transport_latent_path "$artifact"       --transport_mode "$mode"       --transport_injection_mode inter_step_residual       --transport_injection_scale 1.0       --transport_injection_step 0       > "$output_dir/stdout.log"       2> "$output_dir/stderr.log"

    local rc=$?

    echo "$rc"       > "$output_dir/exit_code.txt"

    date --iso-8601=seconds       > "$output_dir/end_time.txt"

    nvidia-smi -i "$GPU_ID"       > "$output_dir/nvidia_smi_after.txt"

    free -h       > "$output_dir/memory_after.txt"

    echo "$task_name exit=$rc"

    if [ "$rc" -ne 0 ]; then
        return "$rc"
    fi

    if [ ! -s "$output_video" ]; then
        echo "输出视频不存在或为空：$output_video"
        return 91
    fi

    local injection_count

    injection_count=$(
        grep -c           "Applying inter-step transport residual"           "$output_dir/stdout.log"         || true
    )

    echo "$injection_count"       > "$output_dir/residual_injection_count.txt"

    echo "$task_name residual_count=$injection_count"

    if [ "$injection_count" -ne 7 ]; then
        echo "Residual注入次数异常"
        return 92
    fi

    return 0
}

echo "waiting_gpu2"   > "$SECOND_WAVE_ROOT/status.txt"

if ! wait_for_gpu; then
    echo "failed_gpu_not_free"       > "$SECOND_WAVE_ROOT/status.txt"

    exit 75
fi

echo "===== 开始 Balanced Shuffled ====="

run_inference   "balanced_shuffled"   "$SHUFFLED_DIR"   "$SHUFFLED_VIDEO"   "$BALANCED_ARTIFACT"   "shuffled"

SHUFFLED_RC=$?

echo "$SHUFFLED_RC"   > "$SECOND_WAVE_ROOT/balanced_shuffled_exit_code.txt"

if [ "$SHUFFLED_RC" -ne 0 ]; then
    echo "failed_balanced_shuffled_exit_$SHUFFLED_RC"       > "$SECOND_WAVE_ROOT/status.txt"

    exit "$SHUFFLED_RC"
fi

echo "Balanced Shuffled成功，等待GPU完全释放"

sleep 30

if ! wait_for_gpu; then
    echo "failed_gpu_not_released_after_shuffled"       > "$SECOND_WAVE_ROOT/status.txt"

    exit 76
fi

echo "===== 开始 Quality Correct ====="

run_inference   "quality_correct"   "$QUALITY_DIR"   "$QUALITY_VIDEO"   "$QUALITY_ARTIFACT"   "correct"

QUALITY_RC=$?

echo "$QUALITY_RC"   > "$SECOND_WAVE_ROOT/quality_correct_exit_code.txt"

if [ "$QUALITY_RC" -ne 0 ]; then
    echo "failed_quality_correct_exit_$QUALITY_RC"       > "$SECOND_WAVE_ROOT/status.txt"

    exit "$QUALITY_RC"
fi

echo "success"   > "$SECOND_WAVE_ROOT/status.txt"

date --iso-8601=seconds   > "$SECOND_WAVE_ROOT/end_time.txt"

echo "========================================"
echo "GPU 2串行任务全部成功"
echo "SHUFFLED_VIDEO=$SHUFFLED_VIDEO"
echo "QUALITY_VIDEO=$QUALITY_VIDEO"
echo "结束时间=$(date --iso-8601=seconds)"
echo "========================================"
