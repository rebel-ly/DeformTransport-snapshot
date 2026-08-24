#!/usr/bin/env bash
set -euo pipefail

root=/workspace/DeformTransport
run_root=$root/server_runs/20260804_234925_autonomous_deformtransport
job="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

final_sim=$run_root/04_smoke/OFFICIAL_SANTA_81F_ASSEMBLY_MANUAL_20260805_003555/final_sim
artifact=$run_root/04_smoke/OFFICIAL_SANTA_81F_WAN_VAE_MANUAL_20260805_013606/outputs/vae_latent_outputs.pt
checkpoint=$root/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt
python=/workspace/tools/miniforge3/envs/realwonder-gen/bin/python

GPU_ID="${GPU_ID:-2}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_HOME=$run_root/prepared_inputs/torch_cache

trap '
code=$?
echo "$code" > "$job/exit_code.txt"
date --iso-8601=seconds > "$job/end_time.txt"
' EXIT

cd "$root"

printf '%s\n' "$GPU_ID" > "$job/physical_gpu.txt"
date --iso-8601=seconds > "$job/start_time.txt"
git rev-parse HEAD > "$job/git_head.txt"
git status --short > "$job/git_status.txt"

required_inputs=(
    "$checkpoint"
    "$final_sim/config.yaml"
    "$final_sim/noises.npy"
    "$final_sim/flows.npy"
    "$final_sim/resized_input_image.png"
    "$final_sim/prompt.txt"
    "$artifact"
)

for path in "${required_inputs[@]}"; do
    if [ ! -f "$path" ]; then
        echo "缺少必要输入：$path" >&2
        exit 64
    fi
done

sha256sum "${required_inputs[@]}" \
    > "$job/all_inputs_sha256.txt"

run_variant() {
    local mode=$1
    local dir=$job/$mode
    local output=$dir/santa_official_${mode}_seed0.mp4

    mkdir -p "$dir"

    if [ -e "$output" ]; then
        echo "拒绝覆盖已有输出：$output" >&2
        return 73
    fi

    local args=(
        --checkpoint_path "$checkpoint"
        --sim_data_path "$final_sim"
        --output_path "$output"
        --seed 0
        --eval_degradation 0.5
        --local_attn_size 21
    )

    if [ "$mode" != "baseline" ]; then
        args+=(
            --transport_latent_path "$artifact"
            --transport_mode "$mode"
        )
    fi

    local free_mib
    local temp_c
    local mem_kib

    free_mib=$(
        nvidia-smi -i "$GPU_ID" \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
        | head -1 \
        | tr -d ' '
    )

    temp_c=$(
        nvidia-smi -i "$GPU_ID" \
            --query-gpu=temperature.gpu \
            --format=csv,noheader,nounits \
        | head -1 \
        | tr -d ' '
    )

    mem_kib=$(
        awk '/MemAvailable/{print $2}' /proc/meminfo
    )

    {
        echo "physical_gpu=$GPU_ID"
        echo "free_mib=$free_mib"
        echo "temperature_c=$temp_c"
        echo "mem_available_kib=$mem_kib"
    } > "$dir/resource_gate.txt"

    if [ "$free_mib" -lt 38000 ]; then
        echo "GPU可用显存不足：${free_mib} MiB" >&2
        return 75
    fi

    if [ "$temp_c" -ge 82 ]; then
        echo "GPU温度过高：${temp_c} C" >&2
        return 75
    fi

    if [ "$mem_kib" -lt 31457280 ]; then
        echo "系统可用内存不足：${mem_kib} KiB" >&2
        return 75
    fi

    printf '%q ' \
        "$python" -u infer_sim.py "${args[@]}" \
        > "$dir/inference_command.txt"
    printf '\n' >> "$dir/inference_command.txt"

    date --iso-8601=seconds > "$dir/start_time.txt"
    nvidia-smi -i "$GPU_ID" \
        > "$dir/nvidia-smi_before.txt"

    sha256sum \
        "$checkpoint" \
        "$final_sim/config.yaml" \
        "$final_sim/noises.npy" \
        "$final_sim/flows.npy" \
        "$final_sim/resized_input_image.png" \
        "$final_sim/prompt.txt" \
        > "$dir/inputs_sha256.txt"

    if [ "$mode" != "baseline" ]; then
        sha256sum "$artifact" \
            >> "$dir/inputs_sha256.txt"
    fi

    set +e

    timeout \
        --signal=TERM \
        --kill-after=20s \
        3600s \
        "$python" -u infer_sim.py "${args[@]}" \
        > "$dir/stdout.log" \
        2> "$dir/stderr.log"

    local code=$?

    set -e

    echo "$code" > "$dir/exit_code.txt"
    date --iso-8601=seconds > "$dir/end_time.txt"

    nvidia-smi -i "$GPU_ID" \
        > "$dir/nvidia-smi_after.txt"

    if [ "$code" -ne 0 ]; then
        echo "$mode 推理失败，exit_code=$code" >&2
        return "$code"
    fi

    if [ ! -s "$output" ]; then
        echo "$mode 返回0但没有生成有效视频" >&2
        return 66
    fi

    sha256sum "$output" \
        > "$dir/output_sha256.txt"

    echo "success" > "$dir/status.txt"
}

echo baseline > "$job/current_stage.txt"
run_variant baseline

echo correct > "$job/current_stage.txt"
run_variant correct

echo shuffled > "$job/current_stage.txt"
run_variant shuffled

echo complete > "$job/current_stage.txt"
echo "success" > "$job/status.txt"
