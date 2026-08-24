#!/usr/bin/env bash
set -eo pipefail
repo=/mnt/sdbd/home/liuyu_qyh/DeformTransport; run_id=20260804_234925_autonomous_deformtransport; root="$repo/server_runs/$run_id"; croot="/workspace/DeformTransport/server_runs/$run_id"; uuid="$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader,nounits|tr -d ' ')"; pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits|awk -F, -v u="$uuid" '$1~u{print $2}')"; [ -z "$pids" ] || exit 75; [ ! -e /tmp/deformtransport_gpu_3.lock ] || exit 73; stamp="$(date +%Y%m%d_%H%M%S)"; job="I2V_VAE_CONDITIONING_$stamp"; dir="$root/04_smoke/$job"; cdir="$croot/04_smoke/$job"; mkdir -p "$dir/outputs"; echo 3 > "$dir/GPU编号.txt"; date --iso-8601=seconds > "$dir/开始时间.txt"; nvidia-smi -i 3 > "$dir/nvidia-smi_before.txt"; sha256sum "$root/prepared_inputs/santa_21f_final_sim_proxy_v1/resized_input_image.png" > "$dir/inputs_sha256.txt"; tee "$dir/manifest.yaml" >/dev/null <<EOF
RUN_ID: $run_id
job_id: $job
任务: 原生WanVideoUnit_ImageEmbedderVAE条件编码回归
GPU编号: 3
GPU_UUID: $uuid
像素帧数: 21
EOF
tee /tmp/deformtransport_gpu_3.lock >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=3
job_id=$job
任务名称=I2V_VAE条件编码
PID=$$
运行目录=$dir
启动时间=$(date --iso-8601=seconds)
EOF
printf '%s\n' 'CUDA_VISIBLE_DEVICES=3 python probe.py <Santa initial> <outputs>' > "$dir/command.sh"; echo 运行中 > "$dir/status.txt"; docker exec -i -e CUDA_VISIBLE_DEVICES=3 -e PYTHONPATH=/workspace/DeformTransport deformtransport-dev timeout 600s bash -lc 'source /workspace/tools/miniforge3/etc/profile.d/conda.sh && conda activate realwonder-gen && cd /workspace/DeformTransport && python -u server_runs/'$run_id'/04_smoke/i2v_vae_conditioning_queued/probe.py server_runs/'$run_id'/prepared_inputs/santa_21f_final_sim_proxy_v1/resized_input_image.png '$cdir'/outputs' > "$dir/stdout.log" 2> "$dir/stderr.log" & task=$!; echo "$task" > "$dir/pid.txt"; set +e; while kill -0 "$task" 2>/dev/null; do ts="$(date --iso-8601=seconds)"; line="$(nvidia-smi -i 3 --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)"; mem="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"; printf '%s,%s,%s\n' "$ts" "$line" "$mem" >> "$dir/resource_usage.csv"; sleep 15; done; wait "$task"; code=$?; set -e; echo "$code" > "$dir/exit_code.txt"; date --iso-8601=seconds > "$dir/结束时间.txt"; nvidia-smi -i 3 > "$dir/nvidia-smi_after.txt"; [ "$code" -eq 0 ] && [ -s "$dir/outputs/report.json" ] && echo 成功 > "$dir/status.txt" || echo "失败_退出码=$code" > "$dir/status.txt"; rm -f /tmp/deformtransport_gpu_3.lock; printf 'job_id=%s\nGPU=3\nPID=%s\n运行目录=%s\n退出码=%s\n' "$job" "$$" "$dir" "$code"; exit "$code"
