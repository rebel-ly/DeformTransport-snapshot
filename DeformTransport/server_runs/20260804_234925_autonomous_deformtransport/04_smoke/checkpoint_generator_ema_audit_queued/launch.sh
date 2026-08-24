#!/usr/bin/env bash
set -eo pipefail
repo=/mnt/sdbd/home/liuyu_qyh/DeformTransport; run_id=20260804_234925_autonomous_deformtransport; root="$repo/server_runs/$run_id"; croot="/workspace/DeformTransport/server_runs/$run_id"; gpu=3
uuid="$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader,nounits|tr -d ' ')"; pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits|awk -F, -v u="$uuid" '$1~u{print $2}')"; res="$(nvidia-smi -i 3 --query-gpu=memory.free,temperature.gpu --format=csv,noheader,nounits)"; free="$(echo "$res"|cut -d, -f1|tr -d ' ')"; temp="$(echo "$res"|cut -d, -f2|tr -d ' ')"; mem="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"; [ -z "$pids" ] && [ "$free" -ge 30000 ] && [ "$temp" -lt 80 ] && [ "$mem" -ge 52428800 ] || exit 75
lock=/tmp/deformtransport_gpu_3.lock; [ ! -e "$lock" ] || exit 73; stamp="$(date +%Y%m%d_%H%M%S)"; job="REALWONDER_CHECKPOINT_GENERATOR_EMA_AUDIT_$stamp"; dir="$root/04_smoke/$job"; cdir="$croot/04_smoke/$job"; mkdir -p "$dir/outputs"; date --iso-8601=seconds > "$dir/开始时间.txt"; echo 3 > "$dir/GPU编号.txt"; nvidia-smi -i 3 > "$dir/nvidia-smi_before.txt"; cp "$root/00_audit/model_sha256.txt" "$dir/inputs_sha256.txt"
tee "$dir/manifest.yaml" >/dev/null <<EOF
RUN_ID: $run_id
job_id: $job
任务: checkpoint_generator与generator_ema_GPU逐张量差异审计
GPU编号: 3
GPU_UUID: $uuid
timeout秒: 600
EOF
tee "$dir/command.sh" >/dev/null <<EOF
#!/usr/bin/env bash
set -eo pipefail
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
python -u $croot/04_smoke/checkpoint_generator_ema_audit_queued/probe.py /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt $cdir/outputs
EOF
chmod +x "$dir/command.sh"; tee "$lock" >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=3
GPU_UUID=$uuid
job_id=$job
任务名称=checkpoint_generator_ema差异审计
PID=$$
运行目录=$dir
启动时间=$(date --iso-8601=seconds)
EOF
echo 运行中 > "$dir/status.txt"; docker exec -i -e CUDA_VISIBLE_DEVICES=3 deformtransport-dev timeout --signal=TERM --kill-after=15s 600s bash "$cdir/command.sh" > "$dir/stdout.log" 2> "$dir/stderr.log" & task=$!; echo "$task" > "$dir/pid.txt"; set +e
while kill -0 "$task" 2>/dev/null; do ts="$(date --iso-8601=seconds)"; line="$(nvidia-smi -i 3 --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)"; mem="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"; printf '%s,%s,%s\n' "$ts" "$line" "$mem" >> "$dir/resource_usage.csv"; sleep 15; done
wait "$task"; code=$?; set -e; echo "$code" > "$dir/exit_code.txt"; date --iso-8601=seconds > "$dir/结束时间.txt"; nvidia-smi -i 3 > "$dir/nvidia-smi_after.txt"; [ "$code" -eq 0 ] && [ -s "$dir/outputs/generator_ema_audit.json" ] && echo 成功 > "$dir/status.txt" || echo "失败_退出码=$code" > "$dir/status.txt"; rm -f "$lock"; printf 'job_id=%s\nGPU=3\nPID=%s\n运行目录=%s\n退出码=%s\n' "$job" "$$" "$dir" "$code"; exit "$code"
