#!/usr/bin/env bash
set -eo pipefail
[ "$#" -eq 1 ] || exit 64
gpu="$1"; repo=/mnt/sdbd/home/liuyu_qyh/DeformTransport; run_id=20260804_234925_autonomous_deformtransport; root="$repo/server_runs/$run_id"; checkpoint="$repo/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"
uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits|tr -d ' ')"; direct="$(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"; free="$(echo "$direct"|cut -d, -f1|tr -d ' ')"; util="$(echo "$direct"|cut -d, -f2|tr -d ' ')"; temp="$(echo "$direct"|cut -d, -f3|tr -d ' ')"; pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits|awk -F, -v u="$uuid" '$1~u {print $2}')"; [ -z "$pids" ] && [ "$free" -ge 30720 ] && [ "$util" -le 10 ] && [ "$temp" -lt 80 ] || exit 75
mem="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"; [ "$mem" -ge 52428800 ] || exit 76
lock="/tmp/deformtransport_gpu_${gpu}.lock"; [ ! -e "$lock" ] || exit 73; stamp="$(date +%Y%m%d_%H%M%S)"; job="REALWONDER_GENERATOR_LOAD_${stamp}"; dir="$root/04_smoke/$job"; cdir="/workspace/DeformTransport/server_runs/$run_id/04_smoke/$job"; mkdir -p "$dir/outputs"; date --iso-8601=seconds > "$dir/开始时间.txt"; echo "$gpu" > "$dir/GPU编号.txt"; nvidia-smi -i "$gpu" > "$dir/nvidia-smi_before.txt"; (cd "$repo" && git status --short) > "$dir/git_status.txt"; cp "$root/01_environment/final_pip_state_after_repairs.txt" "$dir/environment.txt"
tee "$dir/manifest.yaml" >/dev/null <<EOF
RUN_ID: $run_id
job_id: $job
任务: 原生RealWonder_generator_checkpoint_GPU加载探针
GPU编号: $gpu
GPU_UUID: $uuid
结论边界: 不调用inference_不是Baseline结果
timeout秒: 600
checkpoint: /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt
EOF
tee "$dir/command.sh" >/dev/null <<EOF
#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo \$\$ > "$cdir/container_pid.txt"
exec timeout --signal=TERM --kill-after=15s 600s python -u server_runs/$run_id/04_smoke/generator_load_probe_queued/generator_load_probe.py --checkpoint /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt --output "$cdir/outputs"
EOF
chmod +x "$dir/command.sh"; tee "$lock" >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=$gpu
GPU_UUID=$uuid
job_id=$job
任务名称=原生RealWonder_generator_checkpoint_GPU加载探针
PID=$$
运行目录=$dir
启动时间=$(date --iso-8601=seconds)
EOF
echo 运行中 > "$dir/status.txt"; docker exec -i -e CUDA_VISIBLE_DEVICES="$gpu" deformtransport-dev bash "$cdir/command.sh" > "$dir/stdout.log" 2> "$dir/stderr.log" & task=$!; echo "$task" > "$dir/pid.txt"; set +e
while kill -0 "$task" 2>/dev/null; do ts="$(date --iso-8601=seconds)"; line="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)"; mem="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"; printf '%s,%s,%s\n' "$ts" "$line" "$mem" >> "$dir/resource_usage.csv"; free="$(echo "$line"|cut -d, -f2|tr -d ' ')"; temp="$(echo "$line"|cut -d, -f4|tr -d ' ')"; ecc="$(echo "$line"|cut -d, -f6|tr -d ' ')"; if [ "$free" -lt 10240 ] || [ "$temp" -ge 85 ] || [ "$ecc" -gt 0 ] || [ "$mem" -lt 20971520 ]; then echo '实际资源安全阈值触发' > "$dir/safety_stop_reason.txt"; [ -s "$dir/container_pid.txt" ] && docker exec -i deformtransport-dev bash -lc "kill -TERM $(cat "$dir/container_pid.txt")" >/dev/null 2>&1; break; fi; sleep 5; done
wait "$task"; code=$?; set -e; echo "$code" > "$dir/exit_code.txt"; date --iso-8601=seconds > "$dir/结束时间.txt"; nvidia-smi -i "$gpu" > "$dir/nvidia-smi_after.txt"; [ "$code" -eq 0 ] && [ -s "$dir/outputs/generator_load_report.json" ] && echo 成功 > "$dir/status.txt" || echo "失败_退出码=$code" > "$dir/status.txt"; rm -f "$lock"; printf 'job_id=%s\nGPU=%s\nPID=%s\n运行目录=%s\n退出码=%s\n' "$job" "$gpu" "$$" "$dir" "$code"; exit "$code"
