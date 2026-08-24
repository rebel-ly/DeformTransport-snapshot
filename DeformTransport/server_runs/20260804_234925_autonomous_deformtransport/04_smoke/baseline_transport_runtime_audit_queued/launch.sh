#!/usr/bin/env bash
set -eo pipefail
gpu=3; repo=/mnt/sdbd/home/liuyu_qyh/DeformTransport; run_id=20260804_234925_autonomous_deformtransport; root="$repo/server_runs/$run_id"; croot="/workspace/DeformTransport/server_runs/$run_id"
uuid="$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader,nounits|tr -d ' ')"; pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits|awk -F, -v u="$uuid" '$1~u{print $2}')"; state="$(nvidia-smi -i 3 --query-gpu=memory.free,temperature.gpu --format=csv,noheader,nounits)"; free="$(echo "$state"|cut -d, -f1|tr -d ' ')"; temp="$(echo "$state"|cut -d, -f2|tr -d ' ')"; mem="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"; [ -z "$pids" ] && [ "$free" -ge 30000 ] && [ "$temp" -lt 80 ] && [ "$mem" -ge 52428800 ] || exit 75
lock=/tmp/deformtransport_gpu_3.lock; [ ! -e "$lock" ] || exit 73; stamp="$(date +%Y%m%d_%H%M%S)"; job="REALWONDER_BASELINE_TRANSPORT_RUNTIME_AUDIT_$stamp"; dir="$root/04_smoke/$job"; cdir="$croot/04_smoke/$job"; mkdir -p "$dir/outputs"; date --iso-8601=seconds > "$dir/开始时间.txt"; echo 3 > "$dir/GPU编号.txt"; nvidia-smi -i 3 > "$dir/nvidia-smi_before.txt"; cp "$root/00_audit/model_sha256.txt" "$dir/inputs_sha256.txt"; sha256sum "$root/prepared_inputs/santa_21f_final_sim_proxy_v1/noises.npy" "$repo/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt" >> "$dir/inputs_sha256.txt"
tee "$dir/manifest.yaml" >/dev/null <<EOF
RUN_ID: $run_id
job_id: $job
任务: 原生Baseline预生成GPU路径等价性与transport运行时分支审计
GPU编号: 3
GPU_UUID: $uuid
seed: 0
结论边界: 不执行去噪generator_不是视频结果
timeout秒: 900
EOF
tee "$dir/command.sh" >/dev/null <<EOF
#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
python -u server_runs/$run_id/04_smoke/baseline_transport_runtime_audit_queued/probe.py --checkpoint /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt --final-sim $croot/prepared_inputs/santa_21f_final_sim_proxy_v1 --artifact /workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt --output $cdir/outputs
EOF
chmod +x "$dir/command.sh"; tee "$lock" >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=3
GPU_UUID=$uuid
job_id=$job
任务名称=Baseline路径等价性与transport运行时审计
PID=$$
运行目录=$dir
启动时间=$(date --iso-8601=seconds)
EOF
echo 运行中 > "$dir/status.txt"; docker exec -i -e CUDA_VISIBLE_DEVICES=3 deformtransport-dev timeout --signal=TERM --kill-after=15s 900s bash "$cdir/command.sh" > "$dir/stdout.log" 2> "$dir/stderr.log" & task=$!; echo "$task" > "$dir/pid.txt"; set +e
while kill -0 "$task" 2>/dev/null; do ts="$(date --iso-8601=seconds)"; line="$(nvidia-smi -i 3 --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)"; mem="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"; printf '%s,%s,%s\n' "$ts" "$line" "$mem" >> "$dir/resource_usage.csv"; free="$(echo "$line"|cut -d, -f2|tr -d ' ')"; temp="$(echo "$line"|cut -d, -f4|tr -d ' ')"; ecc="$(echo "$line"|cut -d, -f6|tr -d ' ')"; if [ "$free" -lt 4096 ] || [ "$temp" -ge 85 ] || [ "$ecc" -gt 0 ] || [ "$mem" -lt 20971520 ]; then echo 风险停止 > "$dir/safety_stop_reason.txt"; break; fi; sleep 15; done
wait "$task"; code=$?; set -e; echo "$code" > "$dir/exit_code.txt"; date --iso-8601=seconds > "$dir/结束时间.txt"; nvidia-smi -i 3 > "$dir/nvidia-smi_after.txt"; [ "$code" -eq 0 ] && [ -s "$dir/outputs/runtime_audit_report.json" ] && echo 成功 > "$dir/status.txt" || echo "失败_退出码=$code" > "$dir/status.txt"; rm -f "$lock"; printf 'job_id=%s\nGPU=3\nPID=%s\n运行目录=%s\n退出码=%s\n' "$job" "$$" "$dir" "$code"; exit "$code"
