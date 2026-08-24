#!/usr/bin/env bash
set -eo pipefail
if [ "$#" -ne 1 ]; then exit 64; fi
gpu="$1"; case "$gpu" in 0|1|2|3) ;; *) exit 64 ;; esac
repo="/mnt/sdbd/home/liuyu_qyh/DeformTransport"; run_id="20260804_234925_autonomous_deformtransport"; root="$repo/server_runs/$run_id"; py="/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/bin/python"
"$py" "$root/gpu_scheduler/analyze_shareability.py" >/dev/null
state="$($py -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["GPU"][sys.argv[2]]["WanVAE状态"])' "$root/gpu_scheduler/shareability_latest.json" "$gpu")"
[ "$state" = "SHAREABLE" ] || { echo "GPU${gpu} WanVAE状态=$state" >&2; exit 75; }
uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"; lock="/tmp/deformtransport_gpu_${gpu}.lock"; [ ! -e "$lock" ] || exit 73
stamp="$(date +%Y%m%d_%H%M%S)"; job="WAN_VAE_E2E_21F_${stamp}"; dir="$root/04_smoke/$job"; cdir="/workspace/DeformTransport/server_runs/$run_id/04_smoke/$job"; mkdir -p "$dir/outputs"
date --iso-8601=seconds > "$dir/开始时间.txt"; printf '%s\n' "$gpu" > "$dir/GPU编号.txt"; nvidia-smi -i "$gpu" > "$dir/nvidia-smi_before.txt"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$dir/compute_processes_before.txt"; (cd "$repo" && git status --short) > "$dir/git_status.txt"; cp "$root/gpu_scheduler/shareability_latest.json" "$dir/shareability_before.json"; cp "$root/01_environment/final_pip_state_after_repairs.txt" "$dir/环境版本.txt"
docker exec -i deformtransport-dev sha256sum /workspace/DeformTransport/scripts/run_wan_vae_transport_probe.py /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/santa_21f_videoproxy_transport_ready/transport_ready_videoproxy_v3.pt /workspace/DeformTransport/wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth > "$dir/输入模型与脚本SHA256.txt"
tee "$dir/manifest.yaml" >/dev/null <<EOF
RUN_ID: $run_id
job_id: $job
任务: 真实Wan VAE 21帧编码_transport_解码闭环
状态: 启动中
GPU编号: $gpu
GPU_UUID: $uuid
调度分类: SHAREABLE_WAN_VAE
历史峰值显存MiB: 6190
要求空闲显存MiB: 18633
预计时间: 60秒内
timeout秒: 300
seed: 0
输入: /workspace/DeformTransport/server_runs/$run_id/prepared_inputs/santa_21f_videoproxy_transport_ready/transport_ready_videoproxy_v3.pt
checkpoint: /workspace/DeformTransport/wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth
开始时间: "$(date --iso-8601=seconds)"
EOF
tee "$dir/command.sh" >/dev/null <<EOF
#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo \$\$ > "$cdir/container_pid.txt"
exec timeout --signal=TERM --kill-after=10s 300s python -u scripts/run_wan_vae_transport_probe.py \\
 --transport-ready "/workspace/DeformTransport/server_runs/$run_id/prepared_inputs/santa_21f_videoproxy_transport_ready/transport_ready_videoproxy_v3.pt" \\
 --checkpoint wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth \\
 --output-dir "$cdir/outputs" --seed 0 --visual-inspection pending --visual-note "服务器本轮真实GPU闭环；输入为明确标注的MP4派生proxy"
EOF
chmod +x "$dir/command.sh"
"$py" "$root/gpu_scheduler/analyze_shareability.py" >/dev/null; state="$($py -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["GPU"][sys.argv[2]]["WanVAE状态"])' "$root/gpu_scheduler/shareability_latest.json" "$gpu")"; [ "$state" = "SHAREABLE" ] || { echo "最终门禁失败" > "$dir/status.txt"; exit 75; }
tee "$lock" >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=$gpu
GPU_UUID=$uuid
job_id=$job
任务名称=真实Wan_VAE_21帧编码_transport_解码闭环
PID=$$
命令=$dir/command.sh
运行目录=$dir
启动时间=$(date --iso-8601=seconds)
EOF
printf '%s\n' "运行中" > "$dir/status.txt"
docker exec -i -e CUDA_VISIBLE_DEVICES="$gpu" -e DEFORMTRANSPORT_RUN_ID="$run_id" deformtransport-dev bash "$cdir/command.sh" > "$dir/stdout.log" 2> "$dir/stderr.log" & task_pid="$!"; printf '%s\n' "$task_pid" > "$dir/pid.txt"
set +e
while kill -0 "$task_pid" 2>/dev/null; do
 timestamp="$(date --iso-8601=seconds)"; gpu_line="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)"; mem_kib="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; printf '%s,%s,%s\n' "$timestamp" "$gpu_line" "$mem_kib" >> "$dir/resource_usage.csv"; free_mib="$(echo "$gpu_line" | cut -d, -f2 | tr -d ' ')"; temp="$(echo "$gpu_line" | cut -d, -f4 | tr -d ' ')"; ecc="$(echo "$gpu_line" | cut -d, -f6 | tr -d ' ')"; if [ "$free_mib" -lt 10240 ] || [ "$temp" -ge 85 ] || [ "$ecc" -gt 0 ] || [ "$mem_kib" -lt 20971520 ]; then reason="共享安全阈值触发"; echo "$reason" > "$dir/safety_stop_reason.txt"; if [ -s "$dir/container_pid.txt" ]; then cpid="$(cat "$dir/container_pid.txt")"; docker exec -i deformtransport-dev bash -lc "kill -TERM $cpid" >/dev/null 2>&1; fi; break; fi; sleep 5
done
wait "$task_pid"; code="$?"; set -e
printf '%s\n' "$code" > "$dir/exit_code.txt"; nvidia-smi -i "$gpu" > "$dir/nvidia-smi_after.txt"; date --iso-8601=seconds > "$dir/结束时间.txt"; if [ "$code" -eq 0 ] && [ -s "$dir/outputs/report.json" ]; then echo 成功 > "$dir/status.txt"; else echo "失败_退出码=$code" > "$dir/status.txt"; fi; rm -f "$lock"; printf 'job_id=%s\nGPU=%s\nPID=%s\n运行目录=%s\n退出码=%s\n' "$job" "$gpu" "$$" "$dir" "$code"; exit "$code"
