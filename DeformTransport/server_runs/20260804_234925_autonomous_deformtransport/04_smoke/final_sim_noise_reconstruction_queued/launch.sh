#!/usr/bin/env bash
set -eo pipefail
[ "$#" -eq 1 ] || exit 64
gpu="$1"; case "$gpu" in 0|1|2|3);; *) exit 64;; esac
repo=/mnt/sdbd/home/liuyu_qyh/DeformTransport
run_id=20260804_234925_autonomous_deformtransport
root="$repo/server_runs/$run_id"
py=/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/bin/python
final_sim="$root/prepared_inputs/santa_21f_final_sim_proxy_v1"
[ ! -e "$final_sim/noises.npy" ] && [ ! -e "$final_sim/flows.npy" ] || { echo '拒绝覆盖已有noise/flow' >&2; exit 73; }
"$py" "$root/gpu_scheduler/analyze_shareability.py" >/dev/null
state="$($py -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["GPU"][sys.argv[2]]["WanVAE状态"])' "$root/gpu_scheduler/shareability_latest.json" "$gpu")"
if [ "$state" != SHAREABLE ]; then
  uuid_now="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
  direct="$(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"
  free_now="$(echo "$direct" | cut -d, -f1 | tr -d ' ')"; util_now="$(echo "$direct" | cut -d, -f2 | tr -d ' ')"; temp_now="$(echo "$direct" | cut -d, -f3 | tr -d ' ')"
  pids_now="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F, -v u="$uuid_now" '$1 ~ u {print $2}')"
  [ -z "$pids_now" ] && [ "$free_now" -ge 18432 ] && [ "$util_now" -le 10 ] && [ "$temp_now" -lt 80 ] || { echo "GPU${gpu} 快速空卡门禁失败" >&2; exit 75; }
fi
lock="/tmp/deformtransport_gpu_${gpu}.lock"; [ ! -e "$lock" ] || exit 73
stamp="$(date +%Y%m%d_%H%M%S)"; job="FINAL_SIM_NOISE_RECON_${stamp}"; dir="$root/04_smoke/$job"; cdir="/workspace/DeformTransport/server_runs/$run_id/04_smoke/$job"; mkdir -p "$dir"
uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
date --iso-8601=seconds > "$dir/开始时间.txt"; printf '%s\n' "$gpu" > "$dir/GPU编号.txt"; nvidia-smi -i "$gpu" > "$dir/nvidia-smi_before.txt"; (cd "$repo" && git status --short) > "$dir/git_status.txt"; cp "$root/01_environment/final_pip_state_after_repairs.txt" "$dir/environment.txt"
docker exec -i deformtransport-dev sha256sum /workspace/DeformTransport/server_runs/$run_id/04_smoke/final_sim_noise_reconstruction_queued/generate_noise.py /workspace/DeformTransport/server_runs/$run_id/prepared_inputs/torch_cache/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth /workspace/DeformTransport/server_runs/$run_id/prepared_inputs/santa_21f_final_sim_proxy_v1/provenance.json > "$dir/inputs_sha256.txt"
tee "$dir/manifest.yaml" >/dev/null <<EOF
RUN_ID: $run_id
job_id: $job
任务: Santa_21帧proxy使用项目原NoiseWarper与官方RAFT重建structured_noise
GPU编号: $gpu
GPU_UUID: $uuid
调度分类: SHAREABLE_SHORT
seed: 0
timeout秒: 600
输入目录: /workspace/DeformTransport/server_runs/$run_id/prepared_inputs/santa_21f_final_sim_proxy_v1
输出: noises.npy与flows.npy及noise_generation_report.json
结论边界: 有损工程proxy_不是原始final_sim或future_GT
EOF
tee "$dir/command.sh" >/dev/null <<EOF
#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo \$\$ > "$cdir/container_pid.txt"
export TORCH_HOME=/workspace/DeformTransport/server_runs/$run_id/prepared_inputs/torch_cache
exec timeout --signal=TERM --kill-after=10s 600s python -u server_runs/$run_id/04_smoke/final_sim_noise_reconstruction_queued/generate_noise.py server_runs/$run_id/prepared_inputs/santa_21f_final_sim_proxy_v1 --seed 0
EOF
chmod +x "$dir/command.sh"
tee "$lock" >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=$gpu
GPU_UUID=$uuid
job_id=$job
任务名称=Santa_final_sim_proxy_RAFT_structured_noise重建
PID=$$
运行目录=$dir
启动时间=$(date --iso-8601=seconds)
EOF
echo 运行中 > "$dir/status.txt"
docker exec -i -e CUDA_VISIBLE_DEVICES="$gpu" -e DEFORMTRANSPORT_RUN_ID="$run_id" deformtransport-dev bash "$cdir/command.sh" > "$dir/stdout.log" 2> "$dir/stderr.log" & task="$!"; echo "$task" > "$dir/pid.txt"
set +e
while kill -0 "$task" 2>/dev/null; do
 ts="$(date --iso-8601=seconds)"; line="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)"; mem="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; printf '%s,%s,%s\n' "$ts" "$line" "$mem" >> "$dir/resource_usage.csv"; free="$(echo "$line"|cut -d, -f2|tr -d ' ')"; temp="$(echo "$line"|cut -d, -f4|tr -d ' ')"; ecc="$(echo "$line"|cut -d, -f6|tr -d ' ')"; if [ "$free" -lt 10240 ] || [ "$temp" -ge 85 ] || [ "$ecc" -gt 0 ] || [ "$mem" -lt 20971520 ]; then echo '实际资源安全阈值触发' > "$dir/safety_stop_reason.txt"; [ -s "$dir/container_pid.txt" ] && docker exec -i deformtransport-dev bash -lc "kill -TERM $(cat "$dir/container_pid.txt")" >/dev/null 2>&1; break; fi; sleep 5
done
wait "$task"; code=$?; set -e
echo "$code" > "$dir/exit_code.txt"; date --iso-8601=seconds > "$dir/结束时间.txt"; nvidia-smi -i "$gpu" > "$dir/nvidia-smi_after.txt"; if [ "$code" -eq 0 ] && [ -s "$final_sim/noise_generation_report.json" ]; then echo 成功 > "$dir/status.txt"; else echo "失败_退出码=$code" > "$dir/status.txt"; fi; rm -f "$lock"; printf 'job_id=%s\nGPU=%s\nPID=%s\n运行目录=%s\n退出码=%s\n' "$job" "$gpu" "$$" "$dir" "$code"; exit "$code"
