#!/usr/bin/env bash
set -eo pipefail
[ "$#" -eq 2 ] || { echo '用法: launch_one.sh GPU baseline|correct|shuffled' >&2; exit 64; }
gpu="$1"; mode="$2"
case "$gpu" in 0|1|2|3);; *) exit 64;; esac
case "$mode" in baseline|correct|shuffled|flow|blend);; *) exit 64;; esac
repo=/mnt/sdbd/home/liuyu_qyh/DeformTransport
run_id=20260804_234925_autonomous_deformtransport
root="$repo/server_runs/$run_id"
croot="/workspace/DeformTransport/server_runs/$run_id"
final_sim="$root/prepared_inputs/santa_21f_final_sim_proxy_v1"
checkpoint="$repo/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"
artifact="$repo/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/correct_blend_alpha0p5_v1/correct_blend_alpha0p5.pt"
/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/bin/python -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r["通过"] is True' "$final_sim/validation_report.json"
docker exec -i deformtransport-dev test -s /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt || exit 66
if [ "$mode" != baseline ]; then [ -s "$artifact" ] || exit 66; fi
uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F, -v u="$uuid" '$1 ~ u {print $2}')"
res="$(nvidia-smi -i "$gpu" --query-gpu=memory.free,temperature.gpu --format=csv,noheader,nounits)"
free_now="$(echo "$res"|cut -d, -f1|tr -d ' ')"; temp_now="$(echo "$res"|cut -d, -f2|tr -d ' ')"; mem_now="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
[ -z "$pids" ] && [ "$free_now" -ge 40000 ] && [ "$temp_now" -lt 80 ] && [ "$mem_now" -ge 52428800 ] || { echo "GPU${gpu}首次完整推理门禁失败: pids=$pids free=$free_now temp=$temp_now MemAvailableKiB=$mem_now" >&2; exit 75; }
lock="/tmp/deformtransport_gpu_${gpu}.lock"; [ ! -e "$lock" ] || exit 73
stamp="$(date +%Y%m%d_%H%M%S)"; upper="$(echo "$mode"|tr '[:lower:]' '[:upper:]')"; job="REALWONDER_SANTA_${upper}_${stamp}"; dir="$root/04_smoke/$job"; cdir="$croot/04_smoke/$job"; mkdir -p "$dir"
out="$cdir/santa_${mode}_seed0.mp4"
args=(--checkpoint_path /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt --sim_data_path "$croot/prepared_inputs/santa_21f_final_sim_proxy_v1" --output_path "$out" --seed 0 --eval_degradation 0.5 --local_attn_size 21)
if [ "$mode" != baseline ]; then args+=(--transport_latent_path /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/correct_blend_alpha0p5_v1/correct_blend_alpha0p5.pt --transport_mode "$mode"); fi
printf '%q ' python -u infer_sim.py "${args[@]}" > "$dir/inference_command.txt"; printf '\n' >> "$dir/inference_command.txt"
date --iso-8601=seconds > "$dir/开始时间.txt"; echo "$gpu" > "$dir/GPU编号.txt"; nvidia-smi -i "$gpu" > "$dir/nvidia-smi_before.txt"; (cd "$repo" && git status --short) > "$dir/git_status.txt"; cp "$root/01_environment/final_pip_state_after_repairs.txt" "$dir/environment.txt"
cp "$root/00_audit/model_sha256.txt" "$dir/inputs_sha256.txt"; sha256sum "$final_sim/config.yaml" "$final_sim/noises.npy" "$final_sim/resized_input_image.png" "$final_sim/prompt.txt" >> "$dir/inputs_sha256.txt"; if [ "$mode" != baseline ]; then sha256sum "$artifact" >> "$dir/inputs_sha256.txt"; fi
tee "$dir/manifest.yaml" >/dev/null <<EOF
RUN_ID: $run_id
job_id: $job
任务: 原生RealWonder_Santa_${mode}_首次完整smoke
GPU编号: $gpu
GPU_UUID: $uuid
seed: 0
eval_degradation: 0.5
checkpoint: $checkpoint
输入目录: $final_sim
transport模式: $mode
timeout秒: 2400
结论边界: Santa输入为有损工程proxy_不是原始simulation_future_GT
EOF
tee "$dir/command.sh" >/dev/null <<EOF
#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
export TORCH_HOME=/workspace/DeformTransport/server_runs/$run_id/prepared_inputs/torch_cache
python -u infer_sim.py $(printf '%q ' "${args[@]}") &
child=\$!
echo \$child > "$cdir/container_pid.txt"
while kill -0 \$child 2>/dev/null; do
  ts=\$(date --iso-8601=seconds); rss=\$(awk '/VmRSS/{print \$2}' /proc/\$child/status 2>/dev/null || echo 0); hwm=\$(awk '/VmHWM/{print \$2}' /proc/\$child/status 2>/dev/null || echo 0); printf '%s,%s,%s\n' "\$ts" "\$rss" "\$hwm" >> "$cdir/process_memory.csv"; sleep 15
done
wait \$child
EOF
chmod +x "$dir/command.sh"
tee "$lock" >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=$gpu
GPU_UUID=$uuid
job_id=$job
任务名称=原生RealWonder_Santa_${mode}
PID=$$
运行目录=$dir
启动时间=$(date --iso-8601=seconds)
EOF
echo 运行中 > "$dir/status.txt"
docker exec -i -e CUDA_VISIBLE_DEVICES="$gpu" -e DEFORMTRANSPORT_RUN_ID="$run_id" deformtransport-dev timeout --signal=TERM --kill-after=20s 2400s bash "$cdir/command.sh" > "$dir/stdout.log" 2> "$dir/stderr.log" & task="$!"; echo "$task" > "$dir/pid.txt"
set +e
while kill -0 "$task" 2>/dev/null; do
 ts="$(date --iso-8601=seconds)"; line="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)"; mem="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"; disk="$(du -sb "$dir"|awk '{print $1}')"; printf '%s,%s,%s,%s\n' "$ts" "$line" "$mem" "$disk" >> "$dir/resource_usage.csv"; free="$(echo "$line"|cut -d, -f2|tr -d ' ')"; temp="$(echo "$line"|cut -d, -f4|tr -d ' ')"; ecc="$(echo "$line"|cut -d, -f6|tr -d ' ')"; if [ "$free" -lt 2048 ] || [ "$temp" -ge 85 ] || [ "$ecc" -gt 0 ] || [ "$mem" -lt 20971520 ]; then echo '实际资源安全阈值触发' > "$dir/safety_stop_reason.txt"; if [ -s "$dir/container_pid.txt" ]; then docker exec -i deformtransport-dev bash -lc "kill -TERM $(cat "$dir/container_pid.txt")" >/dev/null 2>&1; fi; break; fi; sleep 15
done
wait "$task"; code=$?; set -e
echo "$code" > "$dir/exit_code.txt"; date --iso-8601=seconds > "$dir/结束时间.txt"; nvidia-smi -i "$gpu" > "$dir/nvidia-smi_after.txt"; [ "$code" -eq 0 ] && [ -s "$dir/santa_${mode}_seed0.mp4" ] && echo 成功 > "$dir/status.txt" || echo "失败_退出码=$code" > "$dir/status.txt"; rm -f "$lock"; printf 'job_id=%s\nGPU=%s\nPID=%s\n运行目录=%s\n退出码=%s\n' "$job" "$gpu" "$$" "$dir" "$code"; exit "$code"
