#!/usr/bin/env bash
set -eo pipefail

if [ "$#" -ne 1 ]; then
  echo "用法：launch_if_free.sh GPU编号" >&2
  exit 64
fi
gpu_id="$1"
case "$gpu_id" in 0|1|2|3) ;; *) echo "GPU编号必须为0、1、2或3" >&2; exit 64 ;; esac

run_id="20260804_234925_autonomous_deformtransport"
repo="/mnt/sdbd/home/liuyu_qyh/DeformTransport"
run_root="$repo/server_runs/$run_id"
queue_dir="$run_root/04_smoke/wan_vae_transport_gpu_smoke_queued"
lock_path="/tmp/deformtransport_gpu_${gpu_id}.lock"

check_gate() {
  local stat_line gpu_uuid memory_used utilization temperature
  stat_line="$(nvidia-smi -i "$gpu_id" --query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)"
  IFS=',' read -r gpu_uuid memory_used utilization temperature <<< "$stat_line"
  gpu_uuid="${gpu_uuid// /}"
  memory_used="${memory_used// /}"
  utilization="${utilization// /}"
  temperature="${temperature// /}"
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F',' -v wanted="$gpu_uuid" '{gsub(/ /,"",$1); if ($1 == wanted) found=1} END {exit found ? 0 : 1}'; then
    echo "门禁失败：GPU${gpu_id} 存在 compute PID" >&2
    return 1
  fi
  if [ "$memory_used" -gt 512 ] || [ "$utilization" -gt 5 ] || [ "$temperature" -gt 85 ]; then
    echo "门禁失败：GPU${gpu_id} 显存=${memory_used}MiB，利用率=${utilization}%，温度=${temperature}°C" >&2
    return 1
  fi
  printf '%s,%s,%s,%s\n' "$gpu_uuid" "$memory_used" "$utilization" "$temperature"
}

if [ -e "$lock_path" ]; then
  echo "门禁失败：锁已存在 $lock_path" >&2
  cat "$lock_path" >&2
  exit 73
fi
first_gate="$(check_gate)" || exit 75
second_gate="$(check_gate)" || exit 75

task_stamp="$(date +%Y%m%d_%H%M%S)"
task_name="wan_vae_transport_gpu_smoke_${task_stamp}"
run_dir="$run_root/04_smoke/$task_name"
container_run_dir="/workspace/DeformTransport/server_runs/$run_id/04_smoke/$task_name"
mkdir -p "$run_dir/outputs"
cp "$queue_dir/run_gpu_smoke.py" "$run_dir/run_gpu_smoke.py"
printf '%s\n' "$gpu_id" > "$run_dir/GPU编号.txt"
date --iso-8601=seconds > "$run_dir/开始时间.txt"
nvidia-smi -i "$gpu_id" > "$run_dir/nvidia-smi_before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$run_dir/compute_processes_before.txt"
git -C "$repo" status --short > "$run_dir/git_status.txt"
sha256sum \
  "$repo/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt" \
  "$repo/wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth" \
  "$run_dir/run_gpu_smoke.py" > "$run_dir/输入资产与脚本SHA256.txt"
docker exec -i deformtransport-dev bash -lc 'source /workspace/tools/miniforge3/etc/profile.d/conda.sh; conda activate realwonder-gen; python - <<"PY"
import torch, numpy, diffusers, transformers, tokenizers, accelerate, imageio, einops, omegaconf, peft
for name, value in [("python", __import__("sys").version.split()[0]), ("torch", torch.__version__), ("cuda", torch.version.cuda), ("numpy", numpy.__version__), ("diffusers", diffusers.__version__), ("transformers", transformers.__version__), ("tokenizers", tokenizers.__version__), ("accelerate", accelerate.__version__), ("imageio", imageio.__version__), ("einops", einops.__version__), ("omegaconf", omegaconf.__version__), ("peft", peft.__version__)]: print(f"{name}={value}")
PY' > "$run_dir/环境版本.txt"

tee "$run_dir/manifest.yaml" >/dev/null <<EOF
run_id: $run_id
任务名称: $task_name
状态: 启动准备完成
GPU编号: $gpu_id
首次门禁: "$first_gate"
第二次门禁: "$second_gate"
seed: 0
checkpoint: /workspace/DeformTransport/wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth
artifact: /workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt
输出目录: $container_run_dir/outputs
开始时间: "$(date --iso-8601=seconds)"
EOF
tee "$run_dir/command.sh" >/dev/null <<EOF
#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo \$\$ > "$container_run_dir/container_pid.txt"
exec python -u "$container_run_dir/run_gpu_smoke.py" \\
  --artifact artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt \\
  --checkpoint wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth \\
  --output-dir "$container_run_dir/outputs" \\
  --seed 0
EOF
chmod +x "$run_dir/command.sh"

final_gate="$(check_gate)" || {
  printf '%s\n' "启动前最终门禁失败，未创建锁，未启动任务。" > "$run_dir/status.txt"
  exit 75
}

nohup docker exec -i \
  -e CUDA_VISIBLE_DEVICES="$gpu_id" \
  -e DEFORMTRANSPORT_RUN_ID="$run_id" \
  deformtransport-dev bash "$container_run_dir/command.sh" \
  > "$run_dir/stdout.log" 2> "$run_dir/stderr.log" &
host_pid="$!"
printf '%s\n' "$host_pid" > "$run_dir/pid.txt"
tee "$lock_path" >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=$gpu_id
PID=$host_pid
时间=$(date --iso-8601=seconds)
任务名称=$task_name
EOF
printf '%s\n' "已启动，等待真实模型加载。" > "$run_dir/status.txt"
nohup "$run_root/04_smoke/monitor_owned_process.sh" "$host_pid" "$gpu_id" "$run_dir/resource_usage.log" \
  > "$run_dir/monitor_stdout.log" 2> "$run_dir/monitor_stderr.log" &
printf '%s\n' "$!" > "$run_dir/monitor_pid.txt"
printf 'GPU编号=%s\n任务=%s\n运行目录=%s\nPID=%s\n最终门禁=%s\n锁=%s\n' \
  "$gpu_id" "$task_name" "$run_dir" "$host_pid" "$final_gate" "$lock_path"
