#!/usr/bin/env bash
set -eo pipefail
if [ "$#" -ne 1 ]; then echo "用法：launch_shareable.sh GPU编号" >&2; exit 64; fi
gpu_id="$1"; case "$gpu_id" in 0|1|2|3) ;; *) exit 64 ;; esac
repo="/mnt/sdbd/home/liuyu_qyh/DeformTransport"
run_id="20260804_234925_autonomous_deformtransport"
root="$repo/server_runs/$run_id"
python_host="/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/bin/python"
analyzer="$root/gpu_scheduler/analyze_shareability.py"
"$python_host" "$analyzer" >/dev/null
status="$($python_host -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["GPU"][sys.argv[2]]["状态"])' "$root/gpu_scheduler/shareability_latest.json" "$gpu_id")"
if [ "$status" != "SHAREABLE" ]; then echo "GPU${gpu_id}当前状态=$status，未启动" >&2; exit 75; fi
uuid="$(nvidia-smi -i "$gpu_id" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
lock="/tmp/deformtransport_gpu_${gpu_id}.lock"
if [ -e "$lock" ]; then echo "本项目锁已存在：$lock" >&2; exit 73; fi
stamp="$(date +%Y%m%d_%H%M%S)"
job_id="DT_TRANSPORT_GPU_PARITY_${stamp}"
run_dir="$root/04_smoke/$job_id"
container_dir="/workspace/DeformTransport/server_runs/$run_id/04_smoke/$job_id"
mkdir -p "$run_dir/outputs"
cp "$root/04_smoke/transport_gpu_validation_queued/run_transport_gpu_validation.py" "$run_dir/run_transport_gpu_validation.py"
printf '%s\n' "$gpu_id" > "$run_dir/GPU编号.txt"
date --iso-8601=seconds > "$run_dir/开始时间.txt"
nvidia-smi -i "$gpu_id" > "$run_dir/nvidia-smi_before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$run_dir/compute_processes_before.txt"
(cd "$repo" && git status --short) > "$run_dir/git_status.txt"
cp "$root/gpu_scheduler/shareability_latest.json" "$run_dir/shareability_before.json"
sha256sum "$repo/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt" "$root/prepared_inputs/santa_21f_videoproxy_transport_ready/transport_ready_videoproxy.pt" "$run_dir/run_transport_gpu_validation.py" > "$run_dir/输入资产与脚本SHA256.txt"
tee "$run_dir/manifest.yaml" >/dev/null <<EOF
RUN_ID: $run_id
job_id: $job_id
任务: 真实CUDA材料点transport重算一致性验证
状态: 启动中
GPU编号: $gpu_id
GPU_UUID: $uuid
调度分类: SHAREABLE
预计显存: 小于1GiB
预计时间: 小于60秒
timeout秒: 120
seed: 0
输入transport_ready: /workspace/DeformTransport/server_runs/$run_id/prepared_inputs/santa_21f_videoproxy_transport_ready/transport_ready_videoproxy.pt
输入latent_artifact: /workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt
开始时间: "$(date --iso-8601=seconds)"
EOF
tee "$run_dir/command.sh" >/dev/null <<EOF
#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo \$\$ > "$container_dir/container_pid.txt"
exec timeout --signal=TERM --kill-after=10s 120s python -u "$container_dir/run_transport_gpu_validation.py" \\
  --transport-ready "/workspace/DeformTransport/server_runs/$run_id/prepared_inputs/santa_21f_videoproxy_transport_ready/transport_ready_videoproxy.pt" \\
  --latent-artifact "/workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt" \\
  --output-dir "$container_dir/outputs" \\
  --seed 0
EOF
chmod +x "$run_dir/command.sh"
"$python_host" "$analyzer" >/dev/null
status="$($python_host -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["GPU"][sys.argv[2]]["状态"])' "$root/gpu_scheduler/shareability_latest.json" "$gpu_id")"
if [ "$status" != "SHAREABLE" ]; then printf '%s\n' "最终共享门禁失败，未创建锁，未启动。" > "$run_dir/status.txt"; exit 75; fi
tee "$lock" >/dev/null <<EOF
RUN_ID=$run_id
GPU编号=$gpu_id
GPU_UUID=$uuid
job_id=$job_id
任务名称=真实CUDA材料点transport重算一致性验证
PID=$$
命令=$run_dir/command.sh
运行目录=$run_dir
启动时间=$(date --iso-8601=seconds)
EOF
printf '%s\n' "运行中" > "$run_dir/status.txt"
set +e
docker exec -i -e CUDA_VISIBLE_DEVICES="$gpu_id" -e DEFORMTRANSPORT_RUN_ID="$run_id" deformtransport-dev bash "$container_dir/command.sh" > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
code="$?"
set -e
printf '%s\n' "$code" > "$run_dir/exit_code.txt"
nvidia-smi -i "$gpu_id" > "$run_dir/nvidia-smi_after.txt"
date --iso-8601=seconds > "$run_dir/结束时间.txt"
if [ "$code" -eq 0 ] && [ -s "$run_dir/outputs/结果.json" ]; then printf '%s\n' "成功" > "$run_dir/status.txt"; else printf '%s\n' "失败，退出码=$code" > "$run_dir/status.txt"; fi
rm -f "$lock"
printf 'job_id=%s\nGPU=%s\nPID=%s\n运行目录=%s\n退出码=%s\n' "$job_id" "$gpu_id" "$$" "$run_dir" "$code"
exit "$code"
