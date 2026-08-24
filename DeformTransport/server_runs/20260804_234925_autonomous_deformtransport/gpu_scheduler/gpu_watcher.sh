#!/usr/bin/env bash
set -eo pipefail
run_id="20260804_234925_autonomous_deformtransport"
run_root="/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/$run_id"
root="$run_root/gpu_scheduler"
pidfile="/tmp/deformtransport_gpu_watcher_${run_id}.pid"
if [ -f "$pidfile" ]; then old_pid="$(sed -n '1p' "$pidfile")"; if kill -0 "$old_pid" 2>/dev/null; then echo "已有watcher PID=$old_pid" >&2; exit 73; fi; fi
echo "$$" > "$pidfile"; trap 'rm -f "$pidfile"' EXIT
samples="$root/gpu_samples.csv"; processes="$root/gpu_processes.csv"
[ -s "$samples" ] || echo '采样序号,时间,GPU编号,UUID,利用率百分比,显存已用MiB,显存空闲MiB,温度C,功耗W,功耗上限W,ECC不可纠正错误,系统可用内存KiB' > "$samples"
[ -s "$processes" ] || echo '采样序号,时间,GPU_UUID,PID,用户,进程名,显存MiB' > "$processes"
sample_id="$(awk -F, 'NR>1 && $1+0>max {max=$1+0} END {print max+0}' "$samples")"
while true; do
  sample_id=$((sample_id+1)); timestamp="$(date --iso-8601=seconds)"; mem_available="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw,power.limit,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits | while IFS= read -r line; do echo "$sample_id,$timestamp,$line,$mem_available" >> "$samples"; done
  process_rows="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits)"
  if [ -n "$process_rows" ]; then
    while IFS=',' read -r uuid pid process_name used_memory; do pid="${pid// /}"; user="$(ps -o user= -p "$pid" | awk '{$1=$1;print}')"; echo "$sample_id,$timestamp,$uuid,$pid,$user,$process_name,$used_memory" >> "$processes"; done <<< "$process_rows"
  fi
  if [ ! -e "$root/first_shared_smoke_success.flag" ] && [ ! -e "$root/first_shared_smoke_inflight.flag" ]; then
    /mnt/sdbd/home/liuyu_qyh/tools/miniforge3/bin/python "$root/analyze_shareability.py" >/dev/null 2>&1 || true
    candidate="$(/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); print(next((g for g in "0123" if d["GPU"][g]["状态"]=="SHAREABLE"),""))' "$root/shareability_latest.json" 2>/dev/null || true)"
    if [ -n "$candidate" ]; then
      (
        touch "$root/first_shared_smoke_inflight.flag"
        set +e
        "$run_root/04_smoke/transport_gpu_validation_queued/launch_shareable.sh" "$candidate" >> "$root/auto_launch.log" 2>&1
        code="$?"
        printf '%s,%s,GPU%s,%s\n' "$(date --iso-8601=seconds)" "DT_TRANSPORT_GPU_PARITY" "$candidate" "$code" >> "$root/auto_launch_history.csv"
        if [ "$code" -eq 0 ]; then touch "$root/first_shared_smoke_success.flag"; fi
        rm -f "$root/first_shared_smoke_inflight.flag"
      ) &
      printf '%s\n' "$!" > "$root/auto_launch_pid.txt"
    fi
  fi
  sleep 5
done
