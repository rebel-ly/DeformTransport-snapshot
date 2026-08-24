#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "用法: monitor_owned_process.sh PID GPU编号 日志路径" >&2; exit 64; fi
pid="$1"
gpu_id="$2"
out="$3"
while kill -0 "$pid" 2>/dev/null; do
  {
    date --iso-8601=seconds
    ps -p "$pid" -o pid,user,lstart,etime,state,pcpu,pmem,rss,args --cols 4096
    nvidia-smi -i "$gpu_id" --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
  } >> "$out"
  sleep 5
done
