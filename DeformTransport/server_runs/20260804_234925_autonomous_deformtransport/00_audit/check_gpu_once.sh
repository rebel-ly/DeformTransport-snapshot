#!/usr/bin/env bash
set -euo pipefail
date --iso-8601=seconds
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
