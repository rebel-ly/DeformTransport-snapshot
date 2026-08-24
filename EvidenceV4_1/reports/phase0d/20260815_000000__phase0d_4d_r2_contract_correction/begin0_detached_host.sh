#!/usr/bin/env bash
set -u
R=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_000000__phase0d_4d_r2_contract_correction
O="$R/begin0_gpu2"
mkdir -p "$O"
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/start_time_utc.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,temperature.gpu --format=csv,noheader > "$O/prelaunch_gpu_snapshot.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$O/prelaunch_compute_processes.csv" || true
docker exec --user 10011:10011 --workdir /workspace -e HOME=/workspace -e CUDA_VISIBLE_DEVICES=2 deformtransport-dev bash /workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_000000__phase0d_4d_r2_contract_correction/run_begin0.sh /workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_000000__phase0d_4d_r2_contract_correction/begin0_gpu2 2 > "$O/stdout.log" 2> "$O/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$O/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/end_time_utc.txt"
if [ "$rc" -eq 0 ]; then touch "$O/completion.marker"; fi
exit "$rc"
