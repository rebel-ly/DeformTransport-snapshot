#!/usr/bin/env bash
set -u
R=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_000000__phase0d_4d_r2_contract_correction
O="$R/begin0_gpu2"
mkdir -p "$O"
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/start_time_utc.txt"
bash "$R/run_begin0.sh" "$O" 2 > "$O/stdout.log" 2> "$O/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$O/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$O/end_time_utc.txt"
if [ "$rc" -eq 0 ]; then touch "$O/completion.marker"; fi
exit "$rc"
