#!/usr/bin/env bash
set -u
R=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation;O="$R/c2_gpu0";mkdir -p "$O";date -u +%Y-%m-%dT%H:%M:%SZ > "$O/start_time_utc.txt";bash "$R/run_c2.sh" "$O" 0 > "$O/stdout.log" 2> "$O/stderr.log";rc=$?;echo "$rc" > "$O/exit_code.txt";date -u +%Y-%m-%dT%H:%M:%SZ > "$O/end_time_utc.txt";[ "$rc" -eq 0 ] && touch "$O/completion.marker";exit "$rc"
