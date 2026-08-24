#!/usr/bin/env bash
set -euo pipefail
root=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke
chain=$root/OFFICIAL_SANTA_81F_CHAIN_20260805_050719
job=$root/OFFICIAL_SANTA_81F_TRIO_20260805_050719
pid=$(cat "$chain/pid.txt")
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
test -f "$chain/exit_code.txt"
test "$(cat "$chain/exit_code.txt")" = 0
bash "$job/command.sh" > "$job/stdout.log" 2> "$job/stderr.log"
