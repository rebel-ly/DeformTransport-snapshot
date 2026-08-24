#!/usr/bin/env bash
# Future Phase0D-2R-C paired replay launcher.  This script does not run until invoked.
set -euo pipefail

RUNNER="/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/seed_contract/20260812_172755__santa_v3d_contract_recovery/formal_run_corrected_v2_v3d.sh"

run_pair() {
    local cmd_a="$1"
    local cmd_b="$2"
    local run_a="$3"
    local run_b="$4"

    mkdir -p "$run_a" "$run_b"
    bash -c "$cmd_a" >"$run_a/stdout.log" 2>"$run_a/stderr.log" &
    local pid_a=$!
    printf '%s\n' "$pid_a" >"$run_a/pid.txt"
    bash -c "$cmd_b" >"$run_b/stdout.log" 2>"$run_b/stderr.log" &
    local pid_b=$!
    printf '%s\n' "$pid_b" >"$run_b/pid.txt"

    set +e
    wait "$pid_a"
    local rc_a=$?
    wait "$pid_b"
    local rc_b=$?
    set -e
    printf '%s\n' "$rc_a" >"$run_a/runA_exit_code.txt"
    printf '%s\n' "$rc_b" >"$run_b/runB_exit_code.txt"
    return 0
}

if [[ "${1:-}" == "--mock" ]]; then
    mock_out="${2:?usage: $0 --mock OUTPUT_DIR}"
    run_pair 'exit 0' 'exit 7' "$mock_out/runA" "$mock_out/runB"
    exit 0
fi

seed="${1:?usage: $0 SEED OUTPUT_DIR}"
out="${2:?usage: $0 SEED OUTPUT_DIR}"
case "$seed" in 0|1|2|3|4) ;; *) echo "SEED must be one of 0 1 2 3 4" >&2; exit 64 ;; esac
run_pair "CUDA_VISIBLE_DEVICES=1 bash '$RUNNER' '$seed' '$out/runA'" \
         "CUDA_VISIBLE_DEVICES=2 bash '$RUNNER' '$seed' '$out/runB'" \
         "$out/runA" "$out/runB"
