#!/usr/bin/env bash
# Detached entrypoint for the authorized A2 formal rerun. It does not alter the
# canonical launcher or model source; it only records launch identity.
set -euo pipefail
base=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/corrected_parity_20260814
out="$base/a2_gpu0_formal_rerun"
mkdir -p "$out"
printf '%s\n' "$$" > "$out/container_entry_pid.txt"
ps -o ppid= -p "$$" | tr -d ' ' > "$out/container_entry_ppid.txt"
exec "$base/run_detached_arm.sh" original "$out"
