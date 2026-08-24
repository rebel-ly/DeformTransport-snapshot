#!/usr/bin/env bash
set -euo pipefail
mode="${1:?mode}"; out="${2:?out}"
base=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/corrected_parity_20260814
mkdir -p "$out"
date -Is > "$out/start_time.txt"
printf 'SOURCE_MODE=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$mode" "${CUDA_VISIBLE_DEVICES:-}" > "$out/environment.txt"
if [ "$mode" = original ]; then source=/workspace/Wan-Move; else source=/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay; fi
PYTHONPATH="$source" /workspace/tools/miniforge3/envs/wan-move/bin/python -c 'import wan,wan.wan_move; print(wan.__file__); print(wan.wan_move.__file__)' > "$out/source_provenance.txt" 2>&1
set +e
"$base/run_single_launcher.sh" "$mode" "$out" > "$out/stdout.log" 2> "$out/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" > "$out/exit_code.txt"
date -Is > "$out/end_time.txt"
if [ "$rc" -eq 0 ]; then printf 'COMPLETE\n' > "$out/completion.marker"; else printf 'FAILED\n' > "$out/completion.marker"; fi
exit "$rc"
