#!/usr/bin/env bash
# Deterministic status checks only; it never launches generation or evaluation.
set -euo pipefail
PHASE=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen
PREVIEW=$PHASE/preview_reconstruction_20260814/PREVIEW_RECONSTRUCTION_MANIFEST.json
test -d /workspace
id
test -f $PHASE/MAINTENANCE_CHECKPOINT_20260814_0410.json
test -f $PREVIEW
sha256sum $PREVIEW
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits | head -2
ps -eo pid,ppid,uid,gid,lstart,args | grep -E '20260814_100000__fast_a_b_screen|Wan-Move.*generate.py' | grep -v grep || true
find $PHASE -type f \( -name '*.mp4' -o -name '*.partial' \) -printf '%p %s %TY-%Tm-%TdT%TH:%TM:%TS\n'
