#!/usr/bin/env bash
# Isolated symbolic binding for the frozen corrected-v2 evaluator.
set -euo pipefail
name="${1:?name}"; video="${2:?video}"
R=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_095925__dt_full_5seed_drop_zero62_formal_evaluation
E=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py
P=/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/envs/wan-move/bin/python
ROOT=/mnt/sdbd/home/liuyu_qyh/DeformTransport
D="$R/candidates/$name"; S="$D/suite"; O="$D/out"
mkdir -p "$S/santa/dt_full" "$O"
ln -sfn "$video" "$S/santa/dt_full/santa_dt_full_correct_seed0.mp4"
sha256sum "$video" > "$O/bound_candidate_sha256.txt"
CUDA_VISIBLE_DEVICES=1 "$P" "$E" --root "$ROOT" --suite "$S" --out "$O" --mode appearance > "$O/appearance.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$P" "$E" --root "$ROOT" --suite "$S" --out "$O" --mode motion --case santa --batch 8 > "$O/motion.log" 2>&1
