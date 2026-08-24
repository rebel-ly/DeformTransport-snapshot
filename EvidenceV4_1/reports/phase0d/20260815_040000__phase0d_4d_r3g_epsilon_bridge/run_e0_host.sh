#!/usr/bin/env bash
set -u
R=/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_040000__phase0d_4d_r3g_epsilon_bridge
mkdir -p "$R/e0_gpu2"
docker exec --user 10011:10011 --workdir /workspace -e HOME=/workspace -e CUDA_VISIBLE_DEVICES=2 deformtransport-dev bash /workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_040000__phase0d_4d_r3g_epsilon_bridge/e0_inside.sh
