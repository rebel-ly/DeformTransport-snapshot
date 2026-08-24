#!/usr/bin/env bash
set -u

cd /workspace/DeformTransport

RUN="/workspace/DeformTransport/server_runs/sand_house_cached_sim_20260808_174356"
PY="/workspace/tools/miniforge3/envs/realwonder-gen/bin/python"

FINAL="$RUN/v1_frozen/20260808_201448__sand_house_final_sim_seed0"

CKPT="/workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"

CTD="$RUN/ctd_dev/20260808_235818__counterfactual_transport_debiasing"

run_one () {
    NAME="$1"
    ART="$2"

    STAMP=$(date +%Y%m%d_%H%M%S)
    OUT="$CTD/${STAMP}__${NAME}_seed0"

    mkdir -p "$OUT"

    git rev-parse HEAD > "$OUT/git_head.txt"
    git status --short > "$OUT/git_status.txt"

    sha256sum \
        "$ART" \
        "$FINAL/noises.npy" \
        "$FINAL/config.yaml" \
        "$CKPT" \
        > "$OUT/input_sha256.txt"

    date -Iseconds > "$OUT/start_time.txt"

    CUDA_VISIBLE_DEVICES=2 \
    "$PY" -u infer_sim.py \
        --checkpoint_path "$CKPT" \
        --sim_data_path "$FINAL" \
        --output_path "$OUT/${NAME}_seed0.mp4" \
        --seed 0 \
        --eval_degradation 0.5 \
        --local_attn_size 21 \
        --transport_latent_path "$ART" \
        --transport_mode correct \
        --transport_injection_mode condition_residual \
        --transport_injection_scale 1.0 \
        > "$OUT/stdout.log" \
        2> "$OUT/stderr.log"

    CODE=$?

    echo "$CODE" > "$OUT/exit_code.txt"
    date -Iseconds > "$OUT/end_time.txt"

    echo
    echo "===== $NAME ====="
    echo "OUT=$OUT"
    echo "exit_code=$CODE"

    grep -E \
    'Loading correct artifact-local|transport residual scale|transport residual mean_abs|transport residual max_abs|requested condition delta mean_abs|applied condition delta mean_abs|applied condition delta max_abs|applied condition slot0|Saving video|Done!' \
    "$OUT/stdout.log" || true

    if [ "$CODE" -ne 0 ]; then
        tail -160 "$OUT/stderr.log"
        return "$CODE"
    fi

    sha256sum \
        "$OUT/${NAME}_seed0.mp4" \
        > "$OUT/output_sha256.txt"

    return 0
}

run_one \
    "sand_house_ctd_diff" \
    "$CTD/sand_house_ctd_diff.pt" \
    || exit $?

run_one \
    "sand_house_ctd_orth" \
    "$CTD/sand_house_ctd_orth.pt" \
    || exit $?

echo
echo "CTD_BOTH_GENERATIONS_OK"
