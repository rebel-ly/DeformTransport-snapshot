#!/usr/bin/env bash

set -u

cd /workspace/DeformTransport

RUN="/workspace/DeformTransport/server_runs/sand_house_cached_sim_20260808_174356"

GEN_PY="/workspace/tools/miniforge3/envs/realwonder-gen/bin/python"

FINAL="$RUN/v1_frozen/20260808_201448__sand_house_final_sim_seed0"

CKPT="/workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"

ARTA="$RUN/v2_dev/20260808_225046__frozen_v2_ab_artifacts/sand_house_v2A_natural_adaptive.pt"

ARTB="$RUN/v2_dev/20260808_225046__frozen_v2_ab_artifacts/sand_house_v2B_energy_controlled_adaptive.pt"


run_one () {

    LABEL="$1"
    MODE="$2"
    ART="$3"
    VIDEO_NAME="$4"

    STAMP=$(date +%Y%m%d_%H%M%S)

    OUT="$RUN/v2_dev/${STAMP}__${LABEL}"

    mkdir -p "$OUT"

    echo "$LABEL" \
    > "$OUT/variant.txt"

    echo "$MODE" \
    > "$OUT/transport_mode.txt"

    echo "$ART" \
    > "$OUT/transport_artifact.txt"

    git rev-parse HEAD \
    > "$OUT/git_head.txt"

    git status --short \
    > "$OUT/git_status.txt"

    sha256sum \
    "$ART" \
    "$FINAL/noises.npy" \
    "$FINAL/config.yaml" \
    "$CKPT" \
    > "$OUT/input_sha256.txt"

    date -Iseconds \
    > "$OUT/start_time.txt"

    echo
    echo "================================================"
    echo "START: $LABEL"
    echo "OUT=$OUT"
    echo "ART=$ART"
    echo "MODE=$MODE"
    echo "================================================"

    CUDA_VISIBLE_DEVICES=2 \
    "$GEN_PY" -u \
    infer_sim.py \
    --checkpoint_path "$CKPT" \
    --sim_data_path "$FINAL" \
    --output_path "$OUT/$VIDEO_NAME" \
    --seed 0 \
    --eval_degradation 0.5 \
    --local_attn_size 21 \
    --transport_latent_path "$ART" \
    --transport_mode "$MODE" \
    --transport_injection_mode condition_residual \
    --transport_injection_scale 1.0 \
    > "$OUT/stdout.log" \
    2> "$OUT/stderr.log"

    CODE=$?

    echo "$CODE" \
    > "$OUT/exit_code.txt"

    date -Iseconds \
    > "$OUT/end_time.txt"

    echo
    echo "===== RESULT: $LABEL ====="
    echo "OUT=$OUT"
    echo "exit_code=$CODE"

    grep -E \
    'Loading (correct|shuffled) artifact-local|transport residual scale|transport residual mean_abs|transport residual max_abs|requested condition delta mean_abs|applied condition delta mean_abs|applied condition delta max_abs|applied condition slot0|max_abs|Saving video|Done!' \
    "$OUT/stdout.log" \
    || true

    if [ "$CODE" -ne 0 ]; then

        echo
        echo "===== STDERR ====="

        tail -160 \
        "$OUT/stderr.log"

        return "$CODE"

    fi

    sha256sum \
    "$OUT/$VIDEO_NAME" \
    > "$OUT/output_sha256.txt"

    echo
    echo "SUCCESS: $LABEL"

    return 0
}


# ------------------------------------------------------------
# 1. Highest priority:
# close the V2-B Correct/Shuffled causal pair.
# ------------------------------------------------------------

run_one \
"sand_house_v2B_shuffled_seed0" \
"shuffled" \
"$ARTB" \
"sand_house_v2B_shuffled_seed0.mp4" \
|| exit $?


# ------------------------------------------------------------
# 2. Natural adaptive Correct.
# ------------------------------------------------------------

run_one \
"sand_house_v2A_correct_seed0" \
"correct" \
"$ARTA" \
"sand_house_v2A_correct_seed0.mp4" \
|| exit $?


# ------------------------------------------------------------
# 3. Natural adaptive Shuffled.
# ------------------------------------------------------------

run_one \
"sand_house_v2A_shuffled_seed0" \
"shuffled" \
"$ARTA" \
"sand_house_v2A_shuffled_seed0.mp4" \
|| exit $?


echo
echo "================================================"
echo "ALL_REMAINING_V2_GENERATIONS_OK"
echo "================================================"
