#!/usr/bin/env bash
set -euo pipefail


CASE="${1:?case}"
ARM="${2:?correct|shuffled}"
GPU="${3:?gpu}"


DT=/workspace/DeformTransport
WAN=/workspace/Wan-Move
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
WANENV=/workspace/tools/miniforge3/envs/wan-move


RUN=$(cat \
"$DT/server_runs/wan_move_method_dev/current_v3d_formal_validation.txt")

ART="$RUN/artifacts"


if [ "$CASE" = "santa" ]; then

    test "$ARM" = "shuffled" || {
        echo "Santa new validation only requires shuffled."
        exit 31
    }

    IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"

    PROMPT_FILE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt"

    TRACK="$ART/santa/santa_v3d_identity_shuffled_tracks.npy"
    VIS="$ART/santa/santa_v3d_visibility.npy"
    IDS="$ART/santa/santa_v3d_ids.npy"
    DEPTH="$ART/santa/santa_v3d_depth.npy"

    OUT="$RUN/santa_shuffled"
    VIDEO="$OUT/santa_v3d_identity_shuffled_seed0.mp4"


elif [ "$CASE" = "tree" ]; then

    test "$ARM" = "shuffled" || {
        echo "Tree new validation only requires shuffled."
        exit 32
    }

    IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/resized_input_image.png"

    PROMPT_FILE="$DT/server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0/prompt.txt"

    TRACK="$ART/tree/tree_v3d_identity_shuffled_tracks.npy"
    VIS="$ART/tree/tree_v3d_visibility.npy"
    IDS="$ART/tree/tree_v3d_ids.npy"
    DEPTH="$ART/tree/tree_v3d_depth.npy"

    OUT="$RUN/tree_shuffled"
    VIDEO="$OUT/tree_v3d_identity_shuffled_seed0.mp4"


elif [ "$CASE" = "sandhouse" ]; then

    IMAGE=$(cat \
    "$ART/sandhouse/input_image_path.txt")

    PROMPT_FILE=$(cat \
    "$ART/sandhouse/prompt_path.txt")

    VIS="$ART/sandhouse/sandhouse_v3d_visibility.npy"
    IDS="$ART/sandhouse/sandhouse_v3d_ids.npy"
    DEPTH="$ART/sandhouse/sandhouse_v3d_depth.npy"

    if [ "$ARM" = "correct" ]; then

        TRACK="$ART/sandhouse/sandhouse_v3d_correct_tracks.npy"

        OUT="$RUN/sandhouse_correct"

        VIDEO="$OUT/sandhouse_v3d_correct_seed0.mp4"

    elif [ "$ARM" = "shuffled" ]; then

        TRACK="$ART/sandhouse/sandhouse_v3d_identity_shuffled_tracks.npy"

        OUT="$RUN/sandhouse_shuffled"

        VIDEO="$OUT/sandhouse_v3d_identity_shuffled_seed0.mp4"

    else

        echo "Unknown SandHouse arm: $ARM"
        exit 33
    fi


else

    echo "Unknown case: $CASE"
    exit 34
fi


mkdir -p "$OUT"


for p in \
"$IMAGE" \
"$PROMPT_FILE" \
"$TRACK" \
"$VIS" \
"$IDS" \
"$DEPTH"
do

    test -f "$p" || {
        echo "MISSING: $p"
        exit 40
    }

done


# ============================================================
# Strict runtime frozen-source check before loading 14B model
# ============================================================

cd "$WAN"

sha256sum -c \
"$DT/server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0/installed_source_sha256.txt"


export CUDA_VISIBLE_DEVICES="$GPU"

export CUDA_HOME="$WANENV"
export CUDA_PATH="$WANENV"

export PATH="$WANENV/bin:$PATH"

export LD_LIBRARY_PATH="$WANENV/targets/x86_64-linux/lib:$WANENV/lib:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1

export PYTHONPATH="$WAN:${PYTHONPATH:-}"

export DT_TRANSPORT_VARIANT="v3d"
export DT_TRACK_IDS_PATH="$IDS"
export DT_TRACK_DEPTH_PATH="$DEPTH"


PROMPT=$(cat "$PROMPT_FILE")


date -Iseconds \
> "$OUT/start_time.txt"

echo "$BASHPID" \
> "$OUT/pid.txt"


cat > "$OUT/contract.txt" <<EOF
case=$CASE
arm=$ARM
method=V3D
gpu=$GPU

image=$IMAGE
prompt=$PROMPT_FILE
track=$TRACK
visibility=$VIS
ids=$IDS
depth=$DEPTH

DT_TRANSPORT_VARIANT=v3d

seed=0
frame_num=81
sample_steps=40
sample_shift=3.0
dtype=bf16
t5_cpu=True
offload_model=True
EOF


sha256sum \
"$IMAGE" \
"$PROMPT_FILE" \
"$TRACK" \
"$VIS" \
"$IDS" \
"$DEPTH" \
> "$OUT/input_sha256.txt"


set +e

"$PY" generate.py \
  --task wan-move-i2v \
  --size '480*832' \
  --frame_num 81 \
  --ckpt_dir "$WAN/Wan-Move-14B-480P" \
  --image "$IMAGE" \
  --track "$TRACK" \
  --track_visibility "$VIS" \
  --prompt "$PROMPT" \
  --base_seed 0 \
  --sample_steps 40 \
  --sample_shift 3.0 \
  --t5_cpu \
  --offload_model True \
  --dtype bf16 \
  --save_file "$VIDEO" \
  > "$OUT/stdout.log" \
  2> "$OUT/stderr.log"

EC=$?

set -e


echo "$EC" \
> "$OUT/exit_code.txt"

date -Iseconds \
> "$OUT/end_time.txt"


if [ "$EC" -ne 0 ]; then

    echo \
    "FAILED case=$CASE arm=$ARM ec=$EC" \
    >&2

    exit "$EC"
fi


test -s "$VIDEO" || {
    echo "Generation exited 0 but video is missing/empty."
    exit 50
}


sha256sum \
"$VIDEO" \
> "$OUT/output_sha256.txt"


echo \
"DONE case=$CASE arm=$ARM GPU=$GPU"

