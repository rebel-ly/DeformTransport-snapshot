#!/usr/bin/env bash
set -euo pipefail

CASE="${1:?usage: run_one.sh santa|tree v3s|v3b|v3c|v3d|v3e gpu}"
VAR="${2:?variant}"
GPU="${3:?gpu}"

DT=/workspace/DeformTransport
WAN=/workspace/Wan-Move
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
WANENV=/workspace/tools/miniforge3/envs/wan-move

SUITE=$(cat \
"$DT/server_runs/wan_move_method_dev/current_v3_suite.txt")

ART="$SUITE/artifacts/$CASE"
RUN="$SUITE/$CASE/$VAR"

mkdir -p \
"$RUN"


if [ "$CASE" = "santa" ]; then

    IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"

    PROMPT_FILE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt"

    OLD_TRACK="$DT/server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_tracks_correct.npy"

    OLD_VIS="$DT/server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_visibility_correct.npy"

elif [ "$CASE" = "tree" ]; then

    IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/resized_input_image.png"

    PROMPT_FILE="$DT/server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0/prompt.txt"

    OLD_TRACK="$DT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_tracks_correct.npy"

    OLD_VIS="$DT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_visibility_correct.npy"

else

    echo "Unknown case: $CASE"
    exit 42
fi


if [ "$VAR" = "v3s" ]; then

    TRACK="$ART/${CASE}_v3s_tracks.npy"
    VIS="$ART/${CASE}_v3s_visibility.npy"
    IDS="$ART/${CASE}_v3s_ids.npy"
    DEPTH="$ART/${CASE}_v3s_depth.npy"

    MODE="v3s"

elif [[ "$VAR" =~ ^v3[bcde]$ ]]; then

    TRACK="$OLD_TRACK"
    VIS="$OLD_VIS"

    IDS="$ART/${CASE}_old_correct_ids.npy"
    DEPTH="$ART/${CASE}_old_correct_depth.npy"

    MODE="$VAR"

else

    echo "Unknown variant: $VAR"
    exit 43
fi


for path in \
"$IMAGE" \
"$PROMPT_FILE" \
"$TRACK" \
"$VIS" \
"$IDS" \
"$DEPTH"
do
    test -f "$path" || {
        echo "MISSING: $path"
        exit 41
    }
done


export CUDA_VISIBLE_DEVICES="$GPU"

export CUDA_HOME="$WANENV"
export CUDA_PATH="$WANENV"

export PATH="$WANENV/bin:$PATH"

export LD_LIBRARY_PATH="$WANENV/targets/x86_64-linux/lib:$WANENV/lib:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1

export DT_TRANSPORT_VARIANT="$MODE"
export DT_TRACK_IDS_PATH="$IDS"
export DT_TRACK_DEPTH_PATH="$DEPTH"


printf '%s\n' \
"$BASHPID" \
> "$RUN/pid.txt"

date -Iseconds \
> "$RUN/start_time.txt"


cat > "$RUN/contract.txt" <<EOF
case=$CASE
variant=$VAR
mode=$MODE
gpu=$GPU

track=$TRACK
visibility=$VIS
ids=$IDS
depth=$DEPTH

image=$IMAGE
prompt=$PROMPT_FILE

seed=0
frame_num=81
steps=40
shift=3.0
dtype=bf16
t5_cpu=True
offload_model=True
EOF


PROMPT=$(cat \
"$PROMPT_FILE")


cd "$WAN"


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
  --save_file "$RUN/${CASE}_${VAR}_correct_seed0.mp4" \
  > "$RUN/stdout.log" \
  2> "$RUN/stderr.log"

EC=$?

set -e


printf '%s\n' \
"$EC" \
> "$RUN/exit_code.txt"

date -Iseconds \
> "$RUN/end_time.txt"


if \
    [ "$EC" -eq 0 ] \
    && \
    [ -f "$RUN/${CASE}_${VAR}_correct_seed0.mp4" ]
then

    sha256sum \
    "$RUN/${CASE}_${VAR}_correct_seed0.mp4" \
    > "$RUN/output_sha256.txt"

else

    echo \
    "FAILED case=$CASE variant=$VAR ec=$EC" \
    >&2

    exit "$EC"
fi
