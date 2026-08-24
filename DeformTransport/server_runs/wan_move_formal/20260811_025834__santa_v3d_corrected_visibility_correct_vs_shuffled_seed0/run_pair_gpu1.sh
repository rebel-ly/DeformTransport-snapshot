#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"

DT=/workspace/DeformTransport
WAN=/workspace/Wan-Move
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
WANENV=/workspace/tools/miniforge3/envs/wan-move

SUITE=$DT/server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0
BRIDGE=$DT/server_runs/wan_move_bridge/20260811_024330__santa_corrected_physical_visibility

IMAGE=$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png
PROMPT_FILE=$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt

VIS=$BRIDGE/santa_material_visibility_correct.npy
IDS=$ROOT/artifacts/santa_corrected_ids.npy
DEPTH=$ROOT/artifacts/santa_corrected_depth.npy

export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME="$WANENV"
export CUDA_PATH="$WANENV"
export PATH="$WANENV/bin:$PATH"
export LD_LIBRARY_PATH="$WANENV/targets/x86_64-linux/lib:$WANENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

export DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$IDS"
export DT_TRACK_DEPTH_PATH="$DEPTH"

cd "$WAN"

sha256sum \
wan/wan_move.py \
wan/modules/trajectory.py \
> "$ROOT/source_sha256_at_launch.txt"

if ! cmp -s \
"$SUITE/installed_source_sha256.txt" \
"$ROOT/source_sha256_at_launch.txt"
then
    echo "STOP: SOURCE HASH DOES NOT MATCH ORIGINAL V3 SUITE"
    exit 71
fi

PROMPT=$(cat "$PROMPT_FILE")

for MODE in correct shuffled
do
    OUT="$ROOT/$MODE"
    mkdir -p "$OUT"

    if [ "$MODE" = correct ]; then
        TRACK=$BRIDGE/santa_material_tracks_correct.npy
        SAVE=$OUT/santa_v3d_corrected_visibility_correct_seed0.mp4
    else
        TRACK=$BRIDGE/santa_material_tracks_identity_shuffled_seed0.npy
        SAVE=$OUT/santa_v3d_corrected_visibility_shuffled_seed0.mp4
    fi

    echo "============================================================"
    echo "START $MODE $(date -Iseconds)"
    echo "============================================================"

    date -Iseconds > "$OUT/start_time.txt"

    cat > "$OUT/contract.txt" <<EOT
mode=$MODE
method=V3D
seed=0
gpu_physical=1
track=$TRACK
visibility=$VIS
ids=$IDS
depth=$DEPTH
image=$IMAGE
prompt=$PROMPT_FILE
sample_solver=unipc
steps=40
shift=3.0
guide_scale=5.0
corrected_physical_visibility=true
EOT

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
      --sample_solver unipc \
      --sample_steps 40 \
      --sample_shift 3.0 \
      --sample_guide_scale 5.0 \
      --t5_cpu \
      --offload_model True \
      --dtype bf16 \
      --save_file "$SAVE" \
      > "$OUT/stdout.log" \
      2> "$OUT/stderr.log"

    EC=$?

    set -e

    echo "$EC" > "$OUT/exit_code.txt"
    date -Iseconds > "$OUT/end_time.txt"

    echo "RESULT $MODE exit=$EC"

    if [ "$EC" -ne 0 ]; then
        echo "STOP: $MODE FAILED"
        exit "$EC"
    fi

    test -s "$SAVE" || {
        echo "STOP: MP4 MISSING"
        exit 72
    }

    sha256sum "$SAVE" \
      > "$OUT/output_sha256.txt"

    sleep 20
done

date -Iseconds > "$ROOT/SERIAL_QUEUE_DONE.txt"

echo "ALL CORRECTED SANTA V3D RUNS FINISHED"
