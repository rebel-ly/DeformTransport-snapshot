#!/usr/bin/env bash
set -euo pipefail

CASE="${1:?case santa|tree}"
VAR="${2:?variant}"
GPU="${3:?gpu}"

DT=/workspace/DeformTransport
WAN=/workspace/Wan-Move
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
WANENV=/workspace/tools/miniforge3/envs/wan-move

V3=/workspace/DeformTransport/server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0
V4=$(cat "$DT/server_runs/wan_move_method_dev/current_v4_wave1.txt")

case "$VAR" in
    v4a_d20|v4a_d40|v4b_noise|v4c_hybrid)
        ;;
    *)
        echo "UNKNOWN V4 VARIANT $VAR"
        exit 50
        ;;
esac

ART="$V3/artifacts/$CASE"
OUT="$V4/$CASE/$VAR"

mkdir -p "$OUT"

if [ "$CASE" = "santa" ]; then

    IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"

    PROMPT_FILE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt"

    TRACK="$DT/server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_tracks_correct.npy"

    VIS="$DT/server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_visibility_correct.npy"

else

    IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/resized_input_image.png"

    PROMPT_FILE="$DT/server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0/prompt.txt"

    TRACK="$DT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_tracks_correct.npy"

    VIS="$DT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_visibility_correct.npy"
fi

IDS="$ART/${CASE}_old_correct_ids.npy"
DEPTH="$ART/${CASE}_old_correct_depth.npy"

for p in \
    "$IMAGE" \
    "$PROMPT_FILE" \
    "$TRACK" \
    "$VIS" \
    "$IDS" \
    "$DEPTH"
do
    test -f "$p" || {
        echo "MISSING $p"
        exit 51
    }
done

export CUDA_VISIBLE_DEVICES="$GPU"

export CUDA_HOME="$WANENV"
export CUDA_PATH="$WANENV"

export PATH="$WANENV/bin:$PATH"

export LD_LIBRARY_PATH="$WANENV/targets/x86_64-linux/lib:$WANENV/lib:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1

# Frozen V3D condition transport.
export DT_TRANSPORT_VARIANT=v3d
export DT_TRACK_IDS_PATH="$IDS"
export DT_TRACK_DEPTH_PATH="$DEPTH"

# New architecture intervention.
export DT_V4_VARIANT="$VAR"

printf '%s\n' "$BASHPID" \
    > "$OUT/pid.txt"

date -Iseconds \
    > "$OUT/start_time.txt"

cat > "$OUT/contract.txt" <<EOF
case=$CASE
variant=$VAR
gpu=$GPU

base_transport=v3d
v4_variant=$VAR

image=$IMAGE
prompt=$PROMPT_FILE
track=$TRACK
visibility=$VIS
ids=$IDS
depth=$DEPTH

seed=0
frame_num=81
solver=unipc
steps=40
shift=3.0
guide_scale=5.0
dtype=bf16
offload_model=True

v4a_d20_steps=8
v4a_d40_steps=16
v4b_noise_mix=0.2
v4b_noise_patch_radius=1
v4c_hybrid=d20_plus_noise
EOF

PROMPT=$(cat "$PROMPT_FILE")

VIDEO="$OUT/${CASE}_${VAR}_correct_seed0.mp4"

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
    --sample_solver unipc \
    --sample_steps 40 \
    --sample_shift 3.0 \
    --sample_guide_scale 5.0 \
    --t5_cpu \
    --offload_model True \
    --dtype bf16 \
    --save_file "$VIDEO" \
    > "$OUT/stdout.log" \
    2> "$OUT/stderr.log"

EC=$?

set -e

printf '%s\n' "$EC" \
    > "$OUT/exit_code.txt"

date -Iseconds \
    > "$OUT/end_time.txt"

if [ "$EC" -eq 0 ] && [ -s "$VIDEO" ]; then

    sha256sum "$VIDEO" \
        > "$OUT/output_sha256.txt"

    echo "DONE $CASE $VAR"

else

    echo "FAILED $CASE $VAR EC=$EC"
    tail -60 "$OUT/stderr.log" || true
    exit "$EC"

fi
