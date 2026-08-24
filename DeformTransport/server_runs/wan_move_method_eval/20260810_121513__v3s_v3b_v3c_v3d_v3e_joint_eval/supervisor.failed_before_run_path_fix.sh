#!/usr/bin/env bash

set -u

DT=/workspace/DeformTransport

PY=/workspace/tools/miniforge3/envs/wan-move/bin/python

RUN="$(
    cd "$(
        dirname "$0"
    )"
    &&
    pwd
)"

SUITE=/workspace/DeformTransport/server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0


echo "[$(date -Iseconds)] START joint evaluation"


# ------------------------------------------------------------
# CPU appearance
# ------------------------------------------------------------

"$PY" \
"$RUN/eval_v3.py" \
--root "$DT" \
--suite "$SUITE" \
--out "$RUN" \
--mode appearance \
> "$RUN/appearance_stdout.log" \
2> "$RUN/appearance_stderr.log" &

P1=$!


# ------------------------------------------------------------
# GPU1 Santa motion
# ------------------------------------------------------------

CUDA_VISIBLE_DEVICES=1 \
"$PY" \
"$RUN/eval_v3.py" \
--root "$DT" \
--suite "$SUITE" \
--out "$RUN" \
--mode motion \
--case santa \
--batch 8 \
> "$RUN/santa_motion_stdout.log" \
2> "$RUN/santa_motion_stderr.log" &

P2=$!


# ------------------------------------------------------------
# GPU2 Tree motion
# ------------------------------------------------------------

CUDA_VISIBLE_DEVICES=2 \
"$PY" \
"$RUN/eval_v3.py" \
--root "$DT" \
--suite "$SUITE" \
--out "$RUN" \
--mode motion \
--case tree \
--batch 8 \
> "$RUN/tree_motion_stdout.log" \
2> "$RUN/tree_motion_stderr.log" &

P3=$!


printf '%s\n' "$P1" \
> "$RUN/appearance_pid.txt"

printf '%s\n' "$P2" \
> "$RUN/santa_motion_pid.txt"

printf '%s\n' "$P3" \
> "$RUN/tree_motion_pid.txt"


# ------------------------------------------------------------
# Wait for all three.
# ------------------------------------------------------------

wait "$P1"
E1=$?

wait "$P2"
E2=$?

wait "$P3"
E3=$?


echo "$E1" \
> "$RUN/appearance_exit_code.txt"

echo "$E2" \
> "$RUN/santa_motion_exit_code.txt"

echo "$E3" \
> "$RUN/tree_motion_exit_code.txt"


# ------------------------------------------------------------
# Frozen selector
# ------------------------------------------------------------

if \
    [ "$E1" -eq 0 ] \
    && \
    [ "$E2" -eq 0 ] \
    && \
    [ "$E3" -eq 0 ]
then

    "$PY" \
    "$RUN/select_v3.py" \
    "$RUN" \
    > "$RUN/selection_stdout.log" \
    2> "$RUN/selection_stderr.log"

    ES=$?

    echo "$ES" \
    > "$RUN/selection_exit_code.txt"

    if [ "$ES" -eq 0 ]
    then

        date -Iseconds \
        > "$RUN/EVAL_DONE.txt"

        echo \
        "[$(date -Iseconds)] EVAL_DONE"

        cat \
        "$RUN/selection_stdout.log"

        exit 0
    fi
fi


date -Iseconds \
> "$RUN/EVAL_FAILED.txt"

echo \
"[$(date -Iseconds)] EVAL_FAILED appearance=$E1 santa_motion=$E2 tree_motion=$E3"

exit 1
