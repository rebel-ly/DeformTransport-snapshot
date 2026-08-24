#!/usr/bin/env bash
set -u

RUN="$1"

DT=/workspace/DeformTransport
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python

EVALDIR=/workspace/DeformTransport/server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval
EVALPY=$EVALDIR/eval_v3.py
REFAPP=$EVALDIR/appearance_report.json

SUITE=/workspace/DeformTransport/server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0


echo "[$(date -Iseconds)] PHASE01 START" \
| tee "$RUN/phase01_status.log"


# ------------------------------------------------------------
# CPU appearance + reproduction gate.
# ------------------------------------------------------------

"$PY" "$RUN/phase01_final.py" \
  --mode appearance \
  --root "$DT" \
  --suite "$SUITE" \
  --eval "$EVALPY" \
  --ref "$REFAPP" \
  --out "$RUN" \
  > "$RUN/appearance_stdout.log" \
  2> "$RUN/appearance_stderr.log" &

P_APP=$!

echo "$P_APP" \
> "$RUN/appearance_pid.txt"


# ------------------------------------------------------------
# CPU RealWonder / region contract audit.
# ------------------------------------------------------------

"$RUN/audit_contracts.sh" "$RUN" \
  > "$RUN/audit_stdout.log" \
  2> "$RUN/audit_stderr.log" &

P_AUDIT=$!

echo "$P_AUDIT" \
> "$RUN/audit_pid.txt"


# ------------------------------------------------------------
# GPU1 safety gate.
# ------------------------------------------------------------

read GPU_MEM GPU_UTIL <<< "$(
    nvidia-smi \
      -i 1 \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
    | tr ',' ' '
)"

GPU_MEM=$(echo "$GPU_MEM" | xargs)
GPU_UTIL=$(echo "$GPU_UTIL" | xargs)

echo \
"[$(date -Iseconds)] GPU1 precheck mem=${GPU_MEM}MiB util=${GPU_UTIL}%" \
| tee -a "$RUN/phase01_status.log"


if \
  [ "$GPU_MEM" -lt 1000 ] \
  && \
  [ "$GPU_UTIL" -le 10 ]
then

    CUDA_VISIBLE_DEVICES=1 \
    "$PY" "$RUN/phase01_final.py" \
      --mode motion \
      --root "$DT" \
      --suite "$SUITE" \
      --eval "$EVALPY" \
      --out "$RUN" \
      --batch 4 \
      > "$RUN/motion_stdout.log" \
      2> "$RUN/motion_stderr.log" &

    P_MOTION=$!

    echo "$P_MOTION" \
    > "$RUN/motion_pid.txt"

    echo \
    "[$(date -Iseconds)] GPU1 RAFT START pid=$P_MOTION" \
    | tee -a "$RUN/phase01_status.log"

else

    P_MOTION=""

    echo "GPU1_BUSY" \
    > "$RUN/MOTION_DEFERRED_GPU_BUSY.txt"

    echo \
    "[$(date -Iseconds)] GPU1 busy: motion deferred" \
    | tee -a "$RUN/phase01_status.log"
fi


# ------------------------------------------------------------
# Wait.
# ------------------------------------------------------------

wait "$P_APP"
E_APP=$?

echo "$E_APP" \
> "$RUN/appearance_exit_code.txt"


wait "$P_AUDIT"
E_AUDIT=$?

echo "$E_AUDIT" \
> "$RUN/audit_exit_code.txt"


if [ -n "$P_MOTION" ]; then
    wait "$P_MOTION"
    E_MOTION=$?
else
    E_MOTION=88
fi

echo "$E_MOTION" \
> "$RUN/motion_exit_code.txt"


if \
  [ "$E_APP" -eq 0 ] \
  && \
  [ "$E_AUDIT" -eq 0 ] \
  && \
  [ "$E_MOTION" -eq 0 ]
then

    date -Iseconds \
    > "$RUN/PHASE01_DONE.txt"

    echo \
    "[$(date -Iseconds)] PHASE01_DONE" \
    | tee -a "$RUN/phase01_status.log"

    exit 0
fi


date -Iseconds \
> "$RUN/PHASE01_PARTIAL_OR_FAILED.txt"

echo \
"[$(date -Iseconds)] PHASE01_END app=$E_APP audit=$E_AUDIT motion=$E_MOTION" \
| tee -a "$RUN/phase01_status.log"

exit 1
