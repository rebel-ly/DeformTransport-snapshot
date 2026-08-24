#!/usr/bin/env bash

set +e

DT=/workspace/DeformTransport
RUN=$(cat "$DT/server_runs/diagnostics/current_final_motion_eval.txt")
SUITE="$RUN/suite"
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
WRAP="$RUN/eval_motion_frozen_wrapper.py"

echo "START tree $(date -Iseconds)" | tee "$RUN/queue.log"

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/workspace/Wan-Move \
"$PY" "$WRAP" \
  --root "$DT" \
  --suite "$SUITE" \
  --out "$RUN/tree" \
  --mode motion \
  --case tree \
  --batch 8 \
  > "$RUN/tree/stdout.log" \
  2> "$RUN/tree/stderr.log"

E1=$?
echo "$E1" > "$RUN/tree/exit_code.txt"
echo "END tree exit=$E1 $(date -Iseconds)" | tee -a "$RUN/queue.log"

if [ "$E1" -ne 0 ]; then
    echo "STOP: TREE MOTION FAILED" | tee -a "$RUN/queue.log"
    exit "$E1"
fi

echo "START santa $(date -Iseconds)" | tee -a "$RUN/queue.log"

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/workspace/Wan-Move \
"$PY" "$WRAP" \
  --root "$DT" \
  --suite "$SUITE" \
  --out "$RUN/santa" \
  --mode motion \
  --case santa \
  --batch 8 \
  > "$RUN/santa/stdout.log" \
  2> "$RUN/santa/stderr.log"

E2=$?
echo "$E2" > "$RUN/santa/exit_code.txt"
echo "END santa exit=$E2 $(date -Iseconds)" | tee -a "$RUN/queue.log"

if [ "$E2" -eq 0 ]; then
    touch "$RUN/SERIAL_QUEUE_DONE.txt"
fi

exit "$E2"
