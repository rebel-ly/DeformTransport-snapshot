#!/usr/bin/env bash
set -u

V4=/workspace/DeformTransport/server_runs/wan_move_method_v4/20260810_164423__wave1_denoise_noise_hybrid_seed0

run_case () {
    CASE="$1"
    GPU="$2"

    for VAR in v4a_d40 v4b_noise v4c_hybrid
    do
        OUT="$V4/$CASE/$VAR"

        echo
        echo "============================================================"
        echo "START $CASE $VAR GPU$GPU $(date -Iseconds)"
        echo "============================================================"

        # 已成功则跳过，避免重复计算
        if [ -s "$OUT/exit_code.txt" ] && \
           [ "$(cat "$OUT/exit_code.txt")" = "0" ] && \
           find "$OUT" -maxdepth 1 -name '*.mp4' -size +1M | grep -q .
        then
            echo "SKIP_ALREADY_SUCCESS $CASE $VAR"
            continue
        fi

        # D40 之前的 137 记录留档
        if [ -s "$OUT/exit_code.txt" ]; then
            STAMP=$(date +%Y%m%d_%H%M%S)
            mkdir -p "$OUT/failed_${STAMP}"

            for F in exit_code.txt stdout.log stderr.log start_time.txt end_time.txt
            do
                [ -f "$OUT/$F" ] && cp "$OUT/$F" "$OUT/failed_${STAMP}/$F"
            done
        fi

        echo "--- GPU before ---"
        nvidia-smi \
          --query-gpu=index,memory.used,memory.free,utilization.gpu \
          --format=csv,noheader | grep "^$GPU," || true

        set +e

        bash "$V4/run_one.sh" \
          "$CASE" \
          "$VAR" \
          "$GPU"

        EC=$?

        set -e

        echo "RESULT $CASE $VAR exit=$EC"

        echo "--- GPU after ---"
        nvidia-smi \
          --query-gpu=index,memory.used,memory.free,utilization.gpu \
          --format=csv,noheader | grep "^$GPU," || true

        if [ "$EC" -ne 0 ]; then
            echo "FAILED_BUT_CONTINUE $CASE $VAR exit=$EC"
        fi

        sleep 15
    done

    date -Iseconds > "$V4/$CASE/RESUME_NOW_DONE.txt"
    echo "${CASE^^}_RESUME_NOW_COMPLETE"
}

run_case santa 1 \
  > "$V4/santa/resume_now.log" \
  2>&1 &

SPID=$!

run_case tree 2 \
  > "$V4/tree/resume_now.log" \
  2>&1 &

TPID=$!

echo "$SPID" > "$V4/santa/resume_now_pid.txt"
echo "$TPID" > "$V4/tree/resume_now_pid.txt"

echo "V4_RESUME_NOW_STARTED"
echo "Santa GPU1 PID=$SPID"
echo "Tree  GPU2 PID=$TPID"
