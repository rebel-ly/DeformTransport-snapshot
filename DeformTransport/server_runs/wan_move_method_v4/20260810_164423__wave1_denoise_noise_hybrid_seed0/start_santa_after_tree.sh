#!/usr/bin/env bash
set -u

V4=/workspace/DeformTransport/server_runs/wan_move_method_v4/20260810_164423__wave1_denoise_noise_hybrid_seed0

echo "[$(date -Iseconds)] Waiting for Tree queue to finish..."

while [ ! -f "$V4/tree/RESUME_NOW_DONE.txt" ]; do
    sleep 60
done

echo "[$(date -Iseconds)] Tree queue finished. Starting Santa serial queue."

for VAR in v4a_d40 v4b_noise v4c_hybrid
do
    OUT="$V4/santa/$VAR"

    echo
    echo "============================================================"
    echo "START santa $VAR GPU1 $(date -Iseconds)"
    echo "============================================================"

    # 如果某项之前已经成功，就直接跳过
    if [ -s "$OUT/exit_code.txt" ] && \
       [ "$(cat "$OUT/exit_code.txt")" = "0" ] && \
       find "$OUT" -maxdepth 1 -name '*.mp4' -size +1M | grep -q .
    then
        echo "SKIP_ALREADY_SUCCESS santa $VAR"
        continue
    fi

    # 保存之前失败记录
    if [ -s "$OUT/exit_code.txt" ]; then
        STAMP=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$OUT/failed_${STAMP}"

        for F in exit_code.txt stdout.log stderr.log start_time.txt end_time.txt
        do
            [ -f "$OUT/$F" ] && cp "$OUT/$F" "$OUT/failed_${STAMP}/$F"
        done
    fi

    set +e

    bash "$V4/run_one.sh" santa "$VAR" 1

    EC=$?

    set -e

    echo "RESULT santa $VAR exit=$EC"

    if [ "$EC" -ne 0 ]; then
        echo "FAILED_BUT_CONTINUE santa $VAR exit=$EC"
    fi

    sleep 20
done

date -Iseconds > "$V4/santa/SERIAL_QUEUE_DONE.txt"

echo "[$(date -Iseconds)] ALL SANTA REMAINING TASKS FINISHED."
