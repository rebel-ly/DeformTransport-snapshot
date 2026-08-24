# Launcher exit-code audit

The Phase0D-2 parallel operational launcher backgrounded Run A and Run B, wrote their PIDs, then used a single bare `wait`. A bare `wait` returns aggregate completion status but does not persist the separate statuses of PID_A and PID_B. Consequently the frozen record correctly reports `NOT_CAPTURED_BY_LAUNCHER`.

Required future recovery-launcher change (not executed here):

```bash
wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
printf '%s\n' "$RC_A" > "$RUN_A/exit_code.txt"
printf '%s\n' "$RC_B" > "$RUN_B/exit_code.txt"
```

`LAUNCHER_EXITCODE_CAPTURE = REQUIRES_FIX`. This is launch-recording engineering only; it does not change generation semantics.
