# F1 Original vs F4-R1 Current Runtime Provenance

| Field | F1 original | F4-R1 current | Classification |
|---|---|---|---|
| Evaluator SHA | `e6a00e...2a77ef5` | `e6a00e...2a77ef5` | IDENTICAL |
| RW / DT-FULL video SHA | recorded frozen values | matching frozen values | IDENTICAL |
| IDs / tracks / visibility SHA | corrected-v2 frozen values | matching frozen values | IDENTICAL |
| Python executable | host `/mnt/.../wan-move/bin/python` | container `/workspace/.../wan-move/bin/python` | DIFFERENT namespace |
| GPU binding | `CUDA_VISIBLE_DEVICES=1` | default visible device (F4-R1) | DIFFERENT |
| RAFT backend | torchvision RAFT large C_T_SKHT_V2 | same frozen evaluator backend | IDENTICAL semantic code |
| RAFT checkpoint | `ff5fadd56d26...dd16a322` | same mounted checkpoint | IDENTICAL |
| Historical per-transition arrays | not preserved | not emitted by frozen evaluator | UNRESOLVED |
| Deterministic/thread flags | not preserved | not a frozen formal setting | UNRESOLVED |

`SEMANTIC_EVALUATOR_CHANGE_DETECTED = False`.

`INPUT_CHANGE_DETECTED = False`.

`RUNTIME_NUMERICAL_DIFFERENCE_DETECTED = True`: five current container repeats are internally bitwise identical but differ from F1 references; a host restoration probe with the recorded Python path and `CUDA_VISIBLE_DEVICES=1` exactly reproduced both historical baselines.
