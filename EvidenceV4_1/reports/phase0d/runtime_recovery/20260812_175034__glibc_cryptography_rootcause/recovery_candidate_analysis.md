# Recovery-candidate analysis (not executed)

| Candidate | Source code change | Python env change | Torch change | Wan-Move semantic risk | Re-freeze source | Re-freeze environment | Blast radius | Reversibility | Assessment |
|---|---:|---:|---:|---|---:|---:|---|---|---|
| R1: replace only incompatible cryptography build with GLIBC<=2.17-compatible build in same wan-move env | False | True | False | Unknown until package lock/import smoke validation | False | True | narrowly scoped to cryptography/dashscope import dependency | package rollback possible | viable but package ABI/version lock must be frozen and validated |
| R2: make DashScope import lazy/optional only when `--use_prompt_extend` | True | False | False | Expected none for formal runner because prompt extension is disabled; must verify patched entry/import plus re-freeze source | True | False | generate.py import behavior | source patch reversible | **recommended minimal recovery**: evidence proves DashScope is not functionally required by this formal invocation |
| R3: use another existing environment | False | True/Runtime switch | Possibly | Unknown | False | True | broad runtime dependency surface; bounded inventory has no equivalent Wan-Move environment | uncertain | not supported by evidence |
| R4: system GLIBC modification | system-level | system-level | potentially | high | N/A | N/A | system-wide | poor | **FORBIDDEN** |

`RECOMMENDED_MINIMAL_RECOVERY = R2`, subject to explicit authorization. It would require a source re-freeze and a no-prompt-extension import smoke check. R1 is an alternative only after a compatible wheel/artifact and package lock are explicitly approved. No candidate has been executed.
