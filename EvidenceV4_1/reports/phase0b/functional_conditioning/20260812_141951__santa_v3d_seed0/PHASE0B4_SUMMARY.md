# Phase0B-4 Functional Conditioning Differential Audit

Status: **CPU_IMPORT_BLOCKED**

The required Phase0B-3 archive gate, frozen-source drift gate, formal input
contract, and authoritative depth/ID lineage gate all passed.  The prescribed
CPU-only function-level execution could not begin because importing the frozen
patched Wan-Move package imports `wan.modules.__init__`, whose T5 module
evaluates `torch.cuda.current_device()` as a class default argument.  With
`CUDA_VISIBLE_DEVICES=""`, this raises `RuntimeError: No CUDA GPUs are
available` before `wan.modules.trajectory` can be imported.

Per the preregistered protocol, Wan-Move was not modified, no alternative
import route or GPU execution was attempted, and the audit stopped.  See
`traceback.txt` and `phase0b4_status.json` for raw evidence.

Pre-execution gates recorded in `phase0b4_status.json`:

- Phase0B-3 archive status: PASS
- Git HEAD and both frozen source SHA256 values: PASS
- Formal Correct/Shuffled/visibility/ID shapes and dtypes: PASS
- Authoritative depth/ID lineage and finite consumed depth values: PASS
