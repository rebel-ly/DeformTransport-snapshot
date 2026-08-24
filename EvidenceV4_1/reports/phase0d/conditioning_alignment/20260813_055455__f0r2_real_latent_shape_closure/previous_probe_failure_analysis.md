# Previous VAE-probe failure analysis

`PREVIOUS_VAE_PROBE_ATTEMPTED=True`. F0-R recorded a VAE-only attempt but retained no structured stdout/stderr nor exit-code artifact. Its first observable blocker was therefore `NO_STRUCTURED_TENSOR_OUTPUT`, not a demonstrated VAE construction or encoding error.

F0-R2 performed the newly required single probe with archived stdout/stderr and exit code. Its exit code is `1`; the first blocker is `ModuleNotFoundError: No module named 'wan'` at the VAE import. The command used a `/tmp` script, so Python's script-directory import path was `/tmp`; the formal Wan-Move source root was not supplied via `PYTHONPATH`. This is an invocation binding failure before VAE instantiation, not a VAE, checkpoint, GPU, or tensor-shape result.

The protocol permits one F0-R2 VAE-only probe. It has now been consumed. No environment/source modification or retry was performed.
