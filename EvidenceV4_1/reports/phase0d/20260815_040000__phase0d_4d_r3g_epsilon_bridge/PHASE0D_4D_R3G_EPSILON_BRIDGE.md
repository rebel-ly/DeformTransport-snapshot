# Phase 0D-4D-R3G: epsilon bridge and C-pair validity

This report is engineering/provenance-only. No C1/C2 scientific metric or video inspection was performed.

- C1 completed normally: exit 0, marker present, 81 decoded frames at 832x464, and no fatal runtime signature.
- The current formal scheduler gives sigma[0] = 0.9996664524078369 (timestep 999), so the D1 premise sigma[0] == 1 is false. D1 is therefore an invalidated exact oracle rather than evidence of epsilon-bridge failure.
- A CUDA private-generator reconstruction following the current `wan_move.py` call order exactly matches the R3 external epsilon: `torch.equal=True`, zero differing scalars.
- C1/C2 use attested common enabled-path configuration with the intended K=0 versus Correct K=1257 carrier-set difference. Their result seal remains closed.
- E0 epsilon-only canonical replay reached 3/40 denoise steps but exited with CUDA OOM before output. A concurrent 6.21 GiB process was named in the raw CUDA error; no retry or intervention was performed. No GPU3 use occurred.
