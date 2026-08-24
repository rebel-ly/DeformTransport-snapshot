# Seed-consumption map (frozen Wan-Move source)

- `generate.py:62`: negative CLI base seed is randomized with `random.randint`; formal command must use explicit `--base_seed`.
- `wan/wan_move.py:226-233`: `seed` creates `torch.Generator(device=self.device)`, manually seeds it, then `torch.randn(..., generator=seed_g)` creates initial diffusion noise. The same generator is passed to scheduler steps (lines 363+); this affects diffusion/output.
- `wan/wan_move.py:278`: `torch.manual_seed(seed)` resets global Torch RNG immediately before trajectory conditioning.
- `trajectory.py:240`: first `torch.randperm(n)` reorders tracks/visibility together.
- `trajectory.py:257`: second `torch.randperm(n)` creates position embeddings accumulated into `track_feats`.
- `trajectory.py:756`: third `torch.randperm(n)` reorders track_pos and all associated V3D fields together.
- `wan/wan_move.py:279-281`: `track_feats` is constructed but is not included in `arg_c` or `arg_null` passed to `self.model` (lines 337-348); `edited_y` is used in `y_cond`. Thus the second RNG site is not generator-consumed on this frozen call path.
- Static/algebraic consequence: seed changes alter diffusion noise and unused track_feats. V3D target support, winners, depths, carrier contributions and final write positions are algebraically invariant because the first/third permutations reindex coupled fields and winner is min(depth, material_id). Exact real-latent `edited_y` across seeds was not functionally hashed: doing so would require VAE encoding and is outside this no-14B-generation contract audit.

No additional global RNG use after trajectory conditioning was detected in the frozen `WanMove.generate` path; scheduler sampling uses `seed_g`.
