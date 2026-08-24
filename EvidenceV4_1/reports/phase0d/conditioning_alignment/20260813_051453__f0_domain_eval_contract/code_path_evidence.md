# Code-path evidence

- Formal runner requests `--size '480*832'`, 81 frames, corrected-v2 N=1257 tracks and visibility: `run_with_formal_wanmove_python.sh` lines 5-24.
- `generate.py:497-510` calls the non-evaluation `WanMove.generate`; `wan_move.py:200-220` obtains source H/W and computes target H/W, then applies `x'=x*w/img_w`, `y'=y*h/img_h`.
- `wan_move.py:222-243` builds a 4x21 mask; `:266-284` encodes a `[3,81,h,w]` source-plus-zero-future tensor and concatenates the mask with the VAE y.
- `wan_move.py:374-389` decodes sampled latent; `generate.py:535-542` supplies that video tensor to `cache_video`; `utils.py:40-54` converts and writes frames but contains no spatial resize/crop.
- No explicit `480 -> 464` operation was located on the frozen formal code path. The 464 height is established by frozen decoded RGB evidence, but its mechanism is unresolved.
- Evaluator candidate `eval_v3.py:260-282` maps 480x832 video to 464x832 with bicubic interpolation. `:956-965` maps only the y coordinate by `464/480`. Its Santa CASES entry asserts N=1277, so it is not a corrected-v2 N=1257 evaluator.
