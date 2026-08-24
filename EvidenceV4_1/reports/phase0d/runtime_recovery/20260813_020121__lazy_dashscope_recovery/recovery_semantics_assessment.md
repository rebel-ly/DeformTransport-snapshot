# Recovery scientific-semantics assessment

The patch changes optional dependency import timing only: DashScope is no longer imported at module import time, and is imported inside `DashScopePromptExpander.__init__`, the first actual DashScope-dependent execution boundary. DashScope API behavior was retained by storing the imported module on `self.dashscope` and using that same module for the pre-existing configuration and API calls.

The recovery did not change corrected-v2 input, N=1257, T=81, V3D trajectory semantics, source feature lookup, visibility, depth, material IDs, arbitration, direct replacement, diffusion seed, checkpoint, prompt, resolution, frame count, sample steps, sample shift, CFG, scheduler, dtype, or VAE/model settings. `trajectory.py` and `wan_move.py` exactly retain their frozen SHA256 values.

The formal command keeps prompt extension disabled, so DashScope-dependent functionality is outside the formal execution path. `NO_FORMAL_PATH_SEMANTIC_CHANGE_DETECTED` is the supported conclusion. This does not claim that source code is unchanged: `wan/utils/prompt_extend.py` changed and is re-frozen by this evidence.
