# Formal prompt-disabled path audit

The frozen corrected-v2 runner is `reports/phase0d/seed_contract/20260812_172755__santa_v3d_contract_recovery/formal_run_corrected_v2_v3d.sh`. It invokes `generate.py` with a fixed prompt read from the frozen Santa prompt file and does not pass `--use_prompt_extend`.

`generate.py:175-178` defines `--use_prompt_extend` as `store_true` with `default=False`. `generate.py:314` constructs either `DashScopePromptExpander` or `QwenPromptExpander` only beneath `if args.use_prompt_extend:`. `generate.py:474` calls the expander only beneath a second `if args.use_prompt_extend:`.

Therefore `FORMAL_PROMPT_EXTENSION_ENABLED = False` and `FORMAL_RUN_REACHES_DASHSCOPE_FUNCTION = False`. The source-level import in `generate.py:20` makes the module available, but no `DashScopePromptExpander` constructor or DashScope API function is reached by the frozen formal command.

The prompt is already fixed and local: `Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.` No external prompt extension is required by this formal path.
