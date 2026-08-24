# DashScope import audit

## Source evidence

- `Wan-Move/generate.py:20` unconditionally imports `DashScopePromptExpander, QwenPromptExpander` from `wan.utils.prompt_extend` at module import time.
- `Wan-Move/wan/utils/prompt_extend.py:12` unconditionally imports `dashscope` at module import time.
- `generate.py:175-178` defines `--use_prompt_extend` as `store_true`, default `False`.
- `generate.py:314-326` only constructs/uses prompt expander inside `if args.use_prompt_extend`; DashScope is selected only when `args.prompt_extend_method == "dashscope"` (315-318).

## Formal runner audit

The frozen Phase0D-1R corrected-v2 runner passes a fixed `--prompt` but no `--use_prompt_extend` or `--prompt_extend_method` option. Therefore `FORMAL_PROMPT_EXTENSION_ENABLED=False` and DashScope is not functionally required by this formal invocation. It nevertheless is loaded because imports at generate.py:20 and prompt_extend.py:12 are eager/unconditional.

## Exact failing chain

`generate.py:20` → `prompt_extend.py:12 import dashscope` → `dashscope.aigc...http_request` → `dashscope.api_entities.encryption` → `cryptography.hazmat..._rust.abi3.so` → dynamic loader rejects GLIBC_2.18/2.25/2.28 on CentOS 7 GLIBC 2.17.
