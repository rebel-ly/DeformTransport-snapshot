# Runtime binding audit

The selected interpreter is the existing, unmodified `/workspace/tools/miniforge3/envs/wan-move/bin/python`. Historical successful Santa Wan-Move resource logs explicitly record this interpreter launching `generate.py` with the Wan-Move checkpoint and Santa inputs. It has stronger provenance than any candidate selected by name alone.

`run_with_formal_wanmove_python.sh` is a new inert container-only wrapper. It is not a modification of the frozen formal runner. Its only changes relative to the frozen formal execution contract are (1) binding the exact existing Python executable, and (2) translating `/mnt/sdbd/home/liuyu_qyh` asset prefixes to the proven `/workspace` container mounts. Generation arguments, source identity, input assets, prompt, seed handling, scheduler defaults, and V3D environment variables are retained.

Generation parameter changes: 0. Scientific input changes: 0. Seed changes: 0. Source-code core changes: 0. The wrapper is used only with `--dry-run` in this phase.
