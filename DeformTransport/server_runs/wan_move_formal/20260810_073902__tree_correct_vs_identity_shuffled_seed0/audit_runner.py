#!/usr/bin/env python3
import difflib
import json
import re
from pathlib import Path

dt = Path("/workspace/DeformTransport")
batch = dt / "server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0"
runners = {
    "santa_correct": dt / "server_runs/wan_move_formal/20260809_195255__santa_correct_vs_identity_shuffled_seed0/correct/run.sh",
    "tree_correct": batch / "correct/run.sh",
    "tree_shuffled": batch / "shuffled/run.sh",
}
expected = {
    "task": "wan-move-i2v",
    "size": "480*832",
    "frame_num": "81",
    "base_seed": "0",
    "sample_steps": "40",
    "sample_shift": "3.0",
    "dtype": "bf16",
    "offload_model": "True",
    "t5_cpu": "present",
    "checkpoint": "/workspace/Wan-Move/Wan-Move-14B-480P",
}


def extract(path):
    text = path.read_text()
    values = {}
    for key in ("task", "size", "frame_num", "base_seed", "sample_steps", "sample_shift", "dtype", "offload_model"):
        match = re.search(r"--" + key + r"\s+(['\"]?)([^\s'\"\\]+)\1", text)
        assert match, (path, key)
        values[key] = match.group(2)
    assert len(re.findall(r"--t5_cpu(?:\s|\\)", text)) == 1
    values["t5_cpu"] = "present"
    if '--ckpt_dir "/workspace/Wan-Move/Wan-Move-14B-480P"' in text:
        values["checkpoint"] = "/workspace/Wan-Move/Wan-Move-14B-480P"
    else:
        assert 'WAN_ROOT="/workspace/Wan-Move"' in text
        assert '--ckpt_dir "$WAN_ROOT/Wan-Move-14B-480P"' in text
        values["checkpoint"] = "/workspace/Wan-Move/Wan-Move-14B-480P"
    for key in ("task", "frame_num", "base_seed", "sample_steps", "sample_shift", "dtype", "offload_model"):
        assert text.count("--" + key + " ") == 1
    return values


manifests = {name: extract(path) for name, path in runners.items()}
checks = {name: manifest == expected for name, manifest in manifests.items()}
assert all(checks.values()), (checks, manifests)
base_lines = [f"{key}={expected[key]}\n" for key in sorted(expected)]
sections = []
for name in ("tree_correct", "tree_shuffled"):
    other = [f"{key}={manifests[name][key]}\n" for key in sorted(expected)]
    diff = list(difflib.unified_diff(base_lines, other, fromfile="santa_correct.normalized", tofile=f"{name}.normalized"))
    sections.append(f"===== {name} vs santa_correct =====\n")
    sections.extend(diff or ["<no algorithmic parameter differences>\n"])
    sections.append("allowed non-algorithmic differences: case image, prompt, tracks, visibility, output/run path, GPU argument\n")
(batch / "runner_vs_santa.diff").write_text("".join(sections))
report = {
    "status": "PRELAUNCH_RUNNER_AUDIT_OK",
    "normalized_manifests": manifests,
    "all_match_successful_santa_algorithmic_contract": all(checks.values()),
    "allowed_differences_only": ["case image", "prompt", "tracks", "visibility", "output/run path", "GPU argument"],
}
(batch / "prelaunch_runner_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
