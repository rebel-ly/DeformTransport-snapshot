#!/usr/bin/env python3
"""Compute data-bound reproduction constants with the immutable evaluator semantics."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=["appearance", "motion"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    spec = importlib.util.spec_from_file_location("frozen_eval_v3_calibration", args.evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = manifest["dataset_binding"]
    cfg = module.CASES["santa"]
    cfg["tracks"] = dataset["tracks_path"]
    cfg["vis"] = dataset["visibility_path"]
    cfg["expect"].update(dataset["expect"])
    cfg["expect"]["lab"] = float("nan")
    cfg["expect"]["rgb"] = float("nan")
    cfg["expect"]["tcme"] = float("nan")
    module.CANDIDATES = []
    root = Path(manifest["method_binding"]["root_path"])
    suite = Path(manifest["method_binding"]["suite_path"])
    if args.mode == "appearance":
        report = module.appearance_case(root, suite, "santa")
        values = {
            "lab": report["methods"]["old_correct"]["tc_mar_lab"]["mean"],
            "rgb": report["methods"]["old_correct"]["tc_mar_rgb_l1"]["mean"],
            "valid_tracks": report["valid_tracks"],
            "obs": report["total_valid_anchor_observations"],
        }
    else:
        report = module.motion_case(root, suite, "santa", args.batch)
        values = {"tcme": report["methods"]["old_correct"]["transition_mean_epe_mean"]}
    Path(args.out).write_text(json.dumps({
        "purpose": "DATA_BINDING_EXPECTATION_CALIBRATION_ONLY",
        "formal_rw_vs_dtfull_comparison": False,
        "immutable_metric_source": str(Path(args.evaluator).resolve()),
        "candidate_methods_disabled": True,
        "mode": args.mode,
        "values": values,
        "report": report,
    }, indent=2) + "\n")
    print(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
