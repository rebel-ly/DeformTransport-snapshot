#!/usr/bin/env python3
"""Audit binding-only diffs plus exact source and normalized-AST metric identity."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


SEMANTIC_FUNCTIONS = [
    "sha256", "read_rgb_image", "to_common", "read_video_common",
    "sample_patches", "patch_mean_lab", "stats", "bootstrap_mean_ci",
    "aggregate", "appearance_case", "bilinear_flow", "load_raft_cached",
    "motion_case",
]
SEMANTIC_CONSTANTS = ["BOOT_SEED", "BOOT_N", "ANCHORS", "OFF"]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def nodes_by_name(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
    return result


def ast_dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def audit_pair(original_path, port_path, build_report_path):
    original = original_path.read_text()
    port = port_path.read_text()
    otree, ptree = ast.parse(original), ast.parse(port)
    onodes, pnodes = nodes_by_name(otree), nodes_by_name(ptree)
    regions = []
    all_sha = True
    all_ast = True
    for name in SEMANTIC_CONSTANTS + SEMANTIC_FUNCTIONS:
        if name not in onodes or name not in pnodes:
            raise RuntimeError(f"semantic region missing: {name}")
        osrc = ast.get_source_segment(original, onodes[name])
        psrc = ast.get_source_segment(port, pnodes[name])
        sha_equal = digest(osrc) == digest(psrc)
        ast_equal = ast_dump(onodes[name]) == ast_dump(pnodes[name])
        all_sha &= sha_equal
        all_ast &= ast_equal
        regions.append({
            "name": name,
            "kind": "constant" if name in SEMANTIC_CONSTANTS else "function",
            "original_lines": [onodes[name].lineno, onodes[name].end_lineno],
            "original_sha256": digest(osrc),
            "port_sha256": digest(psrc),
            "source_sha_equal": sha_equal,
            "ast_equal": ast_equal,
        })

    report = json.loads(build_report_path.read_text())
    rebuilt = original
    for item in sorted(report["replacements"], key=lambda x: x["start"], reverse=True):
        if rebuilt[item["start"]:item["end"]] != item["old"]:
            raise RuntimeError(f"build report old text mismatch at {item['field']}")
        rebuilt = rebuilt[:item["start"]] + item["new"] + rebuilt[item["end"]:]
    exact_builder_output = rebuilt == port

    matcher = difflib.SequenceMatcher(a=original.splitlines(), b=port.splitlines(), autojunk=False)
    hunks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunks.append({"tag": tag, "original_lines": [i1 + 1, i2], "port_lines": [j1 + 1, j2]})
    forbidden = 0 if exact_builder_output else len(hunks)
    return {
        "original": str(original_path.resolve()),
        "port": str(port_path.resolve()),
        "total_source_diff_hunks": len(hunks),
        "allowed_binding_diff_hunks": len(hunks) if exact_builder_output else 0,
        "forbidden_source_diff_hunks": forbidden,
        "exact_deterministic_builder_output": exact_builder_output,
        "metric_semantic_region_count": len(regions),
        "all_metric_region_sha_equal": all_sha,
        "metric_ast_equivalence": "PASS" if all_ast else "FAIL",
        "regions": regions,
        "diff_hunks": hunks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--legacy-build-report", required=True)
    parser.add_argument("--corrected", required=True)
    parser.add_argument("--corrected-build-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--regions-out", required=True)
    parser.add_argument("--forbidden-out", required=True)
    args = parser.parse_args()
    original = Path(args.original)
    legacy = audit_pair(original, Path(args.legacy), Path(args.legacy_build_report))
    corrected = audit_pair(original, Path(args.corrected), Path(args.corrected_build_report))
    combined = {
        "legacy": legacy,
        "corrected_v2": corrected,
        "TOTAL_SOURCE_DIFF_HUNKS": legacy["total_source_diff_hunks"] + corrected["total_source_diff_hunks"],
        "ALLOWED_BINDING_DIFF_HUNKS": legacy["allowed_binding_diff_hunks"] + corrected["allowed_binding_diff_hunks"],
        "FORBIDDEN_SOURCE_DIFF_HUNKS": legacy["forbidden_source_diff_hunks"] + corrected["forbidden_source_diff_hunks"],
        "ALL_METRIC_REGION_SHA_EQUAL": legacy["all_metric_region_sha_equal"] and corrected["all_metric_region_sha_equal"],
        "METRIC_AST_EQUIVALENCE": "PASS" if legacy["metric_ast_equivalence"] == corrected["metric_ast_equivalence"] == "PASS" else "FAIL",
    }
    Path(args.out).write_text(json.dumps(combined, indent=2) + "\n")
    Path(args.regions_out).write_text(json.dumps({"semantic_regions": corrected["regions"]}, indent=2) + "\n")
    Path(args.forbidden_out).write_text(json.dumps({
        "forbidden_source_diff_hunks": combined["FORBIDDEN_SOURCE_DIFF_HUNKS"],
        "legacy": [] if legacy["forbidden_source_diff_hunks"] == 0 else legacy["diff_hunks"],
        "corrected_v2": [] if corrected["forbidden_source_diff_hunks"] == 0 else corrected["diff_hunks"],
    }, indent=2) + "\n")
    print(json.dumps({k: v for k, v in combined.items() if k.isupper()}, sort_keys=True))
    if combined["FORBIDDEN_SOURCE_DIFF_HUNKS"] or not combined["ALL_METRIC_REGION_SHA_EQUAL"] or combined["METRIC_AST_EQUIVALENCE"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
