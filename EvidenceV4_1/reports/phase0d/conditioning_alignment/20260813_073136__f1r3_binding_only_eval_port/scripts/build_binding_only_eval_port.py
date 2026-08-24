#!/usr/bin/env python3
"""Deterministically derive eval_v3 ports by replacing binding AST literals only."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def offsets(text):
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def span(starts, node):
    return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset


def find_assign(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    raise RuntimeError(f"assignment not found: {name}")


def dict_value(node, key):
    if not isinstance(node, ast.Dict):
        raise RuntimeError(f"expected dict while looking for {key}")
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    raise RuntimeError(f"dict key not found: {key}")


def replacement(node, new_text, category, field, starts, source):
    begin, end = span(starts, node)
    return {
        "start": begin,
        "end": end,
        "old": source[begin:end],
        "new": new_text,
        "category": category,
        "field": field,
        "original_lines": [node.lineno, node.end_lineno],
    }


def build(source, manifest):
    if manifest["mode"] == "identity":
        return source, []

    tree = ast.parse(source)
    starts = offsets(source)
    changes = []

    candidates_assign = find_assign(tree, "CANDIDATES")
    changes.append(replacement(
        candidates_assign.value,
        repr(manifest["method_binding"]["candidate_labels"]),
        "METHOD_BINDING", "CANDIDATES", starts, source,
    ))

    cases_assign = find_assign(tree, "CASES")
    santa = dict_value(cases_assign.value, "santa")
    dataset = manifest["dataset_binding"]
    changes.append(replacement(dict_value(santa, "tracks"), repr(dataset["tracks_path"]), "DATASET_BINDING", "CASES.santa.tracks", starts, source))
    changes.append(replacement(dict_value(santa, "vis"), repr(dataset["visibility_path"]), "DATASET_BINDING", "CASES.santa.vis", starts, source))
    expect = dict_value(santa, "expect")
    for key in ("n", "valid_tracks", "obs", "lab", "rgb", "tcme"):
        changes.append(replacement(dict_value(expect, key), repr(dataset["expect"][key]), "DATASET_BINDING", f"CASES.santa.expect.{key}", starts, source))

    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    case_loop = None
    for node in ast.walk(main):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "case" and isinstance(node.iter, (ast.List, ast.Tuple)):
            values = [e.value for e in node.iter.elts if isinstance(e, ast.Constant)]
            if values == ["santa", "tree"]:
                case_loop = node.iter
                break
    if case_loop is None:
        raise RuntimeError("appearance case-selection binding not found")
    changes.append(replacement(case_loop, repr(manifest["output_binding"]["appearance_cases"]), "OUTPUT_BINDING", "main.appearance_cases", starts, source))

    for item in sorted(changes, key=lambda x: x["start"], reverse=True):
        if source[item["start"]:item["end"]] != item["old"]:
            raise RuntimeError(f"source drift at {item['field']}")
        source = source[:item["start"]] + item["new"] + source[item["end"]:]
    ast.parse(source)
    return source, sorted(changes, key=lambda x: x["start"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    original = Path(manifest["original_evaluator"])
    raw = original.read_bytes()
    actual = sha256_bytes(raw)
    if actual != manifest["original_sha256"]:
        raise RuntimeError(f"original evaluator SHA mismatch: {actual}")
    generated, changes = build(raw.decode("utf-8"), manifest)
    out = Path(args.out)
    out.write_text(generated)
    report = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "original": str(original.resolve()),
        "original_sha256": actual,
        "output": str(out.resolve()),
        "output_sha256": sha256_bytes(out.read_bytes()),
        "replacement_count": len(changes),
        "replacements": changes,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
