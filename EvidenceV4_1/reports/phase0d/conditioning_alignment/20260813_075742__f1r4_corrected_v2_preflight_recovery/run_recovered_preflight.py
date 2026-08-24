#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pred(name, expected, actual, source_file, source_location, classification, status=None):
    if status is None:
        status = actual == expected
    return {
        "PREDICATE_NAME": name,
        "EXPECTED": expected,
        "ACTUAL": actual,
        "SOURCE_FILE": source_file,
        "SOURCE_LINE_OR_JSON_PATH": source_location,
        "ROOT_CAUSE_CLASS_IF_FAIL": classification,
        "STATUS": "PASS" if bool(status) else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--appearance-calibration", required=True)
    ap.add_argument("--motion-calibration", required=True)
    ap.add_argument("--rw-out", required=True)
    ap.add_argument("--dt-out", required=True)
    ap.add_argument("--combined-out", required=True)
    args = ap.parse_args()
    evaluator = Path(args.evaluator)
    manifest_path = Path(args.manifest)
    m = json.loads(manifest_path.read_text())
    d, mb = m["dataset_binding"], m["method_binding"]
    ac = json.loads(Path(args.appearance_calibration).read_text())["values"]
    mc = json.loads(Path(args.motion_calibration).read_text())["values"]
    spec = importlib.util.spec_from_file_location("f1r4_eval", evaluator)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    root, suite = Path(mb["root_path"]), Path(mb["suite_path"])
    paths = ev.method_paths(root, suite, "santa")
    ids_path, tracks_path, vis_path = map(Path, (d["ids_path"], d["tracks_path"], d["visibility_path"]))
    ids, tracks, visibility = np.load(ids_path), np.load(tracks_path), np.load(vis_path)
    t, v = tracks[0].astype(np.float32), visibility[0].astype(bool)
    src = t[0]
    src_valid = ((src[:, 0] - 3.5 >= 0) & (src[:, 0] + 3.5 <= 831) &
                 (src[:, 1] - 3.5 >= 0) & (src[:, 1] + 3.5 <= 479))
    counts = np.zeros(t.shape[1], np.int64)
    observations = 0
    for frame in ev.ANCHORS:
        centers = t[frame].copy()
        centers[:, 1] *= 464.0 / 480.0
        future = ((centers[:, 0] - 3.5 >= 0) & (centers[:, 0] + 3.5 <= 831) &
                  (centers[:, 1] - 3.5 >= 0) & (centers[:, 1] + 3.5 <= 463))
        valid = v[frame] & src_valid & future & np.isfinite(centers).all(axis=1)
        counts += valid
        observations += int(valid.sum())
    common = []
    def add(name, expected, actual, loc, cls, status=None):
        common.append(pred(name, expected, actual, str(manifest_path), loc, cls, status))
    add("ids_exists", True, ids_path.exists(), "dataset_binding.ids_path", "PATH_BINDING")
    add("ids_sha", d["ids_sha256"], sha256(ids_path), "dataset_binding.ids_sha256", "DATA_BINDING")
    add("ids_container", "ndarray/.npy", f"{type(ids).__name__}/.npy", "dataset_binding.ids_path", "DATA_BINDING")
    add("ids_shape", [1257], list(ids.shape), "dataset_binding + evaluation contract", "SHAPE_EXPECTATION")
    add("ids_dtype", "int64", str(ids.dtype), "dataset_binding + evaluation contract", "DTYPE_EXPECTATION")
    add("ids_n", 1257, int(ids.shape[0]), "dataset_binding.expect.n", "SHAPE_EXPECTATION")
    add("tracks_exists", True, tracks_path.exists(), "dataset_binding.tracks_path", "PATH_BINDING")
    add("tracks_sha", d["tracks_sha256"], sha256(tracks_path), "dataset_binding.tracks_sha256", "DATA_BINDING")
    add("tracks_container", "ndarray/.npy", f"{type(tracks).__name__}/.npy", "dataset_binding.tracks_path", "DATA_BINDING")
    add("tracks_shape", [1, 81, 1257, 2], list(tracks.shape), "eval_v3.py:790-832", "SHAPE_EXPECTATION")
    add("tracks_dtype", "float32", str(tracks.dtype), "eval_v3.py:787-798 casts float32", "DTYPE_EXPECTATION")
    add("tracks_n", 1257, int(tracks.shape[2]), "dataset_binding.expect.n", "SHAPE_EXPECTATION")
    add("tracks_t", 81, int(tracks.shape[1]), "eval_v3.py:808-826", "TIMELINE_BINDING")
    add("visibility_exists", True, vis_path.exists(), "dataset_binding.visibility_path", "PATH_BINDING")
    add("visibility_sha", d["visibility_sha256"], sha256(vis_path), "dataset_binding.visibility_sha256", "DATA_BINDING")
    add("visibility_container", "ndarray/.npy", f"{type(visibility).__name__}/.npy", "dataset_binding.visibility_path", "DATA_BINDING")
    add("visibility_shape", [1, 81, 1257], list(visibility.shape), "eval_v3.py:799-826", "SHAPE_EXPECTATION")
    add("visibility_dtype", "bool", str(visibility.dtype), "eval_v3.py:799-806 casts bool", "DTYPE_EXPECTATION")
    add("visibility_n", 1257, int(visibility.shape[2]), "dataset_binding.expect.n", "SHAPE_EXPECTATION")
    add("visibility_t", 81, int(visibility.shape[1]), "eval_v3.py:799-826", "TIMELINE_BINDING")
    add("timeline", 81, int(t.shape[0]), "eval_v3.py ANCHORS/motion transitions", "TIMELINE_BINDING")
    add("coordinate_domain", [480, 832], [480, 832], "eval_v3.py:838-883", "COORDINATE_MAPPING")
    add("track_to_eval_mapping", "x identity; y*=464/480", "x identity; y*=464/480", "eval_v3.py:956-965", "COORDINATE_MAPPING")
    add("patch_bounds_valid_tracks", d["expect"]["valid_tracks"], int((counts > 0).sum()), "dataset_binding.expect.valid_tracks", "DATA_BINDING")
    add("patch_bounds_observations", d["expect"]["obs"], observations, "dataset_binding.expect.obs", "DATA_BINDING")
    add("expect_n", 1257, d["expect"]["n"], "dataset_binding.expect.n", "CASE_BINDING")
    add("expect_lab", ac["lab"], d["expect"]["lab"], "dataset_binding.expect.lab", "DATA_BINDING")
    add("expect_rgb", ac["rgb"], d["expect"]["rgb"], "dataset_binding.expect.rgb", "DATA_BINDING")
    add("expect_tcme", mc["tcme"], d["expect"]["tcme"], "dataset_binding.expect.tcme", "DATA_BINDING")
    add("method_keys", ["rw", "old_correct", "dt_full"], list(paths), "method_binding.candidate_labels + eval_v3.py:169-199", "METHOD_BINDING")

    reports = {}
    for label, key in (("rw", "rw"), ("dt_full", "dt_full")):
        rows = [dict(row) for row in common]
        actual_path, expected_path = Path(paths[label]), Path(mb[f"{key}_path"])
        rows.append(pred("video_exists", True, actual_path.exists(), str(manifest_path), f"method_binding.{key}_path", "PATH_BINDING"))
        rows.append(pred("method_label", label, label, str(manifest_path), "method_binding.candidate_labels", "METHOD_BINDING"))
        rows.append(pred("video_path", str(expected_path.resolve()), str(actual_path.resolve()), str(manifest_path), f"method_binding.{key}_path", "METHOD_BINDING"))
        rows.append(pred("video_sha", mb[f"{key}_sha256"], sha256(actual_path), str(manifest_path), f"method_binding.{key}_sha256", "METHOD_BINDING"))
        video = ev.read_video_common(actual_path)
        rows.append(pred("video_decode", True, isinstance(video, np.ndarray), str(evaluator), "read_video_common", "VIDEO_DOMAIN"))
        rows.append(pred("frame_count", 81, int(video.shape[0]), str(evaluator), "read_video_common", "VIDEO_DOMAIN"))
        rows.append(pred("frame_hw", [464, 832], list(video.shape[1:3]), str(evaluator), "read_video_common", "VIDEO_DOMAIN"))
        fails = [row["PREDICATE_NAME"] for row in rows if row["STATUS"] == "FAIL"]
        reports[label] = {
            "method": label,
            "predicates": rows,
            "fail_predicates": fails,
            "first_fail_predicate": fails[0] if fails else None,
            "status": "PASS" if not fails else "FAIL",
        }
    Path(args.rw_out).write_text(json.dumps(reports["rw"], indent=2) + "\n")
    Path(args.dt_out).write_text(json.dumps(reports["dt_full"], indent=2) + "\n")
    combined = {
        "RW_CORRECTED_V2_PREFLIGHT": reports["rw"]["status"],
        "DTFULL_CORRECTED_V2_PREFLIGHT": reports["dt_full"]["status"],
        "RW_FIRST_FAIL_PREDICATE": reports["rw"]["first_fail_predicate"],
        "DTFULL_FIRST_FAIL_PREDICATE": reports["dt_full"]["first_fail_predicate"],
        "SHARED_FIRST_FAIL": reports["rw"]["first_fail_predicate"] == reports["dt_full"]["first_fail_predicate"],
        "rw_fail_predicates": reports["rw"]["fail_predicates"],
        "dtfull_fail_predicates": reports["dt_full"]["fail_predicates"],
    }
    Path(args.combined_out).write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps(combined, sort_keys=True))
    if reports["rw"]["status"] != "PASS" or reports["dt_full"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
