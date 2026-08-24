import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ANCHORS = list(range(4, 81, 4))
EARLY = list(range(4, 41, 4))
LATE = list(range(44, 81, 4))
RETURN_INTERMEDIATE = list(range(4, 80, 4))

MIN_RETURN_SUPPORT = 30


def load_ev(path):
    spec = importlib.util.spec_from_file_location(
        "frozen_eval_v3",
        str(path),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def future_patch_valid(tracks_t):
    p = tracks_t.astype(np.float32).copy()

    p[:, 1] *= 464.0 / 480.0

    return (
        np.isfinite(p).all(axis=1)
        & (p[:, 0] - 3.5 >= 0)
        & (p[:, 0] + 3.5 <= 831)
        & (p[:, 1] - 3.5 >= 0)
        & (p[:, 1] + 3.5 <= 463)
    )


def raw_point_in_frame(tracks_t):
    p = tracks_t

    return (
        np.isfinite(p).all(axis=1)
        & (p[:, 0] >= 0)
        & (p[:, 0] <= 831)
        & (p[:, 1] >= 0)
        & (p[:, 1] <= 479)
    )


def source_patch_valid(tracks0):
    p = tracks0

    return (
        np.isfinite(p).all(axis=1)
        & (p[:, 0] - 3.5 >= 0)
        & (p[:, 0] + 3.5 <= 831)
        & (p[:, 1] - 3.5 >= 0)
        & (p[:, 1] + 3.5 <= 479)
    )


def method_paths(ev, root, suite, case):
    cfg = ev.CASES[case]

    return {
        "rw":
            root / cfg["rw"],

        "v3d":
            suite
            / case
            / "v3d"
            / f"{case}_v3d_correct_seed0.mp4",
    }


def bootstrap_case_balanced(
    case_diffs,
    nboot=10000,
    seed=0,
):
    rng = np.random.default_rng(seed)

    pieces = []

    cases = list(case_diffs.keys())

    for start in range(0, nboot, 250):
        k = min(250, nboot - start)

        boot_case_means = []

        for case in cases:
            d = np.asarray(
                case_diffs[case],
                np.float64,
            )

            idx = rng.integers(
                0,
                len(d),
                size=(k, len(d)),
            )

            boot_case_means.append(
                d[idx].mean(axis=1)
            )

        pieces.append(
            np.stack(
                boot_case_means,
                axis=0,
            ).mean(axis=0)
        )

    x = np.concatenate(pieces)

    return [
        float(np.percentile(x, 2.5)),
        float(np.percentile(x, 97.5)),
    ]


def rw_v3d_decision(ci):
    # RW - V3D; positive means V3D better.
    if ci[0] > 0:
        return "WIN"
    if ci[1] < 0:
        return "LOSS"
    return "TIE"


def track_mean(errors, anchors):
    x = np.stack(
        [errors[t] for t in anchors],
        axis=0,
    )

    finite = np.isfinite(x)

    count = finite.sum(axis=0)

    sums = np.where(
        finite,
        x,
        0.0,
    ).sum(axis=0)

    out = np.full(
        x.shape[1],
        np.nan,
        np.float64,
    )

    good = count > 0

    out[good] = sums[good] / count[good]

    return out, count


def load_case_arrays(ev, root, case):
    cfg = ev.CASES[case]

    tracks = np.load(
        root / cfg["tracks"]
    )[0].astype(np.float32)

    visibility = np.load(
        root / cfg["vis"]
    )[0].astype(bool)

    assert tracks.ndim == 3
    assert tracks.shape[0] == 81
    assert tracks.shape[2] == 2
    assert visibility.shape == tracks.shape[:2]

    return tracks, visibility


def build_support(ev, root, out, case):
    tracks, vis = load_case_arrays(
        ev,
        root,
        case,
    )

    n = tracks.shape[1]

    src_patch = source_patch_valid(
        tracks[0]
    )

    src_visible = (
        src_patch
        & vis[0]
    )

    balanced = src_visible.copy()

    valid_anchor = {}

    for t in ANCHORS:
        v = (
            vis[t]
            & future_patch_valid(
                tracks[t]
            )
        )

        valid_anchor[t] = v

        balanced &= v

    # --------------------------------------------------------
    # Return after true in-frame visibility loss.
    # --------------------------------------------------------

    return_occ = (
        src_visible
        & vis[80]
        & future_patch_valid(
            tracks[80]
        )
    )

    had_inframe_occlusion = np.zeros(
        n,
        dtype=bool,
    )

    had_out_of_view = np.zeros(
        n,
        dtype=bool,
    )

    for t in RETURN_INTERMEDIATE:
        inframe = raw_point_in_frame(
            tracks[t]
        )

        had_inframe_occlusion |= (
            (~vis[t])
            & inframe
        )

        had_out_of_view |= (
            ~inframe
        )

    return_occ &= had_inframe_occlusion

    return_oov = (
        src_visible
        & vis[80]
        & future_patch_valid(
            tracks[80]
        )
        & had_out_of_view
    )

    ids_balanced = np.where(
        balanced
    )[0].astype(np.int64)

    ids_return = np.where(
        return_occ
    )[0].astype(np.int64)

    ids_return_oov = np.where(
        return_oov
    )[0].astype(np.int64)

    np.save(
        out / f"{case}_balanced_track_ids.npy",
        ids_balanced,
    )

    np.save(
        out / f"{case}_return_occlusion_track_ids.npy",
        ids_return,
    )

    np.save(
        out / f"{case}_return_oov_track_ids.npy",
        ids_return_oov,
    )

    result = {
        "all_tracks":
            int(n),

        "source_patch_valid_tracks":
            int(src_patch.sum()),

        "source_valid_visible_tracks":
            int(src_visible.sum()),

        "balanced_complete_case_tracks":
            int(len(ids_balanced)),

        "balanced_fraction_of_all":
            float(
                len(ids_balanced) / n
            ),

        "balanced_fraction_of_source_valid_visible":
            float(
                len(ids_balanced)
                / max(1, int(src_visible.sum()))
            ),

        "return_after_occlusion_tracks":
            int(len(ids_return)),

        "return_after_occlusion_support":
            (
                "SUFFICIENT"
                if len(ids_return) >= MIN_RETURN_SUPPORT
                else "INSUFFICIENT_SUPPORT"
            ),

        "return_after_out_of_view_tracks":
            int(len(ids_return_oov)),
    }

    print(
        case,
        json.dumps(result),
        flush=True,
    )

    return result


def mode_support(args):
    ev = load_ev(args.eval)

    root = Path(args.root)
    out = Path(args.out)

    report = {
        "protocol":
            "geometry-only pre-registered support construction",
        "cases":
            {},
    }

    for case in ["santa", "tree"]:
        report["cases"][case] = build_support(
            ev,
            root,
            out,
            case,
        )

    (
        out / "support_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n"
    )

    print(
        "SUPPORT_CONSTRUCTION_DONE",
        flush=True,
    )


def source_features(ev, root, case):
    cfg = ev.CASES[case]

    tracks, vis = load_case_arrays(
        ev,
        root,
        case,
    )

    n = tracks.shape[1]

    source = ev.read_rgb_image(
        root / cfg["source"]
    )

    src_valid = source_patch_valid(
        tracks[0]
    )

    src_patch = np.full(
        (n, 8, 8, 3),
        np.nan,
        np.float32,
    )

    good = np.where(
        src_valid
    )[0]

    src_patch[good] = ev.sample_patches(
        source,
        tracks[0, good],
    )

    src_lab = np.full(
        (n, 3),
        np.nan,
        np.float32,
    )

    src_lab[good] = ev.patch_mean_lab(
        src_patch[good]
    )

    return (
        tracks,
        vis,
        src_valid,
        src_patch,
        src_lab,
    )


def appearance_errors(
    ev,
    root,
    suite,
    case,
):
    (
        tracks,
        vis,
        src_valid,
        src_patch,
        src_lab,
    ) = source_features(
        ev,
        root,
        case,
    )

    n = tracks.shape[1]

    paths = method_paths(
        ev,
        root,
        suite,
        case,
    )

    errors = {
        "rw": {},
        "v3d": {},
    }

    for method, path in paths.items():
        print(
            "APPEARANCE_LOAD",
            case,
            method,
            path,
            flush=True,
        )

        video = ev.read_video_common(
            path
        )

        for t in ANCHORS:
            centers = tracks[t].copy()

            centers[:, 1] *= (
                464.0 / 480.0
            )

            valid = (
                src_valid
                & vis[t]
                & future_patch_valid(
                    tracks[t]
                )
            )

            ids = np.where(
                valid
            )[0]

            patch = ev.sample_patches(
                video[t],
                centers[ids],
            )

            lab = np.linalg.norm(
                ev.patch_mean_lab(
                    patch
                )
                - src_lab[ids],
                axis=1,
            )

            full = np.full(
                n,
                np.nan,
                np.float64,
            )

            full[ids] = lab

            errors[method][t] = full

        del video

    return errors, paths, src_valid


def mode_appearance(args):
    ev = load_ev(args.eval)

    root = Path(args.root)
    suite = Path(args.suite)
    out = Path(args.out)

    ref = json.loads(
        Path(args.ref).read_text()
    )

    report = {
        "protocol":
            "pre-registered balanced temporal TC-MAR",

        "primary_endpoint":
            "complete-case persistent-visible Late 44..80",

        "mechanistic_secondary":
            "Return-after-Occlusion t=80",

        "sign":
            "RW - V3D; positive means V3D better",

        "bootstrap_unit":
            "whole material track",

        "cases":
            {},
    }

    all_errors = {}

    gate_pass = True

    # ========================================================
    # SHA + reproduction gate.
    # ========================================================

    for case in [
        "santa",
        "tree",
    ]:
        errors, paths, src_valid = (
            appearance_errors(
                ev,
                root,
                suite,
                case,
            )
        )

        all_errors[case] = errors

        sha_gate = {}

        for method in ["rw", "v3d"]:
            observed_sha = sha256(
                paths[method]
            )

            expected_sha = (
                ref["cases"][case]
                ["methods"][method]
                ["sha256"]
            )

            ok = (
                observed_sha
                == expected_sha
            )

            sha_gate[method] = {
                "observed":
                    observed_sha,

                "expected":
                    expected_sha,

                "pass":
                    ok,
            }

            gate_pass &= ok

        repro = {}

        agg = {}

        counts = {}

        for method in ["rw", "v3d"]:
            value, count = track_mean(
                errors[method],
                ANCHORS,
            )

            agg[method] = value
            counts[method] = count

            valid = count > 0

            observed = float(
                value[valid].mean()
            )

            expected = float(
                ref["cases"][case]
                ["methods"][method]
                ["tc_mar_lab"]["mean"]
            )

            ok = (
                abs(
                    observed
                    - expected
                )
                <= 0.002
            )

            repro[method] = {
                "observed":
                    observed,

                "expected":
                    expected,

                "abs_error":
                    abs(
                        observed
                        - expected
                    ),

                "pass":
                    ok,
            }

            gate_pass &= ok

        assert np.array_equal(
            counts["rw"],
            counts["v3d"],
        )

        valid = (
            (counts["rw"] > 0)
            & np.isfinite(agg["rw"])
            & np.isfinite(agg["v3d"])
        )

        diff = (
            agg["rw"][valid]
            - agg["v3d"][valid]
        )

        ci = ev.bootstrap_mean_ci(
            diff
        )

        observed_decision = (
            rw_v3d_decision(
                ci
            )
        )

        expected_block = (
            ref["cases"][case]
            ["vs_realwonder"]["v3d"]
        )

        expected_decision = (
            expected_block["decision"]
        )

        expected_ci = (
            expected_block[
                "bootstrap_95_ci"
            ]
        )

        ci_pass = (
            abs(ci[0] - expected_ci[0])
            <= 0.01
            and
            abs(ci[1] - expected_ci[1])
            <= 0.01
        )

        decision_pass = (
            observed_decision
            == expected_decision
        )

        gate_pass &= (
            ci_pass
            and decision_pass
        )

        report["cases"][case] = {
            "sha_gate":
                sha_gate,

            "reproduction":
                repro,

            "reproduction_rw_minus_v3d":
                float(diff.mean()),

            "reproduction_ci":
                ci,

            "reproduction_decision":
                observed_decision,

            "expected_decision":
                expected_decision,

            "ci_pass":
                ci_pass,

            "decision_pass":
                decision_pass,
        }

    report["reproduction_gate_pass"] = (
        bool(gate_pass)
    )

    if not gate_pass:
        (
            out / "temporal_tc_mar.json"
        ).write_text(
            json.dumps(
                report,
                indent=2,
            ) + "\n"
        )

        raise RuntimeError(
            "HISTORICAL_REPRODUCTION_GATE_FAILED"
        )

    print(
        "HISTORICAL_REPRODUCTION_GATE_PASS",
        flush=True,
    )

    # ========================================================
    # Balanced temporal analysis.
    # ========================================================

    late_case_diff = {}
    early_case_diff = {}

    return_eligible = {}

    for case in [
        "santa",
        "tree",
    ]:
        errors = all_errors[case]

        balanced_ids = np.load(
            out
            / f"{case}_balanced_track_ids.npy"
        ).astype(np.int64)

        return_ids = np.load(
            out
            / f"{case}_return_occlusion_track_ids.npy"
        ).astype(np.int64)

        anchors_report = {}

        for t in ANCHORS:
            rw = errors["rw"][t][
                balanced_ids
            ]

            v3d = errors["v3d"][t][
                balanced_ids
            ]

            assert np.isfinite(rw).all()
            assert np.isfinite(v3d).all()

            d = rw - v3d

            ci = ev.bootstrap_mean_ci(
                d
            )

            # Unbalanced exploratory.
            ru = errors["rw"][t]
            vu = errors["v3d"][t]

            valid_u = (
                np.isfinite(ru)
                & np.isfinite(vu)
            )

            du = (
                ru[valid_u]
                - vu[valid_u]
            )

            ciu = ev.bootstrap_mean_ci(
                du
            )

            anchors_report[str(t)] = {
                "balanced_n":
                    int(
                        len(balanced_ids)
                    ),

                "balanced_rw":
                    float(rw.mean()),

                "balanced_v3d":
                    float(v3d.mean()),

                "balanced_rw_minus_v3d":
                    float(d.mean()),

                "balanced_ci":
                    ci,

                "balanced_decision":
                    rw_v3d_decision(
                        ci
                    ),

                "unbalanced_n":
                    int(
                        valid_u.sum()
                    ),

                "unbalanced_rw":
                    float(
                        ru[valid_u].mean()
                    ),

                "unbalanced_v3d":
                    float(
                        vu[valid_u].mean()
                    ),

                "unbalanced_rw_minus_v3d":
                    float(
                        du.mean()
                    ),

                "unbalanced_ci":
                    ciu,
            }

        def window(ts):
            rw = np.stack(
                [
                    errors["rw"][t][
                        balanced_ids
                    ]
                    for t in ts
                ],
                axis=0,
            ).mean(axis=0)

            v3d = np.stack(
                [
                    errors["v3d"][t][
                        balanced_ids
                    ]
                    for t in ts
                ],
                axis=0,
            ).mean(axis=0)

            d = rw - v3d

            ci = ev.bootstrap_mean_ci(
                d
            )

            return {
                "n":
                    int(len(d)),

                "rw":
                    float(rw.mean()),

                "v3d":
                    float(v3d.mean()),

                "rw_minus_v3d":
                    float(d.mean()),

                "ci":
                    ci,

                "decision":
                    rw_v3d_decision(
                        ci
                    ),

                "_diff":
                    d,
            }

        early = window(EARLY)
        late = window(LATE)

        early_case_diff[case] = (
            early.pop("_diff")
        )

        late_case_diff[case] = (
            late.pop("_diff")
        )

        # ----------------------------------------------------
        # Return-after-occlusion secondary at t=80.
        # ----------------------------------------------------

        secondary = {
            "n":
                int(len(return_ids)),

            "support":
                (
                    "SUFFICIENT"
                    if len(return_ids)
                    >= MIN_RETURN_SUPPORT
                    else
                    "INSUFFICIENT_SUPPORT"
                ),
        }

        if len(return_ids) > 0:
            rw80 = errors["rw"][80][
                return_ids
            ]

            v380 = errors["v3d"][80][
                return_ids
            ]

            valid = (
                np.isfinite(rw80)
                & np.isfinite(v380)
            )

            rw80 = rw80[valid]
            v380 = v380[valid]

            d80 = rw80 - v380

            if len(d80) > 0:
                ci80 = (
                    ev.bootstrap_mean_ci(
                        d80
                    )
                )

                secondary.update({
                    "valid_n":
                        int(len(d80)),

                    "rw":
                        float(
                            rw80.mean()
                        ),

                    "v3d":
                        float(
                            v380.mean()
                        ),

                    "rw_minus_v3d":
                        float(
                            d80.mean()
                        ),

                    "ci":
                        ci80,

                    "decision":
                        rw_v3d_decision(
                            ci80
                        ),
                })

                if (
                    len(d80)
                    >= MIN_RETURN_SUPPORT
                ):
                    return_eligible[case] = (
                        d80
                    )

        report["cases"][case].update({
            "balanced_anchors":
                anchors_report,

            "early_exploratory":
                early,

            "late_primary_case_component":
                late,

            "return_after_occlusion_t80":
                secondary,
        })

    # ========================================================
    # Main case-balanced primary endpoint.
    # ========================================================

    primary_mean = float(
        (
            late_case_diff["santa"].mean()
            +
            late_case_diff["tree"].mean()
        )
        / 2.0
    )

    primary_ci = bootstrap_case_balanced(
        late_case_diff,
        nboot=ev.BOOT_N,
        seed=ev.BOOT_SEED,
    )

    report["primary_result"] = {
        "endpoint":
            "balanced persistent-visible Late TC-MAR",

        "rw_minus_v3d":
            primary_mean,

        "ci":
            primary_ci,

        "decision":
            rw_v3d_decision(
                primary_ci
            ),
    }

    early_mean = float(
        (
            early_case_diff["santa"].mean()
            +
            early_case_diff["tree"].mean()
        )
        / 2.0
    )

    early_ci = bootstrap_case_balanced(
        early_case_diff,
        nboot=ev.BOOT_N,
        seed=ev.BOOT_SEED,
    )

    report["early_exploratory_result"] = {
        "rw_minus_v3d":
            early_mean,

        "ci":
            early_ci,

        "decision":
            rw_v3d_decision(
                early_ci
            ),
    }

    # Cross-case subgroup inference only if both cases satisfy support.
    if set(return_eligible.keys()) == {
        "santa",
        "tree",
    }:
        m = float(
            (
                return_eligible["santa"].mean()
                +
                return_eligible["tree"].mean()
            )
            / 2.0
        )

        ci = bootstrap_case_balanced(
            return_eligible,
            nboot=ev.BOOT_N,
            seed=ev.BOOT_SEED,
        )

        report[
            "return_after_occlusion_cross_case"
        ] = {
            "inferential_status":
                "SUPPORTED",

            "rw_minus_v3d":
                m,

            "ci":
                ci,

            "decision":
                rw_v3d_decision(
                    ci
                ),
        }

    else:
        report[
            "return_after_occlusion_cross_case"
        ] = {
            "inferential_status":
                "DESCRIPTIVE_ONLY",

            "eligible_cases":
                sorted(
                    return_eligible.keys()
                ),
        }

    (
        out / "temporal_tc_mar.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n"
    )

    (
        out / "TEMPORAL_TCMAR_DONE.txt"
    ).write_text(
        "PASS\n"
    )

    print()
    print("PRIMARY_RESULT")
    print(
        "RW_MINUS_V3D",
        primary_mean,
    )
    print(
        "CI",
        primary_ci,
    )
    print(
        "DECISION",
        rw_v3d_decision(
            primary_ci
        ),
    )
    print(
        "TEMPORAL_TCMAR_DONE",
        flush=True,
    )


def mode_motion(args):
    ev = load_ev(args.eval)

    root = Path(args.root)
    suite = Path(args.suite)
    out = Path(args.out)

    result = {
        "protocol":
            "exploratory balanced temporal TC-ME companion",

        "bootstrap_unit":
            "whole material track",

        "cases":
            {},
    }

    for case in [
        "santa",
        "tree",
    ]:
        cfg = ev.CASES[case]

        tracks, vis = load_case_arrays(
            ev,
            root,
            case,
        )

        ids0 = np.load(
            out
            / f"{case}_balanced_track_ids.npy"
        ).astype(np.int64)

        keep = np.zeros(
            tracks.shape[1],
            dtype=bool,
        )

        keep[ids0] = True

        # Require the same identities to be valid for
        # every t-1 -> t motion transition used.
        for t in ANCHORS:
            prev = t - 1

            centers = (
                tracks[prev]
                / 2.0
            )

            valid = (
                vis[prev]
                & vis[t]
                & np.isfinite(
                    tracks[prev]
                ).all(axis=1)
                & np.isfinite(
                    tracks[t]
                ).all(axis=1)
                & (centers[:, 0] >= 0)
                & (centers[:, 0] <= 415)
                & (centers[:, 1] >= 0)
                & (centers[:, 1] <= 239)
            )

            keep &= valid

        ids = np.where(
            keep
        )[0]

        if len(ids) == 0:
            raise RuntimeError(
                f"{case}: zero balanced motion tracks"
            )

        print(
            case,
            "MOTION_TRACKS",
            len(ids),
            flush=True,
        )

        device = torch.device(
            "cuda:0"
        )

        model, transforms = (
            ev.load_raft_cached(
                device
            )
        )

        paths = method_paths(
            ev,
            root,
            suite,
            case,
        )

        errors = {}

        for method, path in (
            paths.items()
        ):
            print(
                "MOTION_LOAD",
                case,
                method,
                path,
                flush=True,
            )

            video = ev.read_video_common(
                path
            )

            method_err = {}

            with torch.inference_mode():
                for start in range(
                    0,
                    len(ANCHORS),
                    args.batch,
                ):
                    ts = ANCHORS[
                        start:
                        start + args.batch
                    ]

                    prevs = [
                        t - 1
                        for t in ts
                    ]

                    a = (
                        torch
                        .from_numpy(
                            video[prevs]
                        )
                        .permute(
                            0,
                            3,
                            1,
                            2,
                        )
                        .to(device)
                    )

                    b = (
                        torch
                        .from_numpy(
                            video[ts]
                        )
                        .permute(
                            0,
                            3,
                            1,
                            2,
                        )
                        .to(device)
                    )

                    a = F.interpolate(
                        a,
                        size=(240, 416),
                        mode="area",
                    )

                    b = F.interpolate(
                        b,
                        size=(240, 416),
                        mode="area",
                    )

                    a, b = transforms(
                        a,
                        b,
                    )

                    pred = (
                        model(a, b)[-1]
                        .float()
                        .cpu()
                        .numpy()
                    )

                    for j, t in enumerate(
                        ts
                    ):
                        prev = t - 1

                        centers = (
                            tracks[
                                prev,
                                ids,
                            ]
                            / 2.0
                        )

                        reference = (
                            (
                                tracks[t, ids]
                                - tracks[
                                    prev,
                                    ids,
                                ]
                            )
                            / 2.0
                        )

                        f = ev.bilinear_flow(
                            pred[j],
                            centers,
                        )

                        epe = np.linalg.norm(
                            f - reference,
                            axis=1,
                        )

                        method_err[t] = (
                            epe.astype(
                                np.float64
                            )
                        )

                    print(
                        "RAFT",
                        case,
                        method,
                        ts[-1],
                        "/80",
                        flush=True,
                    )

            errors[method] = (
                method_err
            )

            del video
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

        case_report = {
            "balanced_motion_tracks":
                int(len(ids)),

            "anchors":
                {},
        }

        for t in ANCHORS:
            rw = errors["rw"][t]
            v3d = errors["v3d"][t]

            d = rw - v3d

            ci = ev.bootstrap_mean_ci(
                d
            )

            case_report[
                "anchors"
            ][str(t)] = {
                "rw":
                    float(rw.mean()),

                "v3d":
                    float(v3d.mean()),

                "rw_minus_v3d":
                    float(d.mean()),

                "ci":
                    ci,

                "decision":
                    rw_v3d_decision(
                        ci
                    ),
            }

        for name, ts in {
            "early_4_40":
                EARLY,

            "late_44_80":
                LATE,
        }.items():
            rw = np.stack(
                [
                    errors["rw"][t]
                    for t in ts
                ],
                axis=0,
            ).mean(axis=0)

            v3d = np.stack(
                [
                    errors["v3d"][t]
                    for t in ts
                ],
                axis=0,
            ).mean(axis=0)

            d = rw - v3d

            ci = ev.bootstrap_mean_ci(
                d
            )

            case_report[name] = {
                "rw":
                    float(rw.mean()),

                "v3d":
                    float(v3d.mean()),

                "rw_minus_v3d":
                    float(d.mean()),

                "ci":
                    ci,

                "decision":
                    rw_v3d_decision(
                        ci
                    ),
            }

        result["cases"][case] = (
            case_report
        )

    (
        out / "temporal_motion.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ) + "\n"
    )

    (
        out / "TEMPORAL_MOTION_DONE.txt"
    ).write_text(
        "PASS\n"
    )

    print(
        "TEMPORAL_MOTION_DONE",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "support",
            "appearance",
            "motion",
        ],
    )

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--suite",
        required=True,
    )

    p.add_argument(
        "--eval",
        required=True,
    )

    p.add_argument(
        "--ref",
    )

    p.add_argument(
        "--out",
        required=True,
    )

    p.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    args = p.parse_args()

    if args.mode == "support":
        mode_support(args)

    elif args.mode == "appearance":
        if args.ref is None:
            raise RuntimeError(
                "--ref required"
            )

        mode_appearance(args)

    else:
        mode_motion(args)


if __name__ == "__main__":
    main()
