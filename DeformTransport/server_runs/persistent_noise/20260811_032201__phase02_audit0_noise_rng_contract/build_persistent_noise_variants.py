import json
from pathlib import Path

import numpy as np

DT = Path("/workspace/DeformTransport")
OUT = Path.cwd() / "variants"
OUT.mkdir(exist_ok=True)

CASES = {
    "santa": {
        "noise": DT / (
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/"
            "official_santa_81f_aligned_final_sim_20260806_234410/"
            "noises.npy"
        ),
        "track": DT / (
            "server_runs/wan_move_bridge/"
            "20260811_024330__santa_corrected_physical_visibility/"
            "santa_material_tracks_correct.npy"
        ),
        "vis": DT / (
            "server_runs/wan_move_bridge/"
            "20260811_024330__santa_corrected_physical_visibility/"
            "santa_material_visibility_correct.npy"
        ),
    },

    "tree": {
        "noise": DT / (
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/"
            "tree_official_precomputed_aligned_final_sim_20260807_185055/"
            "noises.npy"
        ),
        "track": DT / (
            "server_runs/wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_tracks_correct.npy"
        ),
        "vis": DT / (
            "server_runs/wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_visibility_correct.npy"
        ),
    },
}

ANCHORS = np.arange(0, 81, 4)
H, W, C = 60, 104, 32


def squeeze(x):
    if x.shape[0] == 1:
        return x[0]
    return x


def aggregate_write(dst_frame, target_cells, source_cells, source_frame):
    """
    target_cells: target flat cell for each active material point
    source_cells: ancestry flat cell for each active material point

    Duplicate point paths from the SAME ancestry cell count once.
    Distinct ancestry cells colliding at one target are combined with 1/sqrt(k).
    """
    flat_dst = dst_frame.reshape(H * W, C)
    flat_src = source_frame.reshape(H * W, C)

    changed = 0
    max_k = 0

    for tc in np.unique(target_cells):
        keys = np.unique(source_cells[target_cells == tc])
        k = len(keys)

        flat_dst[tc] = (
            flat_src[keys].astype(np.float32).sum(axis=0)
            / np.sqrt(float(k))
        ).astype(dst_frame.dtype)

        changed += 1
        max_k = max(max_k, k)

    return changed, max_k


report = {}

for case, cfg in CASES.items():
    print("\n" + "=" * 72)
    print(case)
    print("=" * 72)

    raw = np.load(cfg["noise"])
    tr = squeeze(np.load(cfg["track"]))
    vis = squeeze(np.load(cfg["vis"])).astype(bool)

    assert raw.shape == (81, H, W, C)
    assert tr.shape[0] == 81
    assert tr.shape[:2] == vis.shape

    x = tr[..., 0]
    y = tr[..., 1]

    cx = np.floor(x / 8.0).astype(np.int64)
    cy = np.floor(y / 8.0).astype(np.int64)

    valid = (
        np.isfinite(tr).all(axis=-1)
        & (cx >= 0) & (cx < W)
        & (cy >= 0) & (cy < H)
    )

    assert np.all(valid[vis])

    cells = cy * W + cx

    # --------------------------------------------------------
    # A1: globally persistent identity from frame 0.
    # --------------------------------------------------------
    a1 = raw.copy()
    source_cells = cells[0].copy()

    assert vis[0].all()
    assert len(np.unique(source_cells)) == len(source_cells)

    a1_stats = []

    for li, t in enumerate(ANCHORS):
        if li == 0:
            # Hard identity gate: frame 0 remains exactly A0.
            continue

        active = vis[t]
        changed, max_k = aggregate_write(
            a1[t],
            cells[t, active],
            source_cells[active],
            raw[0],
        )

        a1_stats.append({
            "latent_index": int(li),
            "pixel_frame": int(t),
            "visible_points": int(active.sum()),
            "changed_cells": int(changed),
            "max_distinct_ancestries_per_target": int(max_k),
        })

    # --------------------------------------------------------
    # A2: re-anchor every native 3-latent-frame block.
    # --------------------------------------------------------
    a2 = raw.copy()
    a2_stats = []

    for block_start in range(0, len(ANCHORS), 3):
        anchor_li = block_start
        anchor_t = int(ANCHORS[anchor_li])

        anchor_cells = cells[anchor_t]

        for li in range(
            block_start,
            min(block_start + 3, len(ANCHORS)),
        ):
            t = int(ANCHORS[li])

            if li == anchor_li:
                # Re-anchor frame itself is exactly A0.
                continue

            active = vis[anchor_t] & vis[t]

            changed, max_k = aggregate_write(
                a2[t],
                cells[t, active],
                anchor_cells[active],
                raw[anchor_t],
            )

            a2_stats.append({
                "block_anchor_latent": int(anchor_li),
                "block_anchor_pixel": int(anchor_t),
                "latent_index": int(li),
                "pixel_frame": int(t),
                "anchor_and_target_visible_points":
                    int(active.sum()),
                "changed_cells": int(changed),
                "max_distinct_ancestries_per_target":
                    int(max_k),
            })

    # Hard gates.
    assert np.array_equal(a1[0], raw[0])
    assert np.array_equal(a2[0], raw[0])

    # Non-latent-anchor raw frames must remain untouched.
    non_anchor = sorted(set(range(81)) - set(ANCHORS.tolist()))

    assert np.array_equal(
        a1[non_anchor],
        raw[non_anchor],
    )

    assert np.array_equal(
        a2[non_anchor],
        raw[non_anchor],
    )

    case_dir = OUT / case
    case_dir.mkdir(exist_ok=True)

    p1 = case_dir / "A1_full_persistent_noises.npy"
    p2 = case_dir / "A2_block3_reanchored_noises.npy"

    np.save(p1, a1)
    np.save(p2, a2)

    changed_a1 = np.any(a1 != raw, axis=-1)
    changed_a2 = np.any(a2 != raw, axis=-1)

    rec = {
        "a0": str(cfg["noise"]),
        "track": str(cfg["track"]),
        "visibility": str(cfg["vis"]),
        "shape": list(raw.shape),

        "a1": {
            "path": str(p1),
            "frame0_bitwise_equal_a0":
                bool(np.array_equal(a1[0], raw[0])),
            "non_anchor_frames_bitwise_equal_a0":
                bool(np.array_equal(a1[non_anchor], raw[non_anchor])),
            "changed_spatial_cells_total":
                int(changed_a1.sum()),
            "stats": a1_stats,
        },

        "a2": {
            "path": str(p2),
            "frame0_bitwise_equal_a0":
                bool(np.array_equal(a2[0], raw[0])),
            "non_anchor_frames_bitwise_equal_a0":
                bool(np.array_equal(a2[non_anchor], raw[non_anchor])),
            "changed_spatial_cells_total":
                int(changed_a2.sum()),
            "stats": a2_stats,
        },
    }

    report[case] = rec

    print("A1:", p1)
    print(
        "  changed spatial cells =",
        rec["a1"]["changed_spatial_cells_total"],
    )
    print("A2:", p2)
    print(
        "  changed spatial cells =",
        rec["a2"]["changed_spatial_cells_total"],
    )
    print("frame0 exact: A1=True A2=True")
    print("non-anchor exact: A1=True A2=True")


Path("persistent_noise_build_report.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print("\nPERSISTENT_NOISE_BUILD_PASS")
