import json
from pathlib import Path
import numpy as np

CASES={
    "santa":{
        "track":
        Path(
            "/workspace/DeformTransport/server_runs/"
            "wan_move_bridge/"
            "20260811_024330__santa_corrected_physical_visibility/"
            "santa_material_tracks_correct.npy"
        ),
        "visibility":
        Path(
            "/workspace/DeformTransport/server_runs/"
            "wan_move_bridge/"
            "20260811_024330__santa_corrected_physical_visibility/"
            "santa_material_visibility_correct.npy"
        ),
    },

    "tree":{
        "track":
        Path(
            "/workspace/DeformTransport/server_runs/"
            "wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_tracks_correct.npy"
        ),
        "visibility":
        Path(
            "/workspace/DeformTransport/server_runs/"
            "wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_visibility_correct.npy"
        ),
    },
}

report={}

for case,c in CASES.items():

    tr=np.load(c["track"])
    vis=np.load(c["visibility"]).astype(bool)

    if tr.ndim==4:
        tr=tr[0]

    if vis.ndim==3:
        vis=vis[0]

    assert tr.shape[:2]==vis.shape
    assert tr.shape[0]==81
    assert tr.shape[-1]==2

    x=tr[...,0]
    y=tr[...,1]

    finite=np.isfinite(tr).all(-1)

    inframe=(
        finite
        & (x>=0) & (x<832)
        & (y>=0) & (y<480)
    )

    cx=np.floor(x/8).astype(np.int64)
    cy=np.floor(y/8).astype(np.int64)

    valid_cell=(
        inframe
        & (cx>=0) & (cx<104)
        & (cy>=0) & (cy<60)
    )

    visible_valid=vis & valid_cell

    anchors=list(range(0,81,4))

    anchor_stats=[]

    for t in anchors:

        active=np.flatnonzero(visible_valid[t])

        cells=cy[t,active]*104+cx[t,active]

        uniq,counts=np.unique(
            cells,
            return_counts=True,
        )

        anchor_stats.append({
            "t":t,
            "visible_points":int(len(active)),
            "occupied_cells":int(len(uniq)),
            "collision_points":int(
                np.sum(np.maximum(counts-1,0))
            ),
            "max_points_per_cell":int(
                counts.max()
                if len(counts)
                else 0
            ),
        })

    rec={
        "track_shape":list(tr.shape),
        "visibility_shape":list(vis.shape),

        "finite_fraction":
            float(finite.mean()),

        "visible_in_noise_lattice_fraction":
            float(
                visible_valid.sum()
                /
                max(vis.sum(),1)
            ),

        "source_visible_points":
            int(visible_valid[0].sum()),

        "source_unique_cells":
            int(
                len(
                    np.unique(
                        cy[0,visible_valid[0]]*104
                        +cx[0,visible_valid[0]]
                    )
                )
            ),

        "anchor_stats":anchor_stats,
    }

    report[case]=rec

    print("\n====",case,"====")
    print("tracks:",tr.shape)
    print(
        "visible -> valid lattice fraction:",
        rec["visible_in_noise_lattice_fraction"],
    )
    print(
        "source points / cells:",
        rec["source_visible_points"],
        "/",
        rec["source_unique_cells"],
    )

    for x in anchor_stats:
        print(
            f't={x["t"]:02d}',
            "points=",x["visible_points"],
            "cells=",x["occupied_cells"],
            "collisions=",x["collision_points"],
            "max/cell=",x["max_points_per_cell"],
        )

Path("material_noise_lattice_audit.json").write_text(
    json.dumps(report,indent=2)+"\n"
)

print("\nMATERIAL_NOISE_LATTICE_AUDIT_DONE")
