import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent

TCMAR = json.loads(
    (ROOT / "temporal_tc_mar.json").read_text()
)

TCME = json.loads(
    (ROOT / "temporal_motion.json").read_text()
)

SUPPORT = json.loads(
    (ROOT / "support_report.json").read_text()
)

OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)


# ============================================================
# Publication style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ANCHORS = list(range(4, 81, 4))


def save(fig, name):
    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{name}.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Figure 1:
# Temporal TC-MAR
# ============================================================

for case in ["santa", "tree"]:

    rows = TCMAR["cases"][case]["balanced_anchors"]

    rw = np.array([
        rows[str(t)]["balanced_rw"]
        for t in ANCHORS
    ])

    v3d = np.array([
        rows[str(t)]["balanced_v3d"]
        for t in ANCHORS
    ])

    fig, ax = plt.subplots(
        figsize=(3.35, 2.35)
    )

    ax.plot(
        ANCHORS,
        rw,
        marker="o",
        label="RealWonder",
    )

    ax.plot(
        ANCHORS,
        v3d,
        marker="s",
        linestyle="--",
        label="V3D",
    )

    # Early/Late boundary
    ax.axvline(
        42,
        linestyle=":",
        linewidth=1.0,
    )

    ax.text(
        22,
        ax.get_ylim()[1],
        "Early",
        ha="center",
        va="top",
        fontsize=8,
    )

    ax.text(
        62,
        ax.get_ylim()[1],
        "Late",
        ha="center",
        va="top",
        fontsize=8,
    )

    ax.set_xlabel("Prediction horizon (frame)")
    ax.set_ylabel("TC-MAR ↓")

    ax.set_xticks(
        [4, 20, 40, 60, 80]
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    save(
        fig,
        f"tcmar_temporal_{case}",
    )


# ============================================================
# Figure 2:
# TC-MAR difference + bootstrap CI
#
# positive = V3D better
# negative = RealWonder better
# ============================================================

for case in ["santa", "tree"]:

    rows = TCMAR["cases"][case]["balanced_anchors"]

    diff = np.array([
        rows[str(t)]["balanced_rw_minus_v3d"]
        for t in ANCHORS
    ])

    lo = np.array([
        rows[str(t)]["balanced_ci"][0]
        for t in ANCHORS
    ])

    hi = np.array([
        rows[str(t)]["balanced_ci"][1]
        for t in ANCHORS
    ])

    fig, ax = plt.subplots(
        figsize=(3.35, 2.35)
    )

    ax.axhline(
        0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.plot(
        ANCHORS,
        diff,
        marker="o",
        label="RW − V3D",
    )

    ax.fill_between(
        ANCHORS,
        lo,
        hi,
        alpha=0.18,
        linewidth=0,
        label="95% bootstrap CI",
    )

    ax.axvline(
        42,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_xlabel("Prediction horizon (frame)")
    ax.set_ylabel("Δ TC-MAR (RW − V3D)")

    ax.set_xticks(
        [4, 20, 40, 60, 80]
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    save(
        fig,
        f"tcmar_difference_{case}",
    )


# ============================================================
# Figure 3:
# Temporal TC-ME
# ============================================================

for case in ["santa", "tree"]:

    rows = TCME["cases"][case]["anchors"]

    rw = np.array([
        rows[str(t)]["rw"]
        for t in ANCHORS
    ])

    v3d = np.array([
        rows[str(t)]["v3d"]
        for t in ANCHORS
    ])

    fig, ax = plt.subplots(
        figsize=(3.35, 2.35)
    )

    ax.plot(
        ANCHORS,
        rw,
        marker="o",
        label="RealWonder",
    )

    ax.plot(
        ANCHORS,
        v3d,
        marker="s",
        linestyle="--",
        label="V3D",
    )

    ax.axvline(
        42,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_xlabel("Prediction horizon (frame)")
    ax.set_ylabel("TC-ME ↓")

    ax.set_xticks(
        [4, 20, 40, 60, 80]
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    save(
        fig,
        f"tcme_temporal_{case}",
    )


# ============================================================
# Figure 4:
# Geometry-defined track composition
# ============================================================

cases = ["Santa", "Tree"]

all_tracks = np.array([
    SUPPORT["cases"]["santa"]["all_tracks"],
    SUPPORT["cases"]["tree"]["all_tracks"],
])

balanced = np.array([
    SUPPORT["cases"]["santa"]["balanced_complete_case_tracks"],
    SUPPORT["cases"]["tree"]["balanced_complete_case_tracks"],
])

returned = np.array([
    SUPPORT["cases"]["santa"]["return_after_occlusion_tracks"],
    SUPPORT["cases"]["tree"]["return_after_occlusion_tracks"],
])

x = np.arange(len(cases))

width = 0.22

fig, ax = plt.subplots(
    figsize=(3.35, 2.35)
)

ax.bar(
    x - width,
    all_tracks,
    width,
    label="All source tracks",
)

ax.bar(
    x,
    balanced,
    width,
    label="Always visible",
)

ax.bar(
    x + width,
    returned,
    width,
    label="Return after visibility loss",
)

ax.set_xticks(x)
ax.set_xticklabels(cases)

ax.set_ylabel("Number of material tracks")

ax.legend(
    frameon=False,
    fontsize=7,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

save(
    fig,
    "track_support_composition",
)


# ============================================================
# Figure 5:
# Tree return-after-occlusion TC-MAR
# ============================================================

return_result = (
    TCMAR["cases"]["tree"]
    ["return_after_occlusion_t80"]
)

rw = return_result["rw"]
v3d = return_result["v3d"]

fig, ax = plt.subplots(
    figsize=(2.7, 2.35)
)

ax.bar(
    [0, 1],
    [rw, v3d],
)

ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["RealWonder", "V3D"]
)

ax.set_ylabel("TC-MAR at frame 80 ↓")

ax.text(
    0.5,
    max(rw, v3d) * 1.04,
    "TIE (95% CI crosses 0)",
    ha="center",
    fontsize=8,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

save(
    fig,
    "tree_return_after_occlusion",
)


print("DONE")
print("Figures saved to:")
print(OUT)


