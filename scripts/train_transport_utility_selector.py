from pathlib import Path
import hashlib
import json
import math
import re
import random

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Frozen experiment design
# ============================================================

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(
    "/workspace/DeformTransport/server_runs/"
    "20260804_234925_autonomous_deformtransport"
)

DEV = ROOT / "12_soft_transport_dev"
PREP = ROOT / "prepared_inputs"

OUT = Path(
    sorted(
        Path(
            "/workspace/DeformTransport/server_runs/"
            "utility_selector"
        ).glob("*__tree_train_santa_val")
    )[-1]
)

# ------------------------------------------------------------
# Tree = TRAIN
# ------------------------------------------------------------

TREE = {
    "artifact": (
        DEV
        / "20260807_212536__tree__quality_ramp4_frozen_santa_recipe"
        / "tree_quality_ramp4_full_generation.pt"
    ),

    "sim_dir": (
        PREP
        / "tree_official_precomputed_aligned_final_sim_20260807_185055"
    ),

    "baseline_dir": (
        DEV
        / "20260807_203228__tree__realwonder_baseline_seed0"
    ),

    "correct_dir": (
        DEV
        / "20260807_235221__tree__quality_condition_residual_seed0"
    ),

    "shuffled_dir": (
        DEV
        / "20260808_010603__tree__shuffled_condition_energy_matched_seed0"
    ),
}

# ------------------------------------------------------------
# Santa = HELD-OUT VALIDATION
# ------------------------------------------------------------

SANTA = {
    "artifact": (
        DEV
        / "20260806_232607__full_generation_compatible_candidates"
        / "quality_ramp4_full_generation.pt"
    ),

    "sim_dir": (
        PREP
        / "official_santa_81f_aligned_final_sim_20260806_234410"
    ),

    "baseline_dir": (
        DEV
        / "20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation"
        / "baseline"
    ),

    "correct_dir": (
        DEV
        / "20260808_012730__santa__frozen_tree_condition075_crosscase"
        / "correct"
    ),

    "shuffled_dir": (
        DEV
        / "20260808_012730__santa__frozen_tree_condition075_crosscase"
        / "shuffled_matched"
    ),
}


# ============================================================
# Fixed model/training protocol.
#
# These values are fixed before Santa evaluation and are not
# chosen using SandHouse.
# ============================================================

HIDDEN1 = 32
HIDDEN2 = 16

EPOCHS = 120
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-4

GO_AUC = 0.60

FEATURE_NAMES = (
    [f"sim_latent_{i}" for i in range(16)]
    + [f"correct_residual_{i}" for i in range(16)]
    + [
        "residual_abs_mean",
        "residual_l2",
        "log1p_contribution_count",
        "normalized_time",
    ]
)

assert len(FEATURE_NAMES) == 36


# ============================================================
# Helpers
# ============================================================

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def unique_mp4(directory):
    files = sorted(directory.glob("*.mp4"))

    if len(files) != 1:
        raise RuntimeError(
            f"{directory}: expected exactly one mp4, "
            f"got {files}"
        )

    return files[0]


def read_text_if_exists(path):
    if path.exists():
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    return ""


def parse_scale(run_dir, mode):
    text = ""

    for name in (
        "stdout.log",
        "inference_command.txt",
        "run_contract.txt",
    ):
        text += "\n" + read_text_if_exists(
            run_dir / name
        )

    if mode == "correct":
        expected = "Loading correct artifact-local"
    else:
        expected = "Loading shuffled artifact-local"

    if expected not in text:
        raise RuntimeError(
            f"{run_dir}: expected {mode} "
            "artifact-local residual not found"
        )

    if (
        "Applied condition-space transport"
        not in text
        and "--transport_injection_mode condition_residual"
        not in text
    ):
        raise RuntimeError(
            f"{run_dir}: not verified as "
            "condition_residual"
        )

    matches = re.findall(
        r"transport residual scale:\s*"
        r"([-+0-9.eE]+)",
        text,
    )

    if matches:
        return float(matches[-1])

    matches = re.findall(
        r"--transport_injection_scale\s+"
        r"([-+0-9.eE]+)",
        text,
    )

    if matches:
        return float(matches[-1])

    raise RuntimeError(
        f"{run_dir}: cannot parse injection scale"
    )


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame)

    cap.release()

    if len(frames) != 81:
        raise RuntimeError(
            f"{path}: expected 81 frames, "
            f"got {len(frames)}"
        )

    x = np.stack(frames)

    if x.shape != (81, 480, 832, 3):
        raise RuntimeError(
            f"{path}: unexpected shape {x.shape}"
        )

    return x


def read_sim_frames(sim_dir):
    frame_dir = sim_dir / "frames"

    paths = sorted(
        frame_dir.glob("*.png")
    )

    if len(paths) != 81:
        paths = sorted(
            frame_dir.glob("*")
        )

    frames = []

    for p in paths:
        x = cv2.imread(
            str(p),
            cv2.IMREAD_COLOR,
        )

        if x is None:
            continue

        if x.shape[:2] != (480, 832):
            x = cv2.resize(
                x,
                (832, 480),
                interpolation=cv2.INTER_AREA,
            )

        frames.append(x)

    if len(frames) != 81:
        raise RuntimeError(
            f"{sim_dir}: expected 81 readable "
            f"simulation frames, got {len(frames)}"
        )

    return np.stack(frames)


def patch_l1(video, sim, frame_ids):
    """
    Pixel L1 -> exact 8x8 latent cells.

    480/60 = 8
    832/104 = 8
    """

    out = []

    for frame_id in frame_ids:
        d = np.abs(
            video[frame_id].astype(np.float32)
            - sim[frame_id].astype(np.float32)
        )

        # mean RGB first
        d = d.mean(axis=2)

        # exact latent grid pooling
        d = d.reshape(
            60, 8,
            104, 8,
        ).mean(axis=(1, 3))

        out.append(d)

    return np.stack(out)


def residual_from_artifact(
    art,
    branch,
):
    key = (
        f"{branch}_transport_residual"
    )

    if key in art:
        return art[key].float()

    fused_key = (
        f"{branch}_fused_latent"
    )

    if (
        fused_key in art
        and "target_latent" in art
    ):
        return (
            art[fused_key].float()
            - art["target_latent"].float()
        )

    raise RuntimeError(
        f"cannot recover {branch} residual"
    )


def rankdata_average(x):
    x = np.asarray(x)
    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )

    sorted_x = x[order]

    i = 0

    while i < len(x):
        j = i + 1

        while (
            j < len(x)
            and sorted_x[j] == sorted_x[i]
        ):
            j += 1

        rank = (
            (i + 1) + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = rank

        i = j

    return ranks


def roc_auc(y, score):
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score)

    pos = int(y.sum())
    neg = int(len(y) - pos)

    if pos == 0 or neg == 0:
        return float("nan")

    ranks = rankdata_average(score)

    sum_pos = ranks[y == 1].sum()

    auc = (
        sum_pos
        - pos * (pos + 1) / 2.0
    ) / (pos * neg)

    return float(auc)


def average_precision(y, score):
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score)

    order = np.argsort(-score)
    yy = y[order]

    positives = int(yy.sum())

    if positives == 0:
        return float("nan")

    tp = np.cumsum(yy)
    precision = (
        tp / np.arange(1, len(yy) + 1)
    )

    return float(
        precision[yy == 1].sum()
        / positives
    )


def spearman(x, y):
    rx = rankdata_average(x)
    ry = rankdata_average(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = math.sqrt(
        float(np.dot(rx, rx))
        * float(np.dot(ry, ry))
    )

    if denom == 0:
        return float("nan")

    return float(
        np.dot(rx, ry) / denom
    )


# ============================================================
# Construct one case
# ============================================================

def build_case(name, cfg):
    print()
    print(
        "=" * 72
    )
    print(
        f"BUILD CASE: {name}"
    )
    print(
        "=" * 72
    )

    for k in (
        "artifact",
        "sim_dir",
        "baseline_dir",
        "correct_dir",
        "shuffled_dir",
    ):
        if not cfg[k].exists():
            raise FileNotFoundError(
                cfg[k]
            )

    baseline_video = unique_mp4(
        cfg["baseline_dir"]
    )

    correct_video = unique_mp4(
        cfg["correct_dir"]
    )

    shuffled_video = unique_mp4(
        cfg["shuffled_dir"]
    )

    correct_scale = parse_scale(
        cfg["correct_dir"],
        "correct",
    )

    shuffled_scale = parse_scale(
        cfg["shuffled_dir"],
        "shuffled",
    )

    print(
        "artifact =",
        cfg["artifact"],
    )
    print(
        "baseline =",
        baseline_video,
    )
    print(
        "correct =",
        correct_video,
    )
    print(
        "shuffled =",
        shuffled_video,
    )
    print(
        "correct_scale =",
        correct_scale,
    )
    print(
        "shuffled_scale =",
        shuffled_scale,
    )

    art = torch.load(
        cfg["artifact"],
        map_location="cpu",
        weights_only=False,
    )

    target = (
        art["target_latent"]
        .float()
        .contiguous()
    )

    correct_raw = (
        residual_from_artifact(
            art,
            "correct",
        )
        .contiguous()
    )

    shuffled_raw = (
        residual_from_artifact(
            art,
            "shuffled",
        )
        .contiguous()
    )

    mask = (
        art["transport_mask"]
        .bool()
        .contiguous()
    )

    if (
        tuple(target.shape)
        != (1, 21, 16, 60, 104)
    ):
        raise RuntimeError(
            f"{name}: target shape "
            f"{tuple(target.shape)}"
        )

    if (
        tuple(correct_raw.shape)
        != tuple(target.shape)
    ):
        raise RuntimeError(
            f"{name}: correct residual shape"
        )

    if (
        tuple(shuffled_raw.shape)
        != tuple(target.shape)
    ):
        raise RuntimeError(
            f"{name}: shuffled residual shape"
        )

    if (
        tuple(mask.shape)
        != (21, 1, 60, 104)
    ):
        raise RuntimeError(
            f"{name}: mask shape "
            f"{tuple(mask.shape)}"
        )

    correct = (
        correct_raw
        * correct_scale
    )

    shuffled = (
        shuffled_raw
        * shuffled_scale
    )

    # slot0 is never a future transport target.
    active = mask[:, 0].clone()
    active[0] = False

    # contribution_count is optional in some old artifacts.
    if "contribution_count" in art:
        count = (
            art["contribution_count"]
            .float()
        )

        if count.ndim == 4:
            count = count[:, 0]

        if tuple(count.shape) != (
            21, 60, 104
        ):
            raise RuntimeError(
                f"{name}: contribution_count "
                f"shape {tuple(count.shape)}"
            )
    else:
        count = mask[:, 0].float()

    indices = (
        art.get(
            "latent_frame_indices",
            torch.arange(
                0,
                81,
                4,
            ),
        )
        .long()
    )

    if not torch.equal(
        indices,
        torch.arange(0, 81, 4),
    ):
        raise RuntimeError(
            f"{name}: temporal contract mismatch"
        )

    sim = read_sim_frames(
        cfg["sim_dir"]
    )

    base_vid = read_video(
        baseline_video
    )

    corr_vid = read_video(
        correct_video
    )

    shuf_vid = read_video(
        shuffled_video
    )

    frame_ids = indices.tolist()

    e_rw = patch_l1(
        base_vid,
        sim,
        frame_ids,
    )

    e_c = patch_l1(
        corr_vid,
        sim,
        frame_ids,
    )

    e_s = patch_l1(
        shuf_vid,
        sim,
        frame_ids,
    )

    # --------------------------------------------------------
    # Labels:
    #
    # material-specific beneficial transport iff
    #
    # correct beats baseline AND shuffled.
    # --------------------------------------------------------

    label_grid = (
        (e_c < e_rw)
        & (e_c < e_s)
    )

    utility_grid = (
        e_rw - e_c
    )

    identity_margin_grid = (
        e_s - e_c
    )

    # --------------------------------------------------------
    # Features.
    # --------------------------------------------------------

    z = (
        target[0]
        .permute(0, 2, 3, 1)
    )

    r = (
        correct[0]
        .permute(0, 2, 3, 1)
    )

    abs_mean = (
        r.abs().mean(
            dim=-1,
            keepdim=True,
        )
    )

    l2norm = (
        torch.linalg.vector_norm(
            r,
            dim=-1,
            keepdim=True,
        )
    )

    log_count = (
        torch.log1p(
            count.clamp_min(0)
        )
        .unsqueeze(-1)
    )

    time = (
        torch.linspace(
            0.0,
            1.0,
            21,
        )
        .view(21, 1, 1, 1)
        .expand(
            21,
            60,
            104,
            1,
        )
    )

    feat = torch.cat([
        z,
        r,
        abs_mean,
        l2norm,
        log_count,
        time,
    ], dim=-1)

    if feat.shape[-1] != 36:
        raise RuntimeError(
            f"{name}: feature dim "
            f"{feat.shape[-1]}"
        )

    active_np = active.numpy()

    X = (
        feat[active]
        .float()
        .numpy()
    )

    y = (
        label_grid[active_np]
        .astype(np.float32)
    )

    utility = (
        utility_grid[active_np]
        .astype(np.float32)
    )

    identity_margin = (
        identity_margin_grid[active_np]
        .astype(np.float32)
    )

    if len(X) == 0:
        raise RuntimeError(
            f"{name}: no active cells"
        )

    prevalence = float(
        y.mean()
    )

    print(
        "active cells =",
        len(X),
    )
    print(
        "positive cells =",
        int(y.sum()),
    )
    print(
        "positive prevalence =",
        prevalence,
    )
    print(
        "utility mean =",
        float(utility.mean()),
    )
    print(
        "identity margin mean =",
        float(identity_margin.mean()),
    )

    return {
        "name":
            name,

        "X":
            X,

        "y":
            y,

        "utility":
            utility,

        "identity_margin":
            identity_margin,

        "correct_scale":
            correct_scale,

        "shuffled_scale":
            shuffled_scale,

        "artifact":
            str(cfg["artifact"]),

        "baseline_video":
            str(baseline_video),

        "correct_video":
            str(correct_video),

        "shuffled_video":
            str(shuffled_video),

        "sim_dir":
            str(cfg["sim_dir"]),
    }


# ============================================================
# Build Tree and Santa.
#
# IMPORTANT: no SandHouse paths or data are read anywhere.
# ============================================================

tree = build_case(
    "tree_train",
    TREE,
)

santa = build_case(
    "santa_heldout_validation",
    SANTA,
)


# ============================================================
# Standardization: TREE ONLY
# ============================================================

mu = tree["X"].mean(
    axis=0,
    dtype=np.float64,
).astype(np.float32)

sigma = tree["X"].std(
    axis=0,
    dtype=np.float64,
).astype(np.float32)

sigma = np.maximum(
    sigma,
    1e-6,
)

Xtr = (
    (tree["X"] - mu)
    / sigma
).astype(np.float32)

Xva = (
    (santa["X"] - mu)
    / sigma
).astype(np.float32)

ytr = tree["y"].astype(
    np.float32
)

yva = santa["y"].astype(
    np.float32
)


# ============================================================
# Model
# ============================================================

class UtilitySelector(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                36,
                HIDDEN1,
            ),
            nn.GELU(),

            nn.Linear(
                HIDDEN1,
                HIDDEN2,
            ),
            nn.GELU(),

            nn.Linear(
                HIDDEN2,
                1,
            ),
        )

    def forward(self, x):
        return (
            self.net(x)
            .squeeze(-1)
        )


model = UtilitySelector()


# ============================================================
# Training: TREE ONLY.
# No Santa-dependent early stopping.
# ============================================================

Xtr_t = torch.from_numpy(Xtr)
ytr_t = torch.from_numpy(ytr)

pos = float(
    ytr_t.sum()
)

neg = float(
    len(ytr_t) - pos
)

if pos <= 0 or neg <= 0:
    raise RuntimeError(
        "Tree labels are degenerate"
    )

pos_weight_value = (
    neg / pos
)

criterion = (
    nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight_value,
            dtype=torch.float32,
        )
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

generator = torch.Generator()
generator.manual_seed(SEED)

print()
print(
    "===== TRAIN TREE ONLY ====="
)
print(
    "samples =",
    len(Xtr_t),
)
print(
    "pos_weight =",
    pos_weight_value,
)


for epoch in range(EPOCHS):
    permutation = torch.randperm(
        len(Xtr_t),
        generator=generator,
    )

    total_loss = 0.0
    total_n = 0

    model.train()

    for start in range(
        0,
        len(permutation),
        BATCH_SIZE,
    ):
        idx = permutation[
            start:start + BATCH_SIZE
        ]

        xb = Xtr_t[idx]
        yb = ytr_t[idx]

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(xb)

        loss = criterion(
            logits,
            yb,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            float(loss)
            * len(idx)
        )

        total_n += len(idx)

    if (
        epoch == 0
        or (epoch + 1) % 20 == 0
        or epoch == EPOCHS - 1
    ):
        print(
            f"epoch={epoch+1:03d} "
            f"loss={total_loss/total_n:.6f}"
        )


# ============================================================
# Evaluation.
# ============================================================

@torch.inference_mode()
def predict(X):
    model.eval()

    x = torch.from_numpy(X)

    probs = []

    for start in range(
        0,
        len(x),
        8192,
    ):
        logits = model(
            x[start:start+8192]
        )

        probs.append(
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

    return np.concatenate(probs)


p_tree = predict(Xtr)
p_santa = predict(Xva)


def metrics(
    y,
    p,
    utility,
):
    auc = roc_auc(
        y,
        p,
    )

    ap = average_precision(
        y,
        p,
    )

    rho = spearman(
        p,
        utility,
    )

    # Top 20% ranking diagnostic.
    n = len(p)
    k = max(
        1,
        int(round(0.20 * n)),
    )

    top_idx = np.argsort(
        -p
    )[:k]

    bottom_idx = np.argsort(
        p
    )[:k]

    return {
        "samples":
            int(n),

        "positive_prevalence":
            float(
                np.mean(y)
            ),

        "roc_auc":
            float(auc),

        "average_precision":
            float(ap),

        "spearman_pred_vs_continuous_utility":
            float(rho),

        "mean_utility_all":
            float(
                np.mean(utility)
            ),

        "mean_utility_top20_predicted":
            float(
                np.mean(
                    utility[top_idx]
                )
            ),

        "mean_utility_bottom20_predicted":
            float(
                np.mean(
                    utility[bottom_idx]
                )
            ),

        "mean_probability":
            float(
                np.mean(p)
            ),
    }


tree_metrics = metrics(
    tree["y"],
    p_tree,
    tree["utility"],
)

santa_metrics = metrics(
    santa["y"],
    p_santa,
    santa["utility"],
)


# Pre-registered GO condition.
go = bool(
    santa_metrics["roc_auc"]
    >= GO_AUC
    and
    santa_metrics[
        "spearman_pred_vs_continuous_utility"
    ] > 0.0
)


print()
print(
    "===== TREE TRAIN METRICS ====="
)

print(
    json.dumps(
        tree_metrics,
        indent=2,
    )
)

print()
print(
    "===== SANTA HELD-OUT METRICS ====="
)

print(
    json.dumps(
        santa_metrics,
        indent=2,
    )
)

print()
print(
    "===== PRE-REGISTERED DECISION ====="
)

print(
    "GO_AUC_THRESHOLD =",
    GO_AUC,
)

print(
    "GO requires Spearman > 0"
)

print(
    "DECISION =",
    "GO" if go else "STOP",
)


# ============================================================
# Save frozen model.
# ============================================================

checkpoint = {
    "method":
        "Transport Utility Selector",

    "architecture":
        "36-32-16-1 MLP",

    "feature_names":
        FEATURE_NAMES,

    "model_state_dict":
        model.state_dict(),

    "feature_mean":
        torch.from_numpy(mu),

    "feature_std":
        torch.from_numpy(sigma),

    "tree_only_training":
        True,

    "santa_heldout_validation":
        True,

    "sandhouse_used":
        False,

    "seed":
        SEED,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "pos_weight":
        pos_weight_value,

    "go_auc_threshold":
        GO_AUC,

    "tree_metrics":
        tree_metrics,

    "santa_metrics":
        santa_metrics,

    "go":
        go,
}

MODEL_PATH = (
    OUT
    / "transport_utility_selector.pt"
)

torch.save(
    checkpoint,
    MODEL_PATH,
)


report = {
    "method":
        "Transport Utility Selector",

    "training_supervision":
        (
            "positive iff Correct patch error "
            "< RealWonder patch error AND "
            "< Shuffled patch error"
        ),

    "proxy_warning":
        (
            "labels use geometry-aligned "
            "simulation RGB as engineering proxy, "
            "not real-video ground truth"
        ),

    "features":
        FEATURE_NAMES,

    "training_case":
        {
            k: v
            for k, v in tree.items()
            if k not in (
                "X",
                "y",
                "utility",
                "identity_margin",
            )
        },

    "validation_case":
        {
            k: v
            for k, v in santa.items()
            if k not in (
                "X",
                "y",
                "utility",
                "identity_margin",
            )
        },

    "tree_metrics":
        tree_metrics,

    "santa_metrics":
        santa_metrics,

    "pre_registered_go_rule": {
        "ROC_AUC_gte":
            GO_AUC,

        "Spearman_gt":
            0.0,
    },

    "decision":
        "GO" if go else "STOP",

    "sandhouse_used":
        False,

    "model_path":
        str(MODEL_PATH),
}

REPORT_PATH = (
    OUT
    / "validation_report.json"
)

REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

SHA_PATH = (
    OUT
    / "sha256.txt"
)

sha_lines = [
    f"{sha256(MODEL_PATH)}  {MODEL_PATH}",
    f"{sha256(TREE['artifact'])}  {TREE['artifact']}",
    f"{sha256(SANTA['artifact'])}  {SANTA['artifact']}",
]

SHA_PATH.write_text(
    "\n".join(sha_lines) + "\n",
    encoding="utf-8",
)


print()
print(
    "===== SHA256 ====="
)

print(
    SHA_PATH.read_text()
)

print(
    "MODEL =",
    MODEL_PATH,
)

print(
    "REPORT =",
    REPORT_PATH,
)

if go:
    print(
        "UTILITY_SELECTOR_SANTA_HELDOUT_GO"
    )
else:
    print(
        "UTILITY_SELECTOR_SANTA_HELDOUT_STOP"
    )
