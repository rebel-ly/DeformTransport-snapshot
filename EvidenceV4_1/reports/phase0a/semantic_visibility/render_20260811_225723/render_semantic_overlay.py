import os
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Pillow compatibility:
# New Pillow uses RESAMPLE_NEAREST;
# older Pillow uses Image.NEAREST.
try:
    RESAMPLE_NEAREST = RESAMPLE_NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST


V2 = Path(os.environ["V2"])
ALDIR = Path(os.environ["ALDIR"])
EVID = Path(os.environ["EVID"])
OUT = Path(os.environ["RENDER_OUT"])

MANIFEST = EVID / "semantic_candidate_manifest.json"
A_PATH = ALDIR / "outputs/aligned_transport_ready.pt"
V_PATH = ALDIR / "outputs/aligned_visibility_contract.pt"


# ============================================================
# Load frozen artifacts
# ============================================================

manifest = json.loads(
    MANIFEST.read_text(encoding="utf-8")
)

if manifest["status"] != (
    "CANDIDATES_FROZEN_VISUAL_ADJUDICATION_PENDING"
):
    raise RuntimeError(
        "candidate manifest is not frozen/pending"
    )

A = torch.load(
    str(A_PATH),
    map_location="cpu",
)

V = torch.load(
    str(V_PATH),
    map_location="cpu",
)

ids = np.load(
    V2 / "santa_material_point_ids.npy"
).astype(np.int64)

source_cells = np.load(
    V2 / "santa_material_source_vae_cell_ids.npy"
).astype(np.int64)

vis = np.load(
    V2 / "santa_material_visibility_correct.npy"
)[0].astype(bool)

render_xy = (
    A["points_2d_render"]
    .cpu()
    .numpy()
    .astype(np.float32)
)

depth = (
    A["depth"]
    .cpu()
    .numpy()
    .astype(np.float32)
)

raster = np.load(
    Path(
        V["source_files"]["raster_npy"]
    )
).astype(np.int64)

image_paths = [
    Path(p)
    for p in A["paths"]["coarse_rgb_frames"]
]

assert render_xy.shape == (81, 28264, 2)
assert vis.shape == (81, 1257)
assert raster.shape == (81, 512, 512)
assert len(image_paths) == 81
assert len(ids) == 1257


# ============================================================
# Font — robust against old Pillow / no TTF
# ============================================================

def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    for p in candidates:
        try:
            if Path(p).is_file():
                return ImageFont.truetype(
                    p,
                    size=size,
                )
        except Exception:
            pass

    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            size=size,
        )
    except Exception:
        return ImageFont.load_default()


FONT = load_font(18)
SMALL = load_font(14)
BIG = load_font(22)


# ============================================================
# Robust text helpers
# Never use multiline_textbbox()
# ============================================================

def line_size(draw, text, font):
    try:
        box = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        return (
            box[2] - box[0],
            box[3] - box[1],
        )

    except Exception:
        try:
            return draw.textsize(
                text,
                font=font,
            )
        except Exception:
            # Last-resort bitmap approximation.
            return (
                max(1, len(text)) * 8,
                14,
            )


def text_box(
    draw,
    xy,
    lines,
    font=SMALL,
):
    x, y = xy

    sizes = [
        line_size(
            draw,
            str(line),
            font,
        )
        for line in lines
    ]

    max_w = max(
        w
        for w, h in sizes
    )

    line_heights = [
        max(
            12,
            h,
        )
        for w, h in sizes
    ]

    spacing = 4
    pad = 6

    total_h = (
        sum(line_heights)
        + spacing * (
            len(lines) - 1
        )
    )

    draw.rectangle(
        (
            x - pad,
            y - pad,
            x + max_w + pad,
            y + total_h + pad,
        ),
        fill=(0, 0, 0),
    )

    yy = y

    for line, h in zip(
        lines,
        line_heights,
    ):
        draw.text(
            (x, yy),
            str(line),
            font=font,
            fill=(255, 255, 255),
        )

        yy += h + spacing


# ============================================================
# Visual marker
# ============================================================

def marker(
    draw,
    xy,
    visible,
    radius=10,
):
    x = float(xy[0])
    y = float(xy[1])

    if visible:
        draw.ellipse(
            (
                x-radius,
                y-radius,
                x+radius,
                y+radius,
            ),
            outline=(0, 255, 0),
            width=4,
        )

        draw.line(
            (
                x-radius-5,
                y,
                x+radius+5,
                y,
            ),
            fill=(0, 255, 0),
            width=2,
        )

        draw.line(
            (
                x,
                y-radius-5,
                x,
                y+radius+5,
            ),
            fill=(0, 255, 0),
            width=2,
        )

    else:
        draw.line(
            (
                x-radius,
                y-radius,
                x+radius,
                y+radius,
            ),
            fill=(255, 0, 0),
            width=4,
        )

        draw.line(
            (
                x-radius,
                y+radius,
                x+radius,
                y-radius,
            ),
            fill=(255, 0, 0),
            width=4,
        )


def clamp(v, lo, hi):
    return max(
        lo,
        min(
            hi,
            v,
        ),
    )


# ============================================================
# Machine observation
# ============================================================

def observation(
    bridge_idx,
    material_id,
    t,
):
    xy = render_xy[
        t,
        material_id,
    ]

    x = float(xy[0])
    y = float(xy[1])

    rx = int(round(x))
    ry = int(round(y))

    rx = clamp(rx, 0, 511)
    ry = clamp(ry, 0, 511)

    target_pixels = np.argwhere(
        raster[t] == material_id
    )

    pixel_count = int(
        len(target_pixels)
    )

    contract_visible = bool(
        vis[t, bridge_idx]
    )

    raster_contains_target = (
        pixel_count > 0
    )

    # For selected source-visible carriers,
    # these must agree exactly.
    if (
        contract_visible
        != raster_contains_target
    ):
        raise RuntimeError(
            "visibility/raster disagreement: "
            f"case idx={bridge_idx}, "
            f"id={material_id}, "
            f"frame={t}"
        )

    raster_id_here = int(
        raster[t, ry, rx]
    )

    target_depth = float(
        depth[t, material_id]
    )

    front_depth = None
    depth_delta = None

    if (
        raster_id_here >= 0
        and
        raster_id_here
        < depth.shape[1]
    ):
        front_depth = float(
            depth[
                t,
                raster_id_here,
            ]
        )

        depth_delta = float(
            target_depth
            - front_depth
        )

    bbox = None

    if pixel_count > 0:
        ys = target_pixels[:, 0]
        xs = target_pixels[:, 1]

        bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

    return {
        "bridge_index":
            int(bridge_idx),

        "material_id":
            int(material_id),

        "frame":
            int(t),

        "simulation_step":
            int(t * 10),

        "contract_visible":
            contract_visible,

        "render_xy":
            [x, y],

        "rounded_render_pixel_xy":
            [rx, ry],

        "target_raster_pixel_count":
            pixel_count,

        "target_raster_bbox_xyxy":
            bbox,

        "raster_id_at_rounded_projection":
            raster_id_here,

        "target_depth":
            target_depth,

        "front_id_depth_at_rounded_projection":
            front_depth,

        "target_minus_front_depth":
            depth_delta,
    }


# ============================================================
# One observation panel
# ============================================================

def render_panel(
    label,
    bridge_idx,
    material_id,
    t,
):
    base = Image.open(
        image_paths[t]
    ).convert("RGB")

    if base.size != (512, 512):
        raise RuntimeError(
            f"unexpected image size "
            f"{base.size}: {image_paths[t]}"
        )

    facts = observation(
        bridge_idx,
        material_id,
        t,
    )

    x, y = facts["render_xy"]

    visible = facts[
        "contract_visible"
    ]

    target_pixels = np.argwhere(
        raster[t] == material_id
    )

    # --------------------------------------------------------
    # Full image
    # --------------------------------------------------------

    full = base.copy()
    d = ImageDraw.Draw(full)

    # First cyan raster support.
    for py, px in target_pixels:
        d.rectangle(
            (
                int(px)-1,
                int(py)-1,
                int(px)+1,
                int(py)+1,
            ),
            fill=(0, 255, 255),
        )

    # Marker on top.
    marker(
        d,
        (x, y),
        visible,
        radius=11,
    )

    text_box(
        d,
        (10, 10),
        [
            label,
            f"material_id={material_id}",
            f"frame={t} step={t*10}",
            (
                "contract="
                + (
                    "VISIBLE"
                    if visible
                    else "INVISIBLE"
                )
            ),
            f"xy=({x:.2f},{y:.2f})",
            (
                "raster_support_pixels="
                f"{facts['target_raster_pixel_count']}"
            ),
        ],
    )

    # --------------------------------------------------------
    # Local zoom around projected point
    # --------------------------------------------------------

    radius = 48

    cx = int(round(x))
    cy = int(round(y))

    left = cx - radius
    top = cy - radius

    left = clamp(
        left,
        0,
        512 - 2*radius,
    )

    top = clamp(
        top,
        0,
        512 - 2*radius,
    )

    right = left + 2*radius
    bottom = top + 2*radius

    crop = base.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    zoom = crop.resize(
        (512, 512),
        RESAMPLE_NEAREST,
    )

    dz = ImageDraw.Draw(zoom)

    sx = 512.0 / (
        right - left
    )

    sy = 512.0 / (
        bottom - top
    )

    # Raster support first.
    for py, px in target_pixels:
        if (
            left <= px < right
            and
            top <= py < bottom
        ):
            zx = (
                px - left
            ) * sx

            zy = (
                py - top
            ) * sy

            dz.rectangle(
                (
                    zx-4,
                    zy-4,
                    zx+4,
                    zy+4,
                ),
                fill=(0, 255, 255),
            )

    zx = (
        x - left
    ) * sx

    zy = (
        y - top
    ) * sy

    marker(
        dz,
        (zx, zy),
        visible,
        radius=20,
    )

    delta = facts[
        "target_minus_front_depth"
    ]

    if delta is None:
        delta_text = "n/a"
    else:
        delta_text = (
            f"{delta:.6f}"
        )

    text_box(
        dz,
        (10, 10),
        [
            "LOCAL ZOOM",
            f"target={material_id}",
            (
                "front_id_at_proj="
                f"{facts['raster_id_at_rounded_projection']}"
            ),
            (
                "target-front depth="
                f"{delta_text}"
            ),
            "cyan=target frontmost raster support",
            (
                "green=visible"
                if visible
                else "red-X=invisible"
            ),
        ],
    )

    return (
        full,
        zoom,
        facts,
    )


# ============================================================
# Build frozen seven cases
# ============================================================

observations = []
sheet_paths = []

for case in manifest["cases"]:

    label = case["label"]

    bridge_idx = int(
        case["bridge_index"]
    )

    material_id = int(
        case["material_id"]
    )

    frames = [
        int(t)
        for t in case["audit_frames"]
    ]

    if len(frames) != 3:
        raise RuntimeError(
            f"{label}: expected 3 frames"
        )

    if int(
        ids[bridge_idx]
    ) != material_id:
        raise RuntimeError(
            f"{label}: manifest/index mismatch"
        )

    columns = []

    for t in frames:

        full, zoom, facts = render_panel(
            label,
            bridge_idx,
            material_id,
            t,
        )

        observations.append(
            {
                "case_label":
                    label,

                **facts,
            }
        )

        col = Image.new(
            "RGB",
            (512, 1100),
            (245, 245, 245),
        )

        dc = ImageDraw.Draw(col)

        dc.text(
            (10, 10),
            f"frame {t} / step {t*10}",
            font=BIG,
            fill=(0, 0, 0),
        )

        dc.text(
            (10, 40),
            (
                "contract visibility = "
                f"{bool(vis[t,bridge_idx])}"
            ),
            font=FONT,
            fill=(0, 0, 0),
        )

        col.paste(
            full,
            (0, 70),
        )

        col.paste(
            zoom,
            (0, 588),
        )

        columns.append(col)

    sheet = Image.new(
        "RGB",
        (1536, 1165),
        (255, 255, 255),
    )

    ds = ImageDraw.Draw(sheet)

    ds.text(
        (15, 8),
        (
            f"{label} | "
            f"bridge_idx={bridge_idx} | "
            f"material_id={material_id} | "
            f"source_cell={int(source_cells[bridge_idx])}"
        ),
        font=BIG,
        fill=(0, 0, 0),
    )

    ds.text(
        (15, 36),
        (
            "Frozen candidate; visual semantic "
            "adjudication pending"
        ),
        font=FONT,
        fill=(80, 80, 80),
    )

    for i, col in enumerate(columns):
        sheet.paste(
            col,
            (i*512, 62),
        )

    out = (
        OUT
        / f"{label}__audit_sheet.png"
    )

    sheet.save(out)

    sheet_paths.append(out)


# ============================================================
# Machine observations
# ============================================================

facts_path = (
    OUT
    / "semantic_machine_observations.json"
)

facts_path.write_text(
    json.dumps(
        {
            "artifact_kind":
                "Phase0A-4.5 semantic visibility machine observations",

            "status":
                "VISUAL_ADJUDICATION_PENDING",

            "bridge":
                str(V2),

            "candidate_manifest":
                str(MANIFEST),

            "coordinate_system":
                "realwonder_512_uv_xy",

            "observation_count":
                len(observations),

            "important_scope":
                (
                    "Simulation/render-space visibility semantics only; "
                    "not real-world RGB visibility ground truth."
                ),

            "observations":
                observations,
        },
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# Human adjudication template
# ============================================================

human = {
    "audit":
        "Phase0A-4.5 Human Semantic Visibility Adjudication",

    "status":
        "PENDING",

    "scope":
        (
            "Simulator/render-space visual semantics only."
        ),

    "case_adjudications":
        [],
}

for case in manifest["cases"]:
    human["case_adjudications"].append(
        {
            "label":
                case["label"],

            "bridge_index":
                case["bridge_index"],

            "material_id":
                case["material_id"],

            "audit_frames":
                case["audit_frames"],

            "observed_visual_semantics":
                None,

            "contract_matches_visual_semantics":
                None,

            "switching_interpretation":
                None,

            "decision":
                "UNRESOLVED",

            "notes":
                None,
        }
    )

human_path = (
    OUT
    / "semantic_human_adjudication_template.json"
)

human_path.write_text(
    json.dumps(
        human,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# Contact sheet
# ============================================================

thumb_w = 620
thumb_h = 470

overview = Image.new(
    "RGB",
    (
        1280,
        4 * 520,
    ),
    (255, 255, 255),
)

do = ImageDraw.Draw(overview)

for i, path in enumerate(
    sheet_paths
):
    image = Image.open(
        path
    ).convert("RGB")

    image.thumbnail(
        (
            thumb_w,
            thumb_h,
        )
    )

    row = i // 2
    col = i % 2

    x = col * 640
    y = row * 520

    do.text(
        (
            x + 8,
            y + 8,
        ),
        path.stem,
        font=FONT,
        fill=(0, 0, 0),
    )

    overview.paste(
        image,
        (
            x + 8,
            y + 38,
        ),
    )

overview_path = (
    OUT
    / "semantic_visibility_contact_sheet.png"
)

overview.save(
    overview_path
)


# ============================================================
# Frozen-source manifest copy
# ============================================================

manifest_copy = (
    OUT
    / "semantic_candidate_manifest_FROZEN_COPY.json"
)

manifest_copy.write_bytes(
    MANIFEST.read_bytes()
)


# ============================================================
# SHA256
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024*1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


targets = (
    sheet_paths
    + [
        facts_path,
        human_path,
        overview_path,
        manifest_copy,
        OUT / "render_semantic_overlay.py",
    ]
)

sha_path = (
    OUT
    / "SHA256SUMS.txt"
)

sha_path.write_text(
    "\n".join(
        f"{sha256(p)}  {p.name}"
        for p in targets
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== PHASE0A-4.5 OVERLAY BUILD PASS ====="
)

print(
    "OUT =",
    OUT
)

print(
    "CASE_SHEETS =",
    len(sheet_paths)
)

print(
    "OBSERVATIONS =",
    len(observations)
)

print(
    "CONTACT_SHEET =",
    overview_path
)

print(
    "MACHINE_FACTS =",
    facts_path
)

print(
    "HUMAN_TEMPLATE =",
    human_path
)

print(
    "SHA256 =",
    sha_path
)

print(
    "STATUS = VISUAL_ADJUDICATION_PENDING"
)
