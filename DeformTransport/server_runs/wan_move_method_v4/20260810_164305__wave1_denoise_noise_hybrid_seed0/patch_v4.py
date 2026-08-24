from pathlib import Path

WAN = Path("/workspace/Wan-Move")

tp = WAN / "wan/modules/trajectory.py"
wp = WAN / "wan/wan_move.py"

t = tp.read_text()
w = wp.read_text()

MARK_TRAJ = "# ===== DeformTransport V4 material transport helper ====="
MARK_WAN  = "# ===== DeformTransport V4 protocol ====="

if MARK_TRAJ in t or MARK_WAN in w:
    raise RuntimeError("V4 patch marker already present; refusing double patch")

# ----------------------------------------------------------
# Append deterministic V4 transport helper.
# It reuses the V3D _DT_CONTEXT established by
# create_pos_feature_map.
# ----------------------------------------------------------

helper = r'''

# ===== DeformTransport V4 material transport helper =====
_DT_V4_PLAN_CACHE = {}


def _dt_v4_plan(track_pos, patch_radius):
    """
    Build deterministic material -> latent target claims once.

    Arbitration:
      smallest positive camera depth,
      then smallest persistent material ID.

    patch_radius=0: center cell only.
    patch_radius=1: 3x3 local support with matched source offsets.
    """
    global _DT_CONTEXT, _DT_V4_PLAN_CACHE

    if _DT_CONTEXT is None:
        raise RuntimeError(
            "DT V4 requires V3D context from create_pos_feature_map"
        )

    if track_pos.ndim != 4 or track_pos.shape[0] != 1:
        raise RuntimeError(
            f"unexpected V4 track_pos shape {tuple(track_pos.shape)}"
        )

    key = (int(patch_radius), tuple(track_pos.shape))

    if key in _DT_V4_PLAN_CACHE:
        return _DT_V4_PLAN_CACHE[key]

    tp = track_pos[0].detach().cpu().numpy()

    tracks = (
        _DT_CONTEXT["tracks"]
        .detach()
        .cpu()
        .numpy()
    )

    vis = (
        _DT_CONTEXT["visibility"]
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )

    depth = (
        _DT_CONTEXT["depth"]
        .detach()
        .cpu()
        .numpy()
    )

    ids = (
        _DT_CONTEXT["ids"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    td = int(_DT_CONTEXT["t_down"])
    sy = int(_DT_CONTEXT["h_down"])
    sx = int(_DT_CONTEXT["w_down"])

    image_h = int(_DT_CONTEXT["height"])
    image_w = int(_DT_CONTEXT["width"])

    tracks_s = tracks[::td]
    vis_s = vis[::td]
    depth_s = depth[::td]

    Tlat = tp.shape[1]

    if tracks_s.shape[0] != Tlat:
        raise RuntimeError(
            f"V4 temporal mismatch tracks={tracks_s.shape[0]} "
            f"track_pos={Tlat}"
        )

    src_valid = (
        (tp[:, 0, 0] >= 0)
        &
        (tp[:, 0, 1] >= 0)
    )

    plan = []

    for tau in range(Tlat):
        if tau == 0:
            plan.append(None)
            continue

        claims = {}

        for ii in range(tp.shape[0]):
            if not src_valid[ii]:
                continue

            if not vis_s[tau, ii]:
                continue

            th = int(tp[ii, tau, 0])
            tw = int(tp[ii, tau, 1])

            if th < 0 or tw < 0:
                continue

            z = float(depth_s[tau, ii])

            if not np.isfinite(z) or z <= 0:
                continue

            mid = int(ids[ii])

            src_x0 = float(tracks_s[0, ii, 0])
            src_y0 = float(tracks_s[0, ii, 1])

            for dy in range(-patch_radius, patch_radius + 1):
                for dx in range(-patch_radius, patch_radius + 1):

                    hh = th + dy
                    ww = tw + dx

                    src_x = src_x0 + dx * sx
                    src_y = src_y0 + dy * sy

                    # Actual latent bounds are checked again at application.
                    # Here preserve source image-domain validity.
                    if not (
                        0.0 <= src_x < image_w
                        and
                        0.0 <= src_y < image_h
                    ):
                        continue

                    k = (hh, ww)

                    claim = (
                        z,
                        mid,
                        src_x,
                        src_y,
                    )

                    old = claims.get(k)

                    if old is None or claim[:2] < old[:2]:
                        claims[k] = claim

        if not claims:
            plan.append(None)
            continue

        # Stable target-cell order.
        keys = sorted(claims.keys())

        target_h = np.asarray(
            [x[0] for x in keys],
            dtype=np.int64,
        )

        target_w = np.asarray(
            [x[1] for x in keys],
            dtype=np.int64,
        )

        src_xy = np.asarray(
            [
                [
                    claims[k][2],
                    claims[k][3],
                ]
                for k in keys
            ],
            dtype=np.float32,
        )

        plan.append(
            {
                "target_h": target_h,
                "target_w": target_w,
                "src_xy": src_xy,
            }
        )

    out = {
        "plan": plan,
        "stride_y": sy,
        "stride_x": sx,
    }

    _DT_V4_PLAN_CACHE[key] = out
    return out


def _dt_v4_sample_source(frame0, source_xy, stride_y, stride_x):
    """
    frame0: [C,H,W], float32
    source_xy: [M,2] in source pixel coordinates.

    Same cell-center convention as frozen V3D:
        latent_x = pixel_x / stride_x - 0.5
        latent_y = pixel_y / stride_y - 0.5
    """
    if source_xy.numel() == 0:
        return torch.empty(
            (0, frame0.shape[0]),
            device=frame0.device,
            dtype=frame0.dtype,
        )

    C, H, W = frame0.shape

    lx = source_xy[:, 0] / float(stride_x) - 0.5
    ly = source_xy[:, 1] / float(stride_y) - 0.5

    if W > 1:
        gx = lx * (2.0 / (W - 1)) - 1.0
    else:
        gx = torch.zeros_like(lx)

    if H > 1:
        gy = ly * (2.0 / (H - 1)) - 1.0
    else:
        gy = torch.zeros_like(ly)

    grid = torch.stack(
        [gx, gy],
        dim=-1,
    ).view(1, 1, -1, 2)

    sampled = torch.nn.functional.grid_sample(
        frame0.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # [1,C,1,M] -> [M,C]
    return (
        sampled[0, :, 0, :]
        .transpose(0, 1)
        .contiguous()
    )


def dt_v4_transport(
    feature,
    track_pos,
    patch_radius=0,
    mix=1.0,
):
    """
    Transport source-time latent features along persistent
    material trajectories using frozen V3D depth arbitration.

    feature:
      [C,T,H,W] or [1,C,T,H,W]

    mix:
      1.0 = hard replacement at supported cells.
      0.2 = fixed soft interpolation used by V4B noise.
    """
    if feature.ndim == 4:
        was_unbatched = True
        x = feature.unsqueeze(0)
    elif feature.ndim == 5 and feature.shape[0] == 1:
        was_unbatched = False
        x = feature
    else:
        raise RuntimeError(
            f"V4 feature shape unsupported: {tuple(feature.shape)}"
        )

    if track_pos.ndim == 3:
        track_pos = track_pos.unsqueeze(0)

    if not (0.0 <= float(mix) <= 1.0):
        raise RuntimeError(f"invalid V4 mix={mix}")

    protocol = _dt_v4_plan(
        track_pos,
        int(patch_radius),
    )

    plan = protocol["plan"]
    sy = protocol["stride_y"]
    sx = protocol["stride_x"]

    original_dtype = x.dtype

    # CPU bf16 grid_sample is not universally supported.
    # Execute intervention mathematically in float32, then cast back.
    work = x.float().clone()

    _, _, T, H, W = work.shape

    if len(plan) != T:
        raise RuntimeError(
            f"V4 plan T={len(plan)} feature T={T}"
        )

    source_frame = work[0, :, 0]

    for tau in range(1, T):
        p = plan[tau]

        if p is None:
            continue

        hh_np = p["target_h"]
        ww_np = p["target_w"]

        inside = (
            (hh_np >= 0)
            &
            (hh_np < H)
            &
            (ww_np >= 0)
            &
            (ww_np < W)
        )

        if not np.any(inside):
            continue

        hh = torch.from_numpy(
            hh_np[inside]
        ).to(
            device=work.device,
            dtype=torch.long,
        )

        ww = torch.from_numpy(
            ww_np[inside]
        ).to(
            device=work.device,
            dtype=torch.long,
        )

        source_xy = torch.from_numpy(
            p["src_xy"][inside]
        ).to(
            device=work.device,
            dtype=torch.float32,
        )

        src = _dt_v4_sample_source(
            source_frame,
            source_xy,
            sy,
            sx,
        )

        # [C,M]
        src = src.transpose(0, 1)

        old = work[0, :, tau, hh, ww]

        if float(mix) == 1.0:
            new = src
        else:
            new = (
                old * (1.0 - float(mix))
                +
                src * float(mix)
            )

        work[0, :, tau, hh, ww] = new

    work = work.to(dtype=original_dtype)

    return work[0] if was_unbatched else work
'''

t2 = t + helper

# ----------------------------------------------------------
# Wan-Move import.
# ----------------------------------------------------------

old_import = (
    "from .modules.trajectory import replace_feature, "
    "get_video_track_video, create_pos_feature_map"
)

new_import = (
    "from .modules.trajectory import replace_feature, "
    "get_video_track_video, create_pos_feature_map, "
    "dt_v4_transport"
)

if w.count(old_import) != 1:
    raise RuntimeError(
        f"unexpected trajectory import count={w.count(old_import)}"
    )

w2 = w.replace(
    old_import,
    new_import,
    1,
)

# ----------------------------------------------------------
# V4 variant protocol is fixed immediately after V3D y edit.
# ----------------------------------------------------------

anchor = (
    "        edited_y = replace_feature("
    "y.unsqueeze(0), track_pos.unsqueeze(0))[0]\n"
)

insert = r'''        edited_y = replace_feature(y.unsqueeze(0), track_pos.unsqueeze(0))[0]

        # ===== DeformTransport V4 protocol =====
        dt_v4_variant = (
            __import__("os").environ
            .get("DT_V4_VARIANT", "none")
            .strip()
            .lower()
        )

        dt_v4_allowed = {
            "none",
            "v4a_d20",
            "v4a_d40",
            "v4b_noise",
            "v4c_hybrid",
        }

        if dt_v4_variant not in dt_v4_allowed:
            raise RuntimeError(
                f"unknown DT_V4_VARIANT={dt_v4_variant}"
            )

        print(
            "DT_V4_PROTOCOL",
            "variant=", dt_v4_variant,
            "steps=", 40,
            "noise_mix=", 0.2,
            flush=True,
        )
'''

if w2.count(anchor) != 1:
    raise RuntimeError(
        f"edited_y anchor count={w2.count(anchor)}"
    )

w2 = w2.replace(
    anchor,
    insert,
    1,
)

# ----------------------------------------------------------
# V4B / V4C initial-noise transport.
# ----------------------------------------------------------

noise_anchor = '''            # sample videos
            latent = noise
'''

noise_insert = '''            # sample videos

            # DeformTransport V4B/V4C:
            # material-consistent initial noise.
            if dt_v4_variant in {"v4b_noise", "v4c_hybrid"}:
                noise = dt_v4_transport(
                    noise,
                    track_pos.unsqueeze(0),
                    patch_radius=1,
                    mix=0.2,
                )

                print(
                    "DT_V4_NOISE_TRANSPORT_DONE",
                    "variant=",
                    dt_v4_variant,
                    flush=True,
                )

            latent = noise
'''

if w2.count(noise_anchor) != 1:
    raise RuntimeError(
        f"noise anchor count={w2.count(noise_anchor)}"
    )

w2 = w2.replace(
    noise_anchor,
    noise_insert,
    1,
)

# ----------------------------------------------------------
# V4A clean-x0 intervention between CFG and scheduler.step().
# ----------------------------------------------------------

cfg_anchor = '''                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
'''

cfg_insert = '''                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                # DeformTransport V4A/V4C:
                # x0 = z_t - sigma_t * v_t
                # x0' = material_transport(x0)
                # v_t' = (z_t - x0') / sigma_t
                if dt_v4_variant in {
                    "v4a_d20",
                    "v4a_d40",
                    "v4c_hybrid",
                }:
                    if dt_v4_variant == "v4a_d40":
                        dt_v4_limit = 16
                    else:
                        dt_v4_limit = 8

                    if _ < dt_v4_limit:
                        dt_sigma = sample_scheduler.sigmas[_]

                        dt_sigma = dt_sigma.to(
                            device=noise_pred.device,
                            dtype=torch.float32,
                        )

                        if float(dt_sigma.item()) <= 1e-8:
                            raise RuntimeError(
                                "V4 encountered non-positive sigma"
                            )

                        dt_latent = latent.to(
                            device=noise_pred.device,
                            dtype=torch.float32,
                        )

                        dt_velocity = noise_pred.to(
                            dtype=torch.float32
                        )

                        dt_x0 = (
                            dt_latent
                            -
                            dt_sigma * dt_velocity
                        )

                        dt_x0_material = dt_v4_transport(
                            dt_x0,
                            track_pos.unsqueeze(0),
                            patch_radius=0,
                            mix=1.0,
                        )

                        noise_pred = (
                            (
                                dt_latent
                                -
                                dt_x0_material.float()
                            )
                            /
                            dt_sigma
                        ).to(
                            dtype=noise_pred.dtype
                        )

                        print(
                            "DT_V4_X0_INTERVENTION",
                            "variant=",
                            dt_v4_variant,
                            "step=",
                            _,
                            "sigma=",
                            float(dt_sigma.item()),
                            flush=True,
                        )
'''

if w2.count(cfg_anchor) != 1:
    raise RuntimeError(
        f"CFG anchor count={w2.count(cfg_anchor)}"
    )

w2 = w2.replace(
    cfg_anchor,
    cfg_insert,
    1,
)

# Syntax check BEFORE touching live source.
compile(t2, str(tp), "exec")
compile(w2, str(wp), "exec")

tp.write_text(t2)
wp.write_text(w2)

print("V4_SOURCE_PATCH_OK")
