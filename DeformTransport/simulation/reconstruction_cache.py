from pathlib import Path
import json

import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras


def load_reconstruction_cache(config):
    cache = Path(str(config["reconstruction_cache"])).resolve()
    assert cache.exists(), f"cache does not exist: {cache}"

    device = torch.device(str(config["device"]))

    meta = json.loads(
        (cache / "metadata.json").read_text()
    )

    # foreground point clouds
    fg_pcs = []
    for path in sorted(cache.glob("fg_pc_*.npz")):
        data = np.load(path)

        fg_pcs.append({
            "points": torch.from_numpy(
                data["points"]
            ).to(device).float(),

            "colors": torch.from_numpy(
                data["colors"]
            ).to(device).float(),
        })

    # foreground meshes
    fg_meshes = []
    for path in sorted(cache.glob("fg_mesh_*.npz")):
        data = np.load(path)

        fg_meshes.append({
            "vertices": torch.from_numpy(
                data["vertices"]
            ).to(device).float(),

            "faces": torch.from_numpy(
                data["faces"]
            ).to(device).long(),

            "colors": torch.from_numpy(
                data["colors"]
            ).to(device).float(),
        })

    assert fg_pcs, "no foreground point cloud cache"
    assert fg_meshes, "no foreground mesh cache"
    assert len(fg_pcs) == len(fg_meshes)

    bg_points = torch.from_numpy(
        np.load(cache / "bg_points.npy")
    ).to(device).float()

    bg_colors = torch.from_numpy(
        np.load(cache / "bg_points_colors.npy")
    ).to(device).float()

    ground_path = cache / "ground_plane_normal.npy"

    ground_plane_normal = (
        np.load(ground_path)
        if ground_path.exists()
        else None
    )

    fx = float(meta["fx_pixels"])
    fov = float(meta["fov_x_input"])

    config["fov_x_input"] = fov

    H, W = meta.get("target_size", [512, 512])

    # Exactly the same camera convention used by
    # SingleViewReconstructor.get_camera_at_origin()
    K = torch.zeros((1, 4, 4), device=device)

    K[0, 0, 0] = fx
    K[0, 1, 1] = fx
    K[0, 0, 2] = W / 2
    K[0, 1, 2] = H / 2
    K[0, 3, 2] = 1
    K[0, 2, 3] = 1

    R = torch.eye(
        3,
        device=device
    ).unsqueeze(0)

    T = torch.zeros(
        (1, 3),
        device=device
    )

    camera = PerspectiveCameras(
        K=K,
        R=R,
        T=T,
        in_ndc=False,
        image_size=((H, W),),
        device=device,
    )

    # Official RealWonder cached renderer
    from demo_web.simulation_engine import _MinimalSVR

    svr = _MinimalSVR(
        config=config,
        camera=camera,
        focal_length=fx,
        bg_points=bg_points,
        bg_points_colors=bg_colors,
        fg_pcs=[
            {
                "points": pc["points"].clone(),
                "colors": pc["colors"].clone(),
            }
            for pc in fg_pcs
        ],
        device=device,
    )

    masks_path = cache / "object_masks.npy"
    if masks_path.exists():
        masks = np.load(masks_path)

        svr.object_masks = [
            torch.from_numpy(mask)
            .to(device)
            .bool()
            for mask in masks
        ]

    print(
        f"[cache] loaded {len(fg_pcs)} object(s)"
    )
    print(
        f"[cache] fg points = "
        f"{[tuple(x['points'].shape) for x in fg_pcs]}"
    )
    print(
        f"[cache] meshes = "
        f"{[tuple(x['vertices'].shape) for x in fg_meshes]}"
    )
    print(
        f"[cache] fx={fx:.6f}, fov={fov:.6f}"
    )

    return (
        svr,
        fg_pcs,
        fg_meshes,
        ground_plane_normal,
        config,
    )
