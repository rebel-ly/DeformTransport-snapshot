#!/usr/bin/env bash
set -u

RUN="$1"

DT=/workspace/DeformTransport
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python


# ------------------------------------------------------------
# Shared geometry assets
# ------------------------------------------------------------

{
echo "===== SHARED GEOMETRY / REGION ASSETS ====="

find "$DT/server_runs" \
  -type f \
  \( \
    -iname '*visibility*.npy' \
    -o -iname '*visibility*.pt' \
    -o -iname '*mask*.npy' \
    -o -iname '*mask*.pt' \
    -o -iname '*mask*.h5' \
    -o -iname '*depth*.npy' \
    -o -iname '*depth*.pt' \
    -o -iname '*depth*.h5' \
    -o -name 'point_trajectories.pt' \
    -o -iname '*trajector*.npy' \
    -o -iname '*trajector*.pt' \
  \) \
  2>/dev/null \
  | grep -Ei 'santa|tree' \
  | sort

} > "$RUN/region_asset_audit.txt" 2>&1


# ------------------------------------------------------------
# RealWonder structured-noise implementation
# ------------------------------------------------------------

{
echo "===== REALWONDER NOISE CODE REFERENCES ====="

grep -RniE \
'noises\.npy|load_noise|structured_noise|structured_noise_sde|eval_degradation|randperm|extract_subdim|grid_sample|warp.*noise|noise.*warp|flow' \
"$DT/infer_sim.py" \
"$DT/vidgen" \
"$DT/simulation" \
"$DT/wan" \
2>/dev/null \
--include='*.py' \
| head -1200

echo
echo "===== WHO WRITES noises.npy ====="

grep -RniE \
'noises\.npy|np\.save.*noise|save.*noise' \
"$DT" \
2>/dev/null \
--include='*.py' \
--exclude-dir=server_runs \
--exclude-dir=.git \
| head -600

echo
echo "===== HISTORICAL REALWONDER RUN CONTRACT ====="

grep -RniE \
'infer_sim\.py|sim_data_path|checkpoint_path|eval_degradation|base_seed|seed|noises\.npy|num_frame_per_block' \
"$DT/server_runs/20260804_234925_autonomous_deformtransport" \
2>/dev/null \
| head -1600

echo
echo "===== infer_sim relevant block ====="

grep -nE \
'load_noise|eval_degradation|num_frame_per_block|num_output_frames|pixel_num_frames|sim_data_path|structured_noise' \
"$DT/infer_sim.py" \
2>/dev/null

echo
echo "===== vidgen noise relevant blocks ====="

grep -RniE \
'def load_noise|def extract_subdim|randperm|structured_noise|structured_noise_sde|add_noise|eval_degradation' \
"$DT/vidgen" \
2>/dev/null \
--include='*.py'

} > "$RUN/pnoise_code_audit.txt" 2>&1


# ------------------------------------------------------------
# Noise assets: exact shapes + hashes + finite stats
# ------------------------------------------------------------

"$PY" - "$RUN" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


run = Path(sys.argv[1])

root = Path(
    "/workspace/DeformTransport/server_runs"
)

rows = []

for p in root.rglob("noises.npy"):
    s = str(p).lower()

    if (
        "santa" not in s
        and
        "tree" not in s
    ):
        continue

    rec = {
        "path":
            str(p),
    }

    try:
        a = np.load(
            p,
            mmap_mode="r",
            allow_pickle=False,
        )

        rec["shape"] = list(
            a.shape
        )

        rec["dtype"] = str(
            a.dtype
        )

        rec["bytes"] = int(
            p.stat().st_size
        )

        h = hashlib.sha256()

        with p.open("rb") as f:
            for block in iter(
                lambda: f.read(
                    1 << 20
                ),
                b"",
            ):
                h.update(block)

        rec["sha256"] = (
            h.hexdigest()
        )

        # Sample only if reasonably shaped numeric array.
        if (
            np.issubdtype(
                a.dtype,
                np.number,
            )
            and a.size > 0
        ):
            flat = np.asarray(
                a.reshape(-1)[
                    ::max(
                        1,
                        a.size // 200000
                    )
                ],
                dtype=np.float64,
            )

            rec["sample_mean"] = float(
                flat.mean()
            )

            rec["sample_var"] = float(
                flat.var()
            )

            rec["finite_fraction"] = float(
                np.isfinite(
                    flat
                ).mean()
            )

    except Exception as e:
        rec["error"] = repr(e)

    rows.append(rec)


report = {
    "count":
        len(rows),

    "assets":
        rows,
}


(
    run / "pnoise_assets.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    ) + "\n"
)


print(
    "PNOISE_ASSET_COUNT",
    len(rows),
)

for x in rows:
    print(
        x["path"],
        x.get("shape"),
        x.get("dtype"),
        x.get("sample_mean"),
        x.get("sample_var"),
    )
PY


date -Iseconds \
> "$RUN/AUDIT_CONTRACTS_DONE.txt"
