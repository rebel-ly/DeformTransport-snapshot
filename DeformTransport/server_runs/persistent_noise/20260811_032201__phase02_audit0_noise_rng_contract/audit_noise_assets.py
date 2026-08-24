import json
from pathlib import Path
import numpy as np

ROOT = Path("/workspace/DeformTransport")
OUT = Path.cwd()

rows = []

for p in ROOT.glob("server_runs/**/noises.npy"):
    try:
        a = np.load(p, mmap_mode="r")

        row = {
            "path": str(p),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "size_mb": round(p.stat().st_size / 1024**2, 3),
        }

        # deterministic lightweight statistics
        flat = a.reshape(-1)
        n = len(flat)

        if n:
            step = max(1, n // 200000)
            x = np.asarray(flat[::step][:200000], dtype=np.float32)

            row.update({
                "sample_n": int(len(x)),
                "sample_mean": float(x.mean()),
                "sample_var": float(x.var()),
                "sample_min": float(x.min()),
                "sample_max": float(x.max()),
            })

        rows.append(row)

    except Exception as e:
        rows.append({
            "path": str(p),
            "error": repr(e),
        })

rows.sort(key=lambda x: x["path"])

report = {
    "n_noise_assets": len(rows),
    "assets": rows,
}

(OUT / "noise_asset_audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print("===== NOISE ASSETS =====")
print("count =", len(rows))

for r in rows:
    if "error" in r:
        print("ERROR", r["path"], r["error"])
        continue

    print()
    print(r["path"])
    print(
        " shape=", r["shape"],
        "dtype=", r["dtype"],
        "MB=", r["size_mb"],
        "mean=", round(r.get("sample_mean", 0), 6),
        "var=", round(r.get("sample_var", 0), 6),
    )

print()
print("SAVED:", OUT / "noise_asset_audit.json")
