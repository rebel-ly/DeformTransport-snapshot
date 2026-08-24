import hashlib
import json
from pathlib import Path

ROOT=Path("/workspace/DeformTransport")

assets={
    "santa":
        ROOT/"server_runs/20260804_234925_autonomous_deformtransport/"
        "prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/noises.npy",

    "tree":
        ROOT/"server_runs/20260804_234925_autonomous_deformtransport/"
        "prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/noises.npy",
}

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while True:
            x=f.read(1024*1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()

out={
    "name":"Persistent-Noise Phase02 A0 Contract",
    "raw_noise_contract":{
        "shape":[81,60,104,32],
        "dtype":"float16",
        "spatial_order":"T,H,W,C raw numpy",
    },
    "realwonder_runtime":{
        "target_latent_frames":21,
        "channel_dim":16,
        "downsample_mode":"nearest",
        "eval_degradation":0.5,
        "num_frame_per_block":3,
        "pixel_frames":81,
    },
    "assets":{}
}

for k,p in assets.items():
    out["assets"][k]={
        "path":str(p),
        "sha256":sha256(p),
        "bytes":p.stat().st_size,
    }

Path("a0_contract.json").write_text(
    json.dumps(out,indent=2)+"\n"
)

print(json.dumps(out,indent=2))
print("A0_CONTRACT_FROZEN")
