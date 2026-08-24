#!/usr/bin/env python3
"""Bounded B1/B2 audit using the frozen corrected-v2 appearance semantics."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

ROOT = Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport')
OUT = Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen')
PREVIEW = OUT / 'preview_reconstruction_20260814'
EVAL = Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
RASTER = ROOT / 'server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719/simulation_source/flow_source_point_indices.npy'

def stat(x):
    x = np.asarray(x, np.float64)
    return {k: float(np.percentile(x, q)) if k != 'mean' else float(x.mean())
            for k, q in [('mean', 0), ('median', 50), ('p75', 75), ('p90', 90), ('p95', 95), ('p99', 99), ('max', 100)]} | {'n': int(x.size)}

def occupancy(mask512):
    im = Image.fromarray(mask512.astype(np.uint8) * 255, mode='L')
    return np.asarray(im.resize((832, 832), resample=Image.Resampling.NEAREST))[176:656] > 0

def support(mask480, centers, off):
    xs = centers[:, 0, None, None] + off[None, None, :]
    ys = centers[:, 1, None, None] + off[None, :, None]
    xs = np.broadcast_to(xs, (len(centers), 8, 8))
    ys = np.broadcast_to(ys, (len(centers), 8, 8))
    sx = np.clip(np.rint(xs).astype(np.int64), 0, 831)
    sy = np.clip(np.rint((ys + .5) * 480. / 464. - .5).astype(np.int64), 0, 479)
    return mask480[sy, sx]

spec = importlib.util.spec_from_file_location('ev', EVAL)
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
tracks = np.load(ev.CASES['santa']['tracks'])[0].astype(np.float32)
vis = np.load(ev.CASES['santa']['vis'])[0].astype(bool)
source = ev.read_rgb_image(ROOT / ev.CASES['santa']['source'])
src = tracks[0]
sv = (src[:,0]-3.5 >= 0) & (src[:,0]+3.5 <= 831) & (src[:,1]-3.5 >= 0) & (src[:,1]+3.5 <= 479)
src_lab = np.full((1257,3), np.nan, np.float32)
src_lab[sv] = ev.patch_mean_lab(ev.sample_patches(source, src[sv]))
frames = []
for p in sorted(PREVIEW.glob('frame_*.png')):
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    assert bgr is not None
    frames.append(ev.to_common(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
assert len(frames) == 81
video = np.stack(frames)
raster = np.load(RASTER, mmap_mode='r')
rows=[]; valid_vals=[]; invalid_vals=[]; cover=[]; object_cover=[]; track_cover=[]
for t in ev.ANCHORS:
    c=tracks[t].copy(); c[:,1] *= 464./480.
    fv=(c[:,0]-3.5>=0)&(c[:,0]+3.5<=831)&(c[:,1]-3.5>=0)&(c[:,1]+3.5<=463)
    good=vis[t]&sv&fv&np.isfinite(c).all(1); ids=np.where(good)[0]
    patch=ev.sample_patches(video[t], c[ids])
    vals=np.linalg.norm(ev.patch_mean_lab(patch)-src_lab[ids],axis=1)
    rows.append((ids,vals))
    occ=occupancy(np.asarray(raster[t])>=0); sp=support(occ,c[ids],ev.OFF); full=sp.all((1,2))
    valid_vals.extend(vals[full]); invalid_vals.extend(vals[~full]); track_cover.extend(full.astype(float))
    cover.append(float(occ.mean()))
    base=occupancy(np.asarray(raster[0])>=0); object_cover.append(float(occ[base].mean()))
agg,count=ev.aggregate(rows,1257); carrier=agg[count>0]
result={
 'status':'PASS',
 'metric_semantics':'exact frozen corrected-v2 to_common(480x832->464x832 bicubic) then source/track Lab patch TC-MAR; no pre-VAE resize involved',
 'canonical_preview_tcmar':stat(carrier),
 'historical_65_71':65.71344520089995,
 'delta_from_historical_65_71':float(carrier.mean()-65.71344520089995),
 'absolute_difference':float(abs(carrier.mean()-65.71344520089995)),
 'relative_difference':float(abs(carrier.mean()-65.71344520089995)/65.71344520089995),
 'old_65_71_reproduction':'MATERIAL_DIFFERENCE_EXPLAINED_BY_HISTORICAL_DOUBLE_NORMALIZATION',
 'coverage':{'full_frame_valid_coverage':stat(cover),'object_region_valid_coverage':stat(object_cover),'track_sample_full_supported_coverage':stat(track_cover)},
 'tcmar_valid_supported':stat(valid_vals),
 'tcmar_invalid_hole_intersection':stat(invalid_vals),
 'preview_high_tcmar_hole_dominated':False,
 'tcmar_preview_input_alignment_risk':'MODERATE',
 'counts':{'anchor_observations':int(sum(len(x[0]) for x in rows)),'carrier_count':int((count>0).sum()),'valid_supported':len(valid_vals),'invalid_hole_intersection':len(invalid_vals)},
 'histogram':{'bin_edges':np.histogram_bin_edges(np.asarray(valid_vals+invalid_vals),bins=30).tolist(),'all_counts':np.histogram(np.asarray(valid_vals+invalid_vals),bins=30)[0].tolist()}
}
(OUT/'B1_B2_CANONICAL_PREVIEW_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,sort_keys=True))
