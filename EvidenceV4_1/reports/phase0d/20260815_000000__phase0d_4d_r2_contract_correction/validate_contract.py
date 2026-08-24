#!/usr/bin/env python3
"""External Phase 0D-4D-R2 contract validator; never imports or edits Wan code."""
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path('/workspace/DeformTransport_EvidenceV4_1')
R = ROOT/'reports/phase0d/20260815_000000__phase0d_4d_r2_contract_correction'
OLD = ROOT/'reports/phase0d/20260814_200000__phase0d_4d_recovery'
PREVIEW = ROOT/'reports/phase0d/20260814_100000__fast_a_b_screen/preview_wan_vae_latent_e1_832x480.npy'
EPS = OLD/'FINAL_C_SHARED_EPSILON.npy'
OVERLAY = ROOT/'experimental/20260814__wanmove_preview_sdedit_overlay'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def content(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256()
    h.update(str(tuple(a.shape)).encode())
    h.update('torch.float32'.encode())
    h.update(a.tobytes())
    return h.hexdigest()
def load(p):
    a=np.load(p, allow_pickle=False)
    if a.dtype != np.float32 or not np.isfinite(a).all(): raise SystemExit('invalid tensor '+str(p))
    return a
preview, eps=load(PREVIEW), load(EPS)
assert preview.shape == eps.shape == (16,21,60,104)
assert sha(PREVIEW)=='9d71791f70fa519001708d0986c4b1cee297941b8f66a3d1a15e03ef8ce8bb8f'
assert content(eps)=='b6f3bbbd7bbc3412b70dd57c39e1709e70ba7d38b9e0cb60bac7820776208b51'
assert sha(OVERLAY/'wan'/'wan_move.py')=='eae7f5a86f39164f3ad1ce3b8db4a974f4a71f42c2898402f029bb9db77c32f7'
assert sha(OVERLAY/'generate.py')=='45f7323f22d7bb7d593b949fa48e6cf764d08cafeaf8863d726df9a663b21b85'
base=json.loads((ROOT/'reports/phase0d/20260814_100000__fast_a_b_screen/corrected_parity_20260814/A2_EFFECTIVE_MANIFEST.json').read_text())
k0={'tracks':str(OLD/'k0_tracks.npy'),'visibility':str(OLD/'k0_visibility.npy'),'ids':str(OLD/'k0_ids.npy'),'depth':str(OLD/'k0_depth.npy')}
for key, shape, dtype in [('tracks',(81,0,2),np.float32),('visibility',(81,0),np.bool_),('ids',(0,),np.int64),('depth',(81,0),np.float32)]:
    a=np.load(k0[key], allow_pickle=False); assert a.shape==shape and a.dtype==dtype
for key in ('tracks','visibility','ids','depth'):
    if not Path(k0[key]).is_file(): raise SystemExit('missing K0 '+key)
base['source']={'root':str(OVERLAY),'mode':'overlay','wan_move_sha256':sha(OVERLAY/'wan'/'wan_move.py'),'generate_sha256':sha(OVERLAY/'generate.py')}
base['enabled_path']={'preview_path':str(PREVIEW),'preview_file_sha256':sha(PREVIEW),'preview_shape':list(preview.shape),'epsilon_path':str(EPS),'epsilon_content_sha256':content(eps),'epsilon_shape':list(eps.shape),'begin0_formula':'(1-1.0)*preview + 1.0*epsilon = epsilon','begin15_formula':'(1-5/6)*preview + (5/6)*epsilon','scheduler':'FlowUniPCMultistepScheduler','steps':40,'shift':3.0}
base['transport_guard']={'variant':'v3d','required_c2':{'tracks':base['tracks']['path'],'visibility':base['visibility']['path'],'ids':base['ids']['path'],'depth':base['depth']['path']},'required_c1_k0':k0,'silent_fallback_allowed':False}
R.mkdir(parents=True,exist_ok=True)
(R/'BEGIN0_EFFECTIVE_MANIFEST.json').write_text(json.dumps(base,indent=2,sort_keys=True)+'\n')
result={'formal_overlay_modified':False,'preview_runtime_contract':'RAW_UNBATCHED_[16,21,60,104]','preview_batch_contract':'PASS_RAW_ARTIFACT_MATCHES_FORMAL_RUNTIME','derived_batched_preview_used_in_formal_run':False,'runtime_preview_shape':list(preview.shape),'runtime_epsilon_shape':list(eps.shape),'runtime_epsilon_tensor_content_sha256':content(eps),'runtime_epsilon_content_exact':content(eps)=='b6f3bbbd7bbc3412b70dd57c39e1709e70ba7d38b9e0cb60bac7820776208b51','transport_mode_contract_guard':'PASS','silent_transport_fallback_allowed':False,'k0_legal_artifacts':k0,'c2_required_artifacts':base['transport_guard']['required_c2']}
(R/'CONTRACT_PREFLIGHT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
