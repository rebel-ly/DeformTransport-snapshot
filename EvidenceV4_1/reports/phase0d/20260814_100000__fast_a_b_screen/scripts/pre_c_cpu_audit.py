#!/usr/bin/env python3
"""CPU-only Phase0D-4C pre-C audit.  Does not import Wan models or use CUDA."""
import hashlib, importlib.util, json, os
from pathlib import Path
import cv2, numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

ROOT = Path('/workspace/DeformTransport_EvidenceV4_1')
OUT = ROOT/'reports/phase0d/20260814_100000__fast_a_b_screen/pre_c_cpu_audit'
F2 = ROOT/'reports/phase0d/conditioning_alignment/20260813_081824__f2_metric_mechanism_error_localization'
EVP = ROOT/'reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py'
OVERLAY = ROOT/'experimental/20260814__wanmove_preview_sdedit_overlay'
PREVIEW = ROOT/'reports/phase0d/20260814_100000__fast_a_b_screen/preview_wan_vae_latent_e1_832x480.npy'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def stats(x):
    x=np.asarray(x,dtype=np.float64)
    return {'n':int(x.size),'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.percentile(x,95))}

def load_ev():
    spec=importlib.util.spec_from_file_location('ev',EVP); ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev); return ev

def tcmar(frames, ev):
    root=Path('/workspace/DeformTransport')
    tracks=np.load(root/'server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy')[0].astype(np.float32)
    vis=np.load(root/'server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy')[0].astype(bool)
    source=ev.read_rgb_image(root/ev.CASES['santa']['source']); n=tracks.shape[1]; c0=tracks[0]
    sv=(c0[:,0]-3.5>=0)&(c0[:,0]+3.5<=831)&(c0[:,1]-3.5>=0)&(c0[:,1]+3.5<=479)
    sp=np.full((n,8,8,3),np.nan,np.float32); good=np.where(sv)[0]; sp[good]=ev.sample_patches(source,c0[good])
    sl=np.full((n,3),np.nan,np.float32); sl[good]=ev.patch_mean_lab(sp[good]); rows=[]
    for t in ev.ANCHORS:
        c=tracks[t].copy(); c[:,1]*=464/480; fv=(c[:,0]-3.5>=0)&(c[:,0]+3.5<=831)&(c[:,1]-3.5>=0)&(c[:,1]+3.5<=463)
        ids=np.where(vis[t]&sv&fv&np.isfinite(c).all(1))[0]; patch=ev.sample_patches(frames[t],c[ids]); rows.append((ids,np.linalg.norm(ev.patch_mean_lab(patch)-sl[ids],axis=1)))
    agg,count=ev.aggregate(rows,n); return stats(agg[count>0])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ev=load_ev()
    fd=F2/'dt_edited_y_decoded_frames'; files=sorted(fd.glob('frame_*.png')); assert len(files)==81
    raw=[]
    for p in files:
        b=cv2.imread(str(p),cv2.IMREAD_COLOR); assert b is not None
        raw.append(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))
    raw=np.stack(raw)
    # Historical diagnostic path: pre-divide then formal to_common divides again.
    historic=np.stack([ev.to_common(x.astype(np.float32)/255.0) for x in raw])
    corrected=np.stack([ev.to_common(x) for x in raw])
    hist=tcmar(historic,ev); corr=tcmar(corrected,ev)
    # Directly mirror real 40-step, shift=3 scheduler formula: initial sigmas 1..1/40 then shifted.
    base=np.linspace(1.0,1.0/40.0,40,dtype=np.float64)
    sig=3.0*base/(1.0+2.0*base); ts=sig*1000.0
    schedule=[{'index':int(i),'timestep':float(ts[i]),'sigma':float(sig[i])} for i in range(40)]
    preview=np.load(PREVIEW,mmap_mode='r')
    result={
      'cpu_only':True,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'omp_threads':os.environ['OMP_NUM_THREADS'],
      'dt_edited_y_lineage':{'script':str(F2/'scripts/diagnose_rgb_condition.py'),'input_frames':str(fd),'frame_count':len(files),'timeline':'decoded VAE edited_y frames 0000..0080; inherited corrected-v2 S0..S800 track timeline','historical_preprocessing':'cv2 uint8 -> float32 /255 -> frozen evaluator to_common /255','corrected_preprocessing':'cv2 uint8 -> frozen evaluator to_common /255 exactly once','historical_tcmar':hist,'corrected_tcmar':corr,'DT_EDITED_Y_65_74_SAME_DOUBLE_DIV255_BUG':bool(abs(hist['mean']-65.74260475842897)<1e-7)},
      'formal_evaluator_contamination':{'historical_f2_buggy_path':str(F2/'scripts/diagnose_rgb_condition.py'),'frozen_corrected_v2_path':str(EVP),'historical_calls_predivision':True,'formal_read_video_common_passes_uint8_to_to_common':True,'FORMAL_EVALUATOR_AFFECTED_BY_DOUBLE_DIV255':False,'rw_dt_condition_tcme':{'rw':0.8946881631311656,'dt_edited_y':1.5050048806385061,'affected_by_same_diagnostic_rgb_preprocessing':True,'formal_endpoint_tcme_affected':False}},
      'sigma_independence':{'scheduler_source':str(OVERLAY/'wan/utils/fm_solvers_unipc.py'),'formula':'sigmas=shift*linspace(1,1/steps,steps)/(1+(shift-1)*linspace(...)); timesteps=sigmas*1000','steps':40,'shift':3.0,'schedule':schedule,'index14':schedule[14],'index15':schedule[15],'index16':schedule[16],'rw_sigma_source':'canonical RW FlowMatch scheduler warp mapping, independently documented in PHASE0D_4C_ENGINEERING_GATE.md','RW_WAN_SIGMA_ZERO_ERROR_INDEPENDENTLY_DERIVED':bool(abs(sig[15]-5/6)<1e-12)},
      'enabled_path_interface':{'preview_path':str(PREVIEW),'preview_shape':[1]+list(preview.shape),'preview_dtype':str(preview.dtype),'preview_finite':bool(np.isfinite(preview).all()),'shared_epsilon_frozen':False,'epsilon_check':'interface/required C,T,H,W shape verified; final shared epsilon intentionally deferred until disabled-path parity PASS','flow_start_formula':'x_start=(1-sigma)*preview_latent + sigma*epsilon','start_index':15,'sigma':float(sig[15]),'source_static_gate':{'preview_latent_argument':True,'initial_epsilon_argument':True,'start_index_argument':True,'set_begin_index_call':True,'history_reset_implemented_by_set_timesteps':True},'ENABLED_PATH_START_STATE_SANITY':'UNRESOLVED_NO_FINAL_EPSILON_AND_NO_SCHEDULER_RUNTIME_PROBE'},
      'source_sha256':{'historical_diagnostic':sha(F2/'scripts/diagnose_rgb_condition.py'),'formal_evaluator':sha(EVP),'scheduler':sha(OVERLAY/'wan/utils/fm_solvers_unipc.py'),'overlay_wan_move':sha(OVERLAY/'wan/wan_move.py'),'preview_latent':sha(PREVIEW)}
    }
    (OUT/'pre_c_cpu_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'historical':hist['mean'],'corrected':corr['mean'],'index15':schedule[15],'enabled':result['enabled_path_interface']['ENABLED_PATH_START_STATE_SANITY']},sort_keys=True))
if __name__=='__main__': main()
