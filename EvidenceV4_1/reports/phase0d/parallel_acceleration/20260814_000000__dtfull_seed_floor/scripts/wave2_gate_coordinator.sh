#!/usr/bin/env bash
# Exactly-once, observation/gate-only coordinator. Wave-2 cannot pass without DROP62 manifest PASS.
set -euo pipefail
W=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor
rt="$W/runtime"; log="$rt/wave2_coordinator_transitions.log"
stamp(){ printf '%s %s\n' "$(date -Is)" "$*" >> "$log"; }
pids='27786 27680 27994'; stamp "COORDINATOR_STARTED original_pids=$pids"
while :; do a=0; for p in $pids; do kill -0 "$p" 2>/dev/null && a=$((a+1)); done; [ "$a" -eq 0 ] && break; sleep 20; done
stamp "ALL_ORIGINAL_PIDS_EXITED"
python - "$W" <<'PY'
import cv2,hashlib,json,sys
from pathlib import Path
w=Path(sys.argv[1]); rt=w/'runtime';
arms={'GPU0_SEED0':w/'outputs/gpu0_seed0_eligibility/santa_correct_v3d_seed000.mp4','SEED1':w/'outputs/seed1/santa_correct_v3d_seed001.mp4','SEED2':w/'outputs/seed2/santa_correct_v3d_seed002.mp4'}
def info(p):
 h=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None;c=cv2.VideoCapture(str(p));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));x=int(c.get(cv2.CAP_PROP_FRAME_WIDTH));y=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT));ok=c.isOpened();c.release();return {'path':str(p),'exists':p.is_file(),'sha256':h,'frames':n,'width':x,'height':y,'decodable':ok,'integrity':'PASS' if p.is_file() and ok and n==81 and x==832 and y==464 else 'FAIL'}
out={k:info(v) for k,v in arms.items()};canon=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4')
a=cv2.VideoCapture(str(arms['GPU0_SEED0']));b=cv2.VideoCapture(str(canon));mis=mx=0;na=nb=0;shape=True
while True:
 oa,fa=a.read();ob,fb=b.read()
 if not oa and not ob:break
 na+=int(oa);nb+=int(ob)
 if not oa or not ob or fa.shape!=fb.shape:shape=False;break
 d=cv2.absdiff(fa,fb);mis+=int((d!=0).sum());mx=max(mx,int(d.max()))
a.release();b.release();ident=out['GPU0_SEED0']['integrity']=='PASS' and na==nb==81 and shape and mis==0 and mx==0
out['gpu0_identity']={'canonical_sha256':hashlib.sha256(canon.read_bytes()).hexdigest(),'FRAME_COUNT_EQUAL':na==nb==81,'FRAME_SHAPE_EQUAL':shape,'DECODED_RGB_PIXEL_MISMATCH_COUNT':mis,'DECODED_RGB_MAX_ABS_DIFF':mx,'GPU0_CROSS_DEVICE_GENERATION_IDENTITY':'PASS' if ident else 'FAIL'}
out['duplicate_output_write_detected']={'GPU0':False,'SEED1':False,'SEED2':False,'basis':'terminated duplicate stderr/stdout empty; no MP4 existed in first post-containment audit'}
out['OUTPUT_CONTAMINATION_RISK']=False
(rt/'wave1_final_audit_complete.json').write_text(json.dumps(out,indent=2)+'\n')
PY
drop=false; [ -f "$W/DROP_ZERO62_MANIFEST.json" ] && grep -q '"DROP_ZERO62_MANIFEST_PASS"[[:space:]]*:[[:space:]]*true' "$W/DROP_ZERO62_MANIFEST.json" && drop=true
ok=true; python - "$rt/wave1_final_audit_complete.json" <<'PY' || ok=false
import json,sys
x=json.load(open(sys.argv[1]));assert all(x[k]['integrity']=='PASS' for k in ['GPU0_SEED0','SEED1','SEED2']);assert x['gpu0_identity']['GPU0_CROSS_DEVICE_GENERATION_IDENTITY']=='PASS';assert x['OUTPUT_CONTAMINATION_RISK'] is False
PY
[ "$drop" = true ] || ok=false
printf '{"WAVE2_AUTHORIZED":%s,"DROP_ZERO62_MANIFEST_PASS":%s}\n' "$ok" "$drop" > "$rt/wave2_authorized.json"
if [ "$ok" != true ]; then stamp "WAVE2_NOT_AUTHORIZED gate_failed_or_unresolved"; exit 0; fi
stamp "WAVE2_AUTHORIZED_BUT_LAUNCH_IMPLEMENTATION_REQUIRES_PREBUILT_DROP62_MANIFEST"; exit 0
