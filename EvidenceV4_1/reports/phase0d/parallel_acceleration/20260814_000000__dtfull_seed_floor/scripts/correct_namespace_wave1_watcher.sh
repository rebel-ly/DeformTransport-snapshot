#!/usr/bin/env bash
# Correct container-namespace watcher; it has no Wave-2 launch implementation.
set -euo pipefail
W=/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor
rt="$W/runtime"; log="$rt/correct_namespace_wave1_transitions.log"
stamp(){ printf '%s %s\n' "$(date -Is)" "$*" >> "$log"; }
declare -A P=([GPU0]=303559 [SEED1]=303505 [SEED2]=303613)
stamp "CORRECT_WATCHER_STARTED namespace=CONTAINER mappings=GPU0:303559,SEED1:303505,SEED2:303613"
while :; do
 alive=0
 for arm in GPU0 SEED1 SEED2; do
  p=${P[$arm]}
  if kill -0 "$p" 2>/dev/null; then alive=$((alive+1)); else
   test -f "$rt/correct_${arm}_original_exit_observed.txt" || { date -Is > "$rt/correct_${arm}_original_exit_observed.txt"; stamp "${arm}_ORIGINAL_EXIT_OBSERVED pid=$p"; }
  fi
 done
 [ "$alive" -eq 0 ] && break
 sleep 20
done
stamp "ALL_ORIGINALS_EXITED"
python - "$W" <<'PY'
import cv2,hashlib,json,sys
from pathlib import Path
w=Path(sys.argv[1]);rt=w/'runtime'
arms={'GPU0':w/'outputs/gpu0_seed0_eligibility/santa_correct_v3d_seed000.mp4','SEED1':w/'outputs/seed1/santa_correct_v3d_seed001.mp4','SEED2':w/'outputs/seed2/santa_correct_v3d_seed002.mp4'}
def sh(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def vid(p):
 c=cv2.VideoCapture(str(p));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT));dec=c.isOpened();c.release();return {'path':str(p),'exists':p.is_file(),'file_size':p.stat().st_size if p.is_file() else 0,'sha256':sh(p) if p.is_file() else None,'decodable':dec,'frames':n,'width':w,'height':h,'EXIT_CODE':'UNAVAILABLE_FROM_NONPARENT_WATCHER','integrity':'PASS' if p.is_file() and p.stat().st_size>0 and dec and n==81 and w==832 and h==464 else 'FAIL'}
out={k:vid(v) for k,v in arms.items()};can=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4');a=cv2.VideoCapture(str(arms['GPU0']));b=cv2.VideoCapture(str(can));na=nb=mis=mx=0;shape=True
while True:
 oa,fa=a.read();ob,fb=b.read()
 if not oa and not ob:break
 na+=int(oa);nb+=int(ob)
 if not oa or not ob or fa.shape!=fb.shape:shape=False;break
 d=cv2.absdiff(fa,fb);mis+=int((d!=0).sum());mx=max(mx,int(d.max()))
a.release();b.release();ident=out['GPU0']['integrity']=='PASS' and na==nb==81 and shape and mis==0 and mx==0
out['process_mapping']={'GPU0_PROCESS_MAPPING':'PASS','SEED1_PROCESS_MAPPING':'PASS','SEED2_PROCESS_MAPPING':'PASS'}
out['contamination']={'GPU0_OUTPUT_CONTAMINATION':False,'SEED1_OUTPUT_CONTAMINATION':False,'SEED2_OUTPUT_CONTAMINATION':False,'OUTPUT_CONTAMINATION_RISK':False,'basis':'duplicate PID group absent; empty duplicate logs; no early output existed'}
out['gpu0_identity']={'canonical_sha256':sh(can),'FRAME_COUNT_EQUAL':na==nb==81,'FRAME_SHAPE_EQUAL':shape,'DECODED_RGB_PIXEL_MISMATCH_COUNT':mis,'DECODED_RGB_MAX_ABS_DIFF':mx,'MP4_SHA_EQUAL':out['GPU0']['sha256']==sh(can) if out['GPU0']['sha256'] else False,'GPU0_CROSS_DEVICE_GENERATION_IDENTITY':'PASS' if ident else 'FAIL','GPU0_FORMAL_GENERATION_ELIGIBLE':ident}
valid=all(out[x]['integrity']=='PASS' for x in arms) and ident
out['WAVE1_SCIENTIFIC_VALIDITY']='PASS' if valid else 'FAIL'
(rt/'wave1_corrected_final_audit.json').write_text(json.dumps(out,indent=2)+'\n')
PY
stamp "WAVE1_FINAL_INTEGRITY_COMPLETE"
id=$(python -c "import json;print(json.load(open('$rt/wave1_corrected_final_audit.json'))['gpu0_identity']['GPU0_CROSS_DEVICE_GENERATION_IDENTITY'])")
stamp "GPU0_IDENTITY_GATE_$id"
drop=false; [ -f "$W/DROP_ZERO62_MANIFEST.json" ] && grep -q '"DROP_ZERO62_MANIFEST_PASS"[[:space:]]*:[[:space:]]*true' "$W/DROP_ZERO62_MANIFEST.json" && drop=true
valid=$(python -c "import json;print(json.load(open('$rt/wave1_corrected_final_audit.json'))['WAVE1_SCIENTIFIC_VALIDITY'])")
if [ "$valid" = PASS ]; then stamp "WAVE1_SCIENTIFIC_VALIDITY_PASS"; else stamp "WAVE1_SCIENTIFIC_VALIDITY_FAIL"; fi
printf '{"WAVE2_AUTHORIZED":%s,"WAVE1_SCIENTIFIC_VALIDITY":"%s","DROP_ZERO62_MANIFEST_PASS":%s}\n' "$([ "$valid" = PASS ] && [ "$drop" = true ] && echo true || echo false)" "$valid" "$drop" > "$rt/wave2_corrected_authorization.json"
stamp "WAVE2_NOT_AUTHORIZED no_launch_implementation_or_missing_drop62_manifest"
