#!/usr/bin/env python3
"""Frozen deterministic companion diagnostics; no external model dependency."""
import argparse, cv2, numpy as np

def common(rgb):
    # Frozen evaluation-only 480->464 mapping; never modifies generation inputs.
    return cv2.resize(rgb,(832,464),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255.0
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--preview',required=True); p.add_argument('--track-mask',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    vo=cv2.VideoCapture(a.output); vp=cv2.VideoCapture(a.preview); rows=[]; mask=np.load(a.track_mask).astype(bool)
    while True:
        ao,bo=vo.read(); ap,bp=vp.read()
        if not ao and not ap: break
        if not(ao and ap): raise RuntimeError('frame-count mismatch')
        o=common(cv2.cvtColor(bo,cv2.COLOR_BGR2RGB)); q=common(cv2.cvtColor(bp,cv2.COLOR_BGR2RGB)); d=np.abs(o-q); reg=mask[:464,:832]
        for name,sel in [('FULL_FRAME',np.ones(reg.shape,bool)),('TRACK_OBJECT_SUPPORTED_REGION',reg)]:
            z=d[sel]; mse=np.mean((o-q)[sel]**2); rows.append((name,float(z.mean()),float(10*np.log10(1.0/max(mse,1e-12)))))
        # SHARPNESS_DIAGNOSTIC: mean squared forward spatial RGB gradient energy.
        grad=(np.diff(o,axis=0)**2).mean()+(np.diff(o,axis=1)**2).mean()
        rows.append(('SHARPNESS_DIAGNOSTIC_MEAN_SPATIAL_GRADIENT_ENERGY',float(grad),float('nan')))
    vo.release(); vp.release(); np.savez(a.out,rows=np.asarray(rows,dtype=object))
if __name__=='__main__': main()
