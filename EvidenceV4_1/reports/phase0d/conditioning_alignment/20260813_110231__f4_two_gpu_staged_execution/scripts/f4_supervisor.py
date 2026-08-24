#!/usr/bin/env python3
"""Compatibility supervisor: schedules solely on process completion."""
import os, subprocess, time
from pathlib import Path
B=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution'); F=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze'); R=B/'scripts/run_subset_frozen_v3d.sh'; rt=B/'runtime'; out=B/'outputs'
def now(p): p.write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z')+'\n')
def start(name,gpu,stem):
 d=out/(name+'_seed0'); d.mkdir(parents=True,exist_ok=True); now(rt/(name+'_start_time.txt'))
 cmd=['bash',str(R),'0',str(d),str(F/'artifacts'/f'{stem}_tracks.npy'),str(F/'artifacts'/f'{stem}_visibility.npy'),str(F/'artifacts'/f'{stem}_ids.npy'),str(F/'artifacts'/f'{stem}_depth.npy')]
 so=open(rt/(name+'_stdout.log'),'w'); se=open(rt/(name+'_stderr.log'),'w'); e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=str(gpu)
 p=subprocess.Popen(cmd,stdout=so,stderr=se,env=e,start_new_session=True); (rt/(name+'_pid.txt')).write_text(str(p.pid)+'\n'); return p,so,se
def done(name,p,so,se):
 c=p.wait(); so.close(); se.close(); now(rt/(name+'_end_time.txt')); (rt/(name+'_exit_code.txt')).write_text(str(c)+'\n'); return c
def main():
 wm,wso,wse=start('wm0',1,'wm0'); fr,fso,fse=start('frag',2,'frag_prune'); first=None
 while first is None:
  if wm.poll() is not None: first=('WM-0','GPU1','wm0',wm,wso,wse)
  elif fr.poll() is not None: first=('DT-FRAG-PRUNE','GPU2','frag',fr,fso,fse)
  else: time.sleep(5)
 arm,gpu,name,p,so,se=first; done(name,p,so,se); (rt/'first_finished_arm.txt').write_text(arm+'\n'); (rt/'first_free_gpu.txt').write_text(gpu+'\n')
 gr,gso,gse=start('grid100',1 if gpu=='GPU1' else 2,'grid100_center');
 if name=='wm0': done('frag',fr,fso,fse)
 else: done('wm0',wm,wso,wse)
 done('grid100',gr,gso,gse)
if __name__=='__main__': main()
