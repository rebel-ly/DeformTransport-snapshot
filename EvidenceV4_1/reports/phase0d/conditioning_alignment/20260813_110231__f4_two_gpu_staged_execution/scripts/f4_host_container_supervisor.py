#!/usr/bin/env python3
"""Host scheduler launches generation in existing verified container; completion only."""
import subprocess,time
from pathlib import Path
B=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution'); F='/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze'; W='/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution'; rt=B/'runtime'
def now(p):p.write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z')+'\n')
def start(n,g,s):
 o=f'{W}/outputs/{n}_container_seed0'; now(rt/(n+'_container2_start_time.txt')); cmd=['docker','exec','-e',f'CUDA_VISIBLE_DEVICES={g}','deformtransport-dev','bash',f'{W}/scripts/run_subset_container_v3d.sh','0',o,f'{F}/artifacts/{s}_tracks.npy',f'{F}/artifacts/{s}_visibility.npy',f'{F}/artifacts/{s}_ids.npy',f'{F}/artifacts/{s}_depth.npy']; so=open(rt/(n+'_container2_stdout.log'),'w');se=open(rt/(n+'_container2_stderr.log'),'w');p=subprocess.Popen(cmd,stdout=so,stderr=se,start_new_session=True);(rt/(n+'_container2_pid.txt')).write_text(str(p.pid)+'\n');return p,so,se
def end(n,p,so,se):c=p.wait();so.close();se.close();now(rt/(n+'_container2_end_time.txt'));(rt/(n+'_container2_exit_code.txt')).write_text(str(c)+'\n');return c
def main():
 w,ws,we=start('wm0',1,'wm0');f,fs,fe=start('frag',2,'frag_prune')
 while w.poll() is None and f.poll() is None:time.sleep(5)
 if w.poll() is not None: arm,gpu,n,p,so,se='WM-0','GPU1','wm0',w,ws,we
 else: arm,gpu,n,p,so,se='DT-FRAG-PRUNE','GPU2','frag',f,fs,fe
 end(n,p,so,se);(rt/'container2_first_finished_arm.txt').write_text(arm+'\n');(rt/'container2_first_free_gpu.txt').write_text(gpu+'\n');g,gs,ge=start('grid100',1 if gpu=='GPU1' else 2,'grid100_center')
 if n=='wm0':end('frag',f,fs,fe)
 else:end('wm0',w,ws,we)
 end('grid100',g,gs,ge)
if __name__=='__main__':main()
