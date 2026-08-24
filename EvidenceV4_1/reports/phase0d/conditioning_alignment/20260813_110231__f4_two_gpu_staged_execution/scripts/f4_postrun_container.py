#!/usr/bin/env python3
"""Container-path entrypoint for the preregistered F4 post-run workflow."""
import runpy
from pathlib import Path
p=Path(__file__).with_name('f4_postrun.py')
s=p.read_text()
s=s.replace("B=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1", "B=Path('/workspace/DeformTransport_EvidenceV4_1")
s=s.replace("ROOT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport')", "ROOT=Path('/workspace/DeformTransport')")
s=s.replace("F3=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1", "F3=Path('/workspace/DeformTransport_EvidenceV4_1")
s=s.replace("F2=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1", "F2=Path('/workspace/DeformTransport_EvidenceV4_1")
s=s.replace("EVAL=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1", "EVAL=Path('/workspace/DeformTransport_EvidenceV4_1")
s=s.replace("DT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1", "DT=Path('/workspace/DeformTransport_EvidenceV4_1")
s=s.replace("torch.device('cuda:0')", "torch.device('cuda:0')")
s=s.replace(".cuda()", ".to(torch.device('cuda:0'))")
ns={'__name__':'__main__','__file__':str(p)}
exec(compile(s, str(p), 'exec'), ns)
