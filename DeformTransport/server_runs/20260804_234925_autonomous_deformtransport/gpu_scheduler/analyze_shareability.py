from __future__ import annotations
import csv,json,statistics
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
root=Path(__file__).resolve().parent
samples=[]
with (root/"gpu_samples.csv").open(encoding="utf-8") as f:
 for r in csv.DictReader(f):
  r={k:(v.strip() if isinstance(v,str) else v) for k,v in r.items()}; r["dt"]=datetime.strptime(r["时间"],"%Y-%m-%dT%H:%M:%S%z")
  for k in ["采样序号","GPU编号","利用率百分比","显存已用MiB","显存空闲MiB","温度C","ECC不可纠正错误","系统可用内存KiB"]: r[k]=int(float(r[k]))
  r["功耗W"]=float(r["功耗W"]); samples.append(r)
latest=max(r["dt"] for r in samples); start=latest-timedelta(seconds=60); window=[r for r in samples if r["dt"]>=start]
proc=defaultdict(int); pids=defaultdict(set)
with (root/"gpu_processes.csv").open(encoding="utf-8") as f:
 for r in csv.DictReader(f):
  sid=int(r["采样序号"]); u=r["GPU_UUID"].strip(); proc[(sid,u)]+=int(r["显存MiB"]); pids[u].add(int(r["PID"]))
def max_streak(values,predicate):
 best=cur=0
 for v in values:
  cur=cur+1 if predicate(v) else 0; best=max(best,cur)
 return best
out={}
for g in range(4):
 rows=sorted([r for r in window if r["GPU编号"]==g],key=lambda r:r["dt"]); u=rows[0]["UUID"]; util=[r["利用率百分比"] for r in rows]; mem=[proc[(r["采样序号"],u)] for r in rows]
 span=(rows[-1]["dt"]-rows[0]["dt"]).total_seconds(); streak=max_streak(util,lambda x:x>70); p95=sorted(util)[max(0,int(0.95*len(util))-1)]
 light={"窗口不少于60秒":span>=55 and len(rows)>=11,"平均利用率不超过20%":statistics.mean(util)<=20,"瞬时峰值不超过60%":max(util)<=60,"无连续30秒利用率超过70%":streak<6,"当前空闲显存不少于20GiB":rows[-1]["显存空闲MiB"]>=20*1024,"对方显存60秒增长不超过4GiB":mem[-1]-mem[0]<=4096,"温度不超过82C":max(r["温度C"] for r in rows)<=82,"ECC无异常":max(r["ECC不可纠正错误"] for r in rows)==0,"系统可用内存不少于30GiB":min(r["系统可用内存KiB"] for r in rows)>=30*1024*1024}
 vae={"平均利用率不超过15%":statistics.mean(util)<=15,"95%采样值不超过60%":p95<=60,"无连续30秒利用率超过70%":streak<6,"当前空闲显存大于18.19GiB":rows[-1]["显存空闲MiB"]>18.19*1024,"对方显存无持续增长":mem[-1]-mem[0]<=1024,"温度不超过80C":max(r["温度C"] for r in rows)<=80}
 out[str(g)]={"状态":"SHAREABLE" if all(light.values()) else "BUSY","轻量任务状态":"SHAREABLE" if all(light.values()) else "BUSY","WanVAE状态":"SHAREABLE" if all(vae.values()) and span>=55 and len(rows)>=11 else "BUSY","UUID":u,"采样数":len(rows),"窗口秒数":span,"平均利用率":statistics.mean(util),"最高利用率":max(util),"95%采样利用率":p95,"连续超过70%最长采样数":streak,"当前空闲显存MiB":rows[-1]["显存空闲MiB"],"其他进程显存起始MiB":mem[0],"其他进程显存当前MiB":mem[-1],"其他进程显存增长MiB":mem[-1]-mem[0],"其他进程显存范围MiB":max(mem)-min(mem),"最高温度C":max(r["温度C"] for r in rows),"ECC最大":max(r["ECC不可纠正错误"] for r in rows),"系统最低可用内存GiB":min(r["系统可用内存KiB"] for r in rows)/1024/1024,"其他用户PID":sorted(pids[u]),"轻量检查":light,"WanVAE检查":vae}
report={"窗口结束":latest.isoformat(),"窗口开始":start.isoformat(),"阈值版本":"用户2026-08-05修正规则_轻量峰值60%", "GPU":out}
(root/"shareability_latest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
