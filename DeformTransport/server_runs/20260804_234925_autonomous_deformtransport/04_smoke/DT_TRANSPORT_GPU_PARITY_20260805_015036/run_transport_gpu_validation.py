"""在真实CUDA上重算Santa材料点transport并与已保存artifact核对。"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
REPO_ROOT=Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from deform_transport.hard_transport import hard_point_transport
from deform_transport.transport_ready import validate_transport_ready

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--transport-ready",type=Path,required=True)
    p.add_argument("--latent-artifact",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=0)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    state=torch.load(a.transport_ready,map_location="cpu",weights_only=False)
    artifact=torch.load(a.latent_artifact,map_location="cpu",weights_only=False)
    validate_transport_ready(state)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA不可用")
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    source=artifact["source_latent"][0,0].cuda()
    frame_indices=artifact["latent_frame_indices"].long()
    def run(mode):
        torch.cuda.synchronize(); begin=time.perf_counter()
        result=hard_point_transport(
            source,
            state["source_points_2d_latent"].cuda(),
            state["points_2d_latent"][frame_indices].cuda(),
            state["source_visible"].cuda(),
            state["source_valid"].cuda(),
            state["projection_valid"][frame_indices].cuda(),
            state["point_id"].cuda(),
            object_id=state["object_id"].cuda(), mode=mode, seed=a.seed)
        torch.cuda.synchronize(); return result,time.perf_counter()-begin
    correct,correct_s=run("correct"); shuffled,shuffled_s=run("shuffled")
    checks={
      "Correct_mask一致":bool(torch.equal(correct["transport_mask"].cpu(),artifact["transport_mask"])),
      "Correct_count一致":bool(torch.equal(correct["contribution_count"].cpu(),artifact["contribution_count"])),
      "Shuffled_mask一致":bool(torch.equal(shuffled["transport_mask"].cpu(),artifact["transport_mask"])),
      "Correct有限":bool(torch.isfinite(correct["transported_grid"]).all()),
      "Shuffled有限":bool(torch.isfinite(shuffled["transported_grid"]).all()),
      "Correct与Shuffled不同":bool(not torch.equal(correct["transported_grid"],shuffled["transported_grid"]))}
    correct_ref=artifact["correct_transported_latent"][0]
    shuffled_ref=artifact["shuffled_transported_latent"][0]
    report={
      "任务":"真实CUDA材料点transport重算与artifact一致性验证","状态":"通过" if all(checks.values()) else "失败",
      "GPU":{"名称":torch.cuda.get_device_name(0),"CUDA_VISIBLE_DEVICES":__import__("os").environ.get("CUDA_VISIBLE_DEVICES")},
      "输入":{"transport_ready":str(a.transport_ready.resolve()),"transport_ready_SHA256":sha256(a.transport_ready),"latent_artifact":str(a.latent_artifact.resolve()),"latent_artifact_SHA256":sha256(a.latent_artifact),"seed":a.seed},
      "规模":{"点数":int(state["point_id"].numel()),"latent帧":int(frame_indices.numel()),"latent_shape":list(correct["transported_grid"].shape)},
      "检查":checks,
      "与历史artifact最大绝对差":{"Correct":float((correct["transported_grid"].cpu()-correct_ref).abs().max()),"Shuffled":float((shuffled["transported_grid"].cpu()-shuffled_ref).abs().max())},
      "运行秒数":{"Correct":correct_s,"Shuffled":shuffled_s,"总计":time.perf_counter()-started},
      "显存MiB":{"峰值已分配":float(torch.cuda.max_memory_allocated()/1024**2),"峰值已保留":float(torch.cuda.max_memory_reserved()/1024**2)}}
    (a.output_dir/"结果.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if report["状态"]!="通过": raise SystemExit(2)
if __name__=="__main__": main()
