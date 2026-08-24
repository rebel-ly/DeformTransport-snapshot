import json,torch
from wan.modules.attention import flash_attention

torch.manual_seed(20260805)
q=torch.randn(1,5,4,32,device='cuda',dtype=torch.bfloat16)
k=torch.randn(1,7,4,32,device='cuda',dtype=torch.bfloat16)
v=torch.randn(1,7,4,32,device='cuda',dtype=torch.bfloat16)
got_plain=flash_attention(q,k,v,version=2)
ref_plain=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
k_lens=torch.tensor([3],device='cuda',dtype=torch.int32)
got_masked=flash_attention(q,k,v,k_lens=k_lens,version=2)
mask=torch.arange(7,device='cuda')[None,None,None,:] < k_lens[:,None,None,None]
ref_masked=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask).transpose(1,2)
r={'任务':'SDPA回退数值与可变key长度语义验证','plain最大差':float((got_plain-ref_plain).abs().max()),'k_lens最大差':float((got_masked-ref_masked).abs().max()),'shape':list(got_masked.shape),'finite':bool(torch.isfinite(got_masked).all())}
r['通过']=r['plain最大差']==0 and r['k_lens最大差']<=0.01 and r['finite']
print(json.dumps(r,ensure_ascii=False,indent=2))
raise SystemExit(0 if r['通过'] else 2)
