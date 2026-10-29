# -*- coding: utf-8 -*-
"""SR architectures, metrics and the matched-radiometry evaluation.

EXTRACTED VERBATIM from FLAIR_RES_01_runpod.ipynb (cells 14, 16, 20 and the
match_radiometry/evaluate pair from cell 18), so that runs launched from Modal
are comparable to the results already published in
FatimahEmadEldin/Resolution-enhancement. Do not "improve" anything here: a
change to a model or a metric silently breaks comparability with every number
in the paper. New behaviour belongs in the caller.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


# ==========================================================================
# architectures  (notebook cell 14)
# ==========================================================================

import torch.nn as nn

class Bicubic(nn.Module):
    def forward(s,lo,size):
        return F.interpolate(lo,size=size,mode="bicubic",align_corners=False).clamp(0,1)

class SRCNN(nn.Module):
    def __init__(s,ch=64,n=5):
        super().__init__()
        L=[nn.Conv2d(3,ch,3,padding=1),nn.ReLU(True)]
        for _ in range(n): L+=[nn.Conv2d(ch,ch,3,padding=1),nn.ReLU(True)]
        L+=[nn.Conv2d(ch,3,3,padding=1)]
        s.body=nn.Sequential(*L)
    def forward(s,lo,size):
        base=F.interpolate(lo,size=size,mode="bicubic",align_corners=False)
        return (base+s.body(base)).clamp(0,1)

class ResBlk(nn.Module):
    def __init__(s,c):
        super().__init__()
        s.f=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(True),nn.Conv2d(c,c,3,padding=1))
    def forward(s,x): return x+0.1*s.f(x)

class EDSR(nn.Module):
    def __init__(s,ch=64,nb=8):
        super().__init__()
        s.head=nn.Conv2d(3,ch,3,padding=1)
        s.body=nn.Sequential(*[ResBlk(ch) for _ in range(nb)],nn.Conv2d(ch,ch,3,padding=1))
        s.up=nn.Sequential(nn.Conv2d(ch,ch*4,3,padding=1),nn.PixelShuffle(2),nn.ReLU(True),
                           nn.Conv2d(ch,ch*4,3,padding=1),nn.PixelShuffle(2),nn.ReLU(True))
        s.tail=nn.Conv2d(ch,3,3,padding=1)
    def forward(s,lo,size):
        base=F.interpolate(lo,size=size,mode="bicubic",align_corners=False)
        x=s.head(lo); x=x+s.body(x); x=s.up(x)
        x=F.interpolate(x,size=size,mode="bilinear",align_corners=False)
        return (base+s.tail(x)).clamp(0,1)

class CA(nn.Module):
    """Channel attention: squeeze to a per-channel scalar, learn a gate.
    Sentinel-2's information is spectral, so learning WHICH channels matter is
    not a generic trick here - it is the structure of the problem."""
    def __init__(s,c,r=8):
        super().__init__()
        s.f=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,max(4,c//r),1),
                          nn.ReLU(True),nn.Conv2d(max(4,c//r),c,1),nn.Sigmoid())
    def forward(s,x): return x*s.f(x)

class RCAB(nn.Module):
    def __init__(s,c):
        super().__init__()
        s.f=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(True),
                          nn.Conv2d(c,c,3,padding=1),CA(c))
    def forward(s,x): return x+s.f(x)

class RCAN(nn.Module):
    """Residual channel-attention network, residual over bicubic."""
    def __init__(s,ch=64,ngroups=3,nblocks=4):
        super().__init__()
        s.head=nn.Conv2d(3,ch,3,padding=1)
        s.groups=nn.ModuleList([
            nn.Sequential(*[RCAB(ch) for _ in range(nblocks)],
                          nn.Conv2d(ch,ch,3,padding=1)) for _ in range(ngroups)])
        s.tail=nn.Conv2d(ch,3,3,padding=1)
    def forward(s,lo,size):
        base=F.interpolate(lo,size=size,mode="bicubic",align_corners=False)
        x=s.head(base)
        for g in s.groups: x=x+g(x)
        return (base+s.tail(x)).clamp(0,1)

def _wp(x,ws):
    B,H,W,C=x.shape
    x=x.view(B,H//ws,ws,W//ws,ws,C)
    return x.permute(0,1,3,2,4,5).reshape(-1,ws*ws,C)

def _wr(w,ws,H,W):
    B=w.shape[0]//((H//ws)*(W//ws))
    x=w.view(B,H//ws,W//ws,ws,ws,-1)
    return x.permute(0,1,3,2,4,5).reshape(B,H,W,-1)

class SwinIRBlk(nn.Module):
    def __init__(s,dim,heads=4,ws=8,shift=0):
        super().__init__()
        s.ws,s.shift,s.h=ws,shift,heads; s.dh=dim//heads
        s.n1=nn.LayerNorm(dim); s.n2=nn.LayerNorm(dim)
        s.qkv=nn.Linear(dim,dim*3); s.proj=nn.Linear(dim,dim)
        s.mlp=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
    def forward(s,x,H,W):
        B,L,C=x.shape; r=x
        x=s.n1(x).view(B,H,W,C)
        pr=(s.ws-W%s.ws)%s.ws; pb=(s.ws-H%s.ws)%s.ws
        if pr or pb: x=F.pad(x,(0,0,0,pr,0,pb))
        Hp,Wp=x.shape[1],x.shape[2]
        if s.shift: x=torch.roll(x,(-s.shift,-s.shift),dims=(1,2))
        w=_wp(x,s.ws)
        qkv=s.qkv(w).reshape(w.shape[0],w.shape[1],3,s.h,s.dh).permute(2,0,3,1,4)
        o=F.scaled_dot_product_attention(qkv[0],qkv[1],qkv[2])
        o=s.proj(o.transpose(1,2).reshape(w.shape[0],w.shape[1],C))
        x=_wr(o,s.ws,Hp,Wp)
        if s.shift: x=torch.roll(x,(s.shift,s.shift),dims=(1,2))
        if pr or pb: x=x[:,:H,:W,:].contiguous()
        x=r+x.reshape(B,L,C)
        return x+s.mlp(s.n2(x))

class SwinIR(nn.Module):
    """Transformer SR: shallow conv features, residual Swin blocks on the
    bicubic-upsampled grid, residual output. The counterpart to the `swin`
    segmentation backbone, so both tasks ask whether attention beats convolution.
    """
    def __init__(s,ch=64,depth=6,ws=8):
        super().__init__()
        s.head=nn.Conv2d(3,ch,3,padding=1)
        s.blocks=nn.ModuleList([SwinIRBlk(ch,4,ws,0 if i%2==0 else ws//2)
                                 for i in range(depth)])
        s.mid=nn.Conv2d(ch,ch,3,padding=1)
        s.tail=nn.Conv2d(ch,3,3,padding=1)
    def forward(s,lo,size):
        base=F.interpolate(lo,size=size,mode="bicubic",align_corners=False)
        x=s.head(base); B,C,H,W=x.shape
        t=x.flatten(2).transpose(1,2)
        for b in s.blocks: t=b(t,H,W)
        y=t.transpose(1,2).reshape(B,C,H,W)
        return (base+s.tail(s.mid(y)+x)).clamp(0,1)

MODELS={"bicubic":Bicubic,"srcnn":SRCNN,"edsr":EDSR,"rcan":RCAN,"swinir":SwinIR}
for k,v in MODELS.items():
    print(f"{k:<9}{sum(p.numel() for p in v().parameters()):>10,} params")

# ==========================================================================
# metrics  (notebook cell 16)
# ==========================================================================

def psnr(a,b):
    mse=((a-b)**2).flatten(1).mean(1)
    return (10*torch.log10(1.0/mse.clamp_min(1e-12))).mean().item()

_W=None
def ssim(a,b):
    global _W
    if _W is None:
        g=torch.arange(11,device=a.device).float()-5
        g=torch.exp(-(g**2)/(2*1.5**2)); g=g/g.sum()
        _W=(g[:,None]@g[None,:]).expand(3,1,11,11).contiguous()
    C1,C2=0.01**2,0.03**2
    mu1=F.conv2d(a,_W,padding=5,groups=3); mu2=F.conv2d(b,_W,padding=5,groups=3)
    s11=F.conv2d(a*a,_W,padding=5,groups=3)-mu1*mu1
    s22=F.conv2d(b*b,_W,padding=5,groups=3)-mu2*mu2
    s12=F.conv2d(a*b,_W,padding=5,groups=3)-mu1*mu2
    return (((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s11+s22+C2))).mean().item()

def sharpness(x):
    gx=(x[...,1:]-x[...,:-1]).abs().mean(); gy=(x[...,1:,:]-x[...,:-1,:]).abs().mean()
    return ((gx+gy)/2).item()
print("metrics ready")

# ==========================================================================
# conditional diffusion  (notebook cell 20)
# ==========================================================================

RUN_DIFFUSION = False

class TimeEmb(nn.Module):
    def __init__(s, d):
        super().__init__(); s.f=nn.Sequential(nn.Linear(1,d),nn.SiLU(),nn.Linear(d,d))
    def forward(s,t): return s.f(t[:,None].float())

class DdpmUNet(nn.Module):
    """Small conditional UNet: predicts the noise added to the target, given
    the noisy target AND the bicubic upsample as conditioning."""
    def __init__(s, ch=64):
        super().__init__()
        s.temb=TimeEmb(ch)
        s.inc=nn.Conv2d(6,ch,3,padding=1)          # 3 noisy + 3 condition
        s.d1=ResBlk(ch); s.dn=nn.Conv2d(ch,ch,2,2)
        s.d2=ResBlk(ch)
        s.up=nn.ConvTranspose2d(ch,ch,2,2)
        s.u1=ResBlk(ch); s.out=nn.Conv2d(ch,3,3,padding=1)
    def forward(s,x,cond,t):
        h=s.inc(torch.cat([x,cond],1))+s.temb(t)[:,:,None,None]
        a=s.d1(h); b=s.d2(s.dn(a))
        return s.out(s.u1(s.up(b)+a))

T_STEPS=200
_betas=torch.linspace(1e-4,0.02,T_STEPS).cuda()
_alph=torch.cumprod(1-_betas,0)

@torch.no_grad()
def ddpm_sample(net, cond, steps=T_STEPS):
    """Ancestral sampling from pure noise, conditioned on the bicubic image."""
    x=torch.randn_like(cond)
    for i in reversed(range(steps)):
        t=torch.full((len(cond),),i,device=cond.device,dtype=torch.long)
        eps=net(x,cond,t.float()/T_STEPS)
        a=_alph[i]; ap=_alph[i-1] if i>0 else torch.tensor(1.0,device=cond.device)
        x0=((x-(1-a).sqrt()*eps)/a.sqrt()).clamp(0,1)
        if i>0:
            x=ap.sqrt()*x0+(1-ap).sqrt()*torch.randn_like(x)
        else:
            x=x0
    return x.clamp(0,1)

if RUN_DIFFUSION:
    tag=f"ddpm__{REGIME}"
    tj=os.path.join(ROOT,"logs",tag+"_test.json")
    # A fresh pod starts empty, so ask the Hub before concluding this arm
    # is unfinished. This is what makes the pod disposable.
    if not os.path.exists(tj): hf_pull(tj)
    ck=os.path.join(ROOT,"checkpoints",tag+"_last.pt")
    if not os.path.exists(ck) and hf_pull(ck):
        print(f"  {tag}: pulled checkpoint from the Hub")
    if os.path.exists(tj):
        print(f"  {tag}: already complete")
    else:
        net=DdpmUNet().cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda'); ep0=1; hist=[]
        if os.path.exists(ck):
            st=torch.load(ck,map_location='cuda',weights_only=False)
            net.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
            ep0=st['epoch']+1; hist=st['hist']; print(f"  resuming at {ep0}")
        for ep in range(ep0,EPOCHS+1):
            net.train(); perm=np.random.permutation(tr); tot=n_=0; t0=time.time()
            for i in range(0,len(perm),BATCH):
                lo,hi=make_batch(perm[i:i+BATCH],REGIME,augment=True)
                cond=F.interpolate(lo,size=hi.shape[-2:],mode="bicubic",
                                   align_corners=False).clamp(0,1)
                t=torch.randint(0,T_STEPS,(len(hi),),device='cuda')
                noise=torch.randn_like(hi); a=_alph[t][:,None,None,None]
                xt=a.sqrt()*hi+(1-a).sqrt()*noise
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=net(xt,cond,t.float()/T_STEPS)
                L=F.mse_loss(pred.float(),noise)
                scaler.scale(L).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
                scaler.step(opt); scaler.update()
                tot+=float(L.detach())*len(hi); n_+=len(hi)
            hist.append({"epoch":ep,"eps_mse":tot/n_,"secs":round(time.time()-t0,1)})
            print(f"    ep {ep:2d}/{EPOCHS}  eps-MSE {tot/n_:.4f}  [{time.time()-t0:.0f}s]")
            torch.save({"model":net.state_dict(),"opt":opt.state_dict(),
                        "epoch":ep,"hist":hist},ck)
            with open(os.path.join(ROOT,"logs",tag+".csv"),"w",newline="") as f:
                w_=csv.DictWriter(f,fieldnames=list(hist[0].keys())); w_.writeheader(); w_.writerows(hist)
            # Push every epoch: a reclaimed pod then costs one epoch.
            hf_push(ck, f"{tag}: epoch {ep}")
            hf_push(os.path.join(ROOT,"checkpoints",tag+"_best.pt"), f"{tag}: best@{ep}")
            hf_push(os.path.join(ROOT,"logs",tag+".csv"), f"{tag}: log {ep}")
        # sampling is 200 forward passes per image, so evaluate a subset
        net.eval(); sub=te[:64]
        P=S=SH=SHG=B=0.; n_=0
        for i in range(0,len(sub),8):
            lo,hi=make_batch(sub[i:i+8],REGIME)
            cond=F.interpolate(lo,size=hi.shape[-2:],mode="bicubic",
                               align_corners=False).clamp(0,1)
            out=ddpm_sample(net,cond)
            k=len(lo); n_+=k
            P+=psnr(out,hi)*k; S+=ssim(out,hi)*k; B+=psnr(cond,hi)*k
            SH+=sharpness(out)*k; SHG+=sharpness(hi)*k
        r={"psnr":P/n_,"ssim":S/n_,"psnr_bicubic":B/n_,"d_psnr":(P-B)/n_,
           "sharpness_ratio":SH/max(SHG,1e-9),"tag":tag,"arch":"ddpm",
           "regime":REGIME,"synthetic":False,"n_eval":n_,
           "params":sum(p.numel() for p in net.parameters()),
           "test_domains":TEST_DOMAINS,"ts":datetime.now(timezone.utc).isoformat()}
        json.dump(r,open(tj,"w"),indent=1)
        hf_push(tj, f"{tag}: result")
        print(f"  {tag}: PSNR {r['psnr']:.2f}  vs bicubic {r['d_psnr']:+.2f} dB  "
              f"SSIM {r['ssim']:.4f}  sharpness {r['sharpness_ratio']:.2f}")
        print(f"  NOTE: a lower PSNR than the L1 models with a HIGHER sharpness "
              f"ratio is the expected diffusion signature, not a failure.")
else:
    print("diffusion skipped (RUN_DIFFUSION = False)")

# ==========================================================================
# matched-radiometry control  (notebook cell 18)
# ==========================================================================

def match_radiometry(pred, tgt):
    """Least-squares affine fit of `pred` onto `tgt`, per image and per channel.

    Why this exists: the networks here predict a residual over bicubic, so they can
    correct the radiometric difference between two real sensors. Bicubic cannot -
    interpolation does not change a pixel's value distribution. Scoring the two
    against each other therefore measures the cross-sensor radiometric gap, not
    resolution, and the gap is large: SPOT arrives as reflectance/0.3 while the
    aerial target arrives as DN/255, and FLAIR's aerial radiometry is relative and
    domain-equalised rather than physical.

    Fitting gain and bias gives the control the same correction for free. Whatever
    advantage the network retains after that is spatial - which is the thing a
    super-resolution number is supposed to report.
    """
    B, C = pred.shape[:2]
    p = pred.reshape(B, C, -1)
    t = tgt.reshape(B, C, -1)
    pm = p.mean(-1, keepdim=True)
    tm = t.mean(-1, keepdim=True)
    var = ((p - pm) ** 2).mean(-1, keepdim=True).clamp_min(1e-8)
    cov = ((p - pm) * (t - tm)).mean(-1, keepdim=True)
    a = cov / var                      # gain
    b = tm - a * pm                    # bias
    return (a * p + b).reshape_as(pred).clamp(0, 1)

@torch.no_grad()
def evaluate(model,regime,idx,make_batch,bs=16):
    if hasattr(model,"eval"): model.eval()
    P=S=SH=SHG=B=BM=PM=0.; n=0
    for i in range(0,len(idx),bs):
        lo,hi=make_batch(idx[i:i+bs],regime)
        out=model(lo,hi.shape[-2:])
        bic=F.interpolate(lo,size=hi.shape[-2:],mode="bicubic",align_corners=False).clamp(0,1)
        k=len(lo); n+=k
        # The affine correction is an ORACLE - it is fitted against hi. Applying
        # it to the control only would let the control calibrate itself on the
        # answer while the model cannot, so both sides get it. What survives is
        # spatial, which is what a super-resolution number should report.
        bicm=match_radiometry(bic,hi); outm=match_radiometry(out,hi)
        P+=psnr(out,hi)*k; S+=ssim(out,hi)*k
        B+=psnr(bic,hi)*k; BM+=psnr(bicm,hi)*k; PM+=psnr(outm,hi)*k
        SH+=sharpness(out)*k; SHG+=sharpness(hi)*k
    # d_psnr compares like with like: both sides oracle-corrected, so the
    # difference is resolution alone. d_psnr_asym is the previous definition
    # (raw model vs corrected control) and d_psnr_raw the original confounded
    # one; keeping all three makes the size of each effect visible and lets
    # earlier runs still be compared.
    return {"psnr":P/n,"psnr_matched":PM/n,"ssim":S/n,
            "psnr_bicubic":BM/n,"psnr_bicubic_raw":B/n,
            "d_psnr":(PM-BM)/n,"d_psnr_asym":(P-BM)/n,"d_psnr_raw":(P-B)/n,
            "sharpness_ratio":SH/max(SHG,1e-9)}
