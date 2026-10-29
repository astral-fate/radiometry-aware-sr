# -*- coding: utf-8 -*-
r"""Prove the matched control removes the radiometric confound and nothing else.

The claim the fix rests on: a large part of the reported PSNR gain is a per-image
affine radiometric difference between two real sensors, which the network can
correct and bicubic cannot. If that is right, then giving bicubic the affine
correction must
  1. hugely raise its PSNR when the only defect is radiometric, and
  2. barely change it when the only defect is blur.
If instead the fit also absorbs genuine detail, test 2 would improve a lot too, and
the fix would be hiding real gains rather than exposing a fake one.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.manual_seed(0)


def match_radiometry(pred, tgt):
    B, C = pred.shape[:2]
    p = pred.reshape(B, C, -1)
    t = tgt.reshape(B, C, -1)
    pm = p.mean(-1, keepdim=True)
    tm = t.mean(-1, keepdim=True)
    var = ((p - pm) ** 2).mean(-1, keepdim=True).clamp_min(1e-8)
    cov = ((p - pm) * (t - tm)).mean(-1, keepdim=True)
    a = cov / var
    b = tm - a * pm
    return (a * p + b).reshape_as(pred).clamp(0, 1)


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean().item()


def sharp(x):
    gx = (x[..., 1:] - x[..., :-1]).abs().mean()
    gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return float((gx + gy) / 2)


B, C, S = 8, 3, 128
# a structured target: smooth field plus fine texture, so blur is measurable
yy, xx = torch.meshgrid(torch.linspace(0, 6, S), torch.linspace(0, 6, S),
                        indexing="ij")
base = (torch.sin(yy) * torch.cos(xx) * 0.25 + 0.5).expand(B, C, S, S).clone()
fine = torch.rand(B, C, S, S) * 0.22
tgt = (base + fine).clamp(0, 1)

print("test 1 - defect is PURELY radiometric (same image, gain+bias applied)")
gain = torch.empty(B, C, 1, 1).uniform_(0.35, 0.6)
bias = torch.empty(B, C, 1, 1).uniform_(0.05, 0.3)
radio = (tgt * gain + bias).clamp(0, 1)
r0, r1 = psnr(radio, tgt), psnr(match_radiometry(radio, tgt), tgt)
print(f"  raw {r0:6.2f} dB  ->  matched {r1:6.2f} dB   ({r1-r0:+.2f})")
print(f"  sharpness unchanged: {sharp(radio):.4f} -> "
      f"{sharp(match_radiometry(radio, tgt)):.4f}")

print("\ntest 2 - defect is PURELY blur (bicubic down/up, radiometry intact)")
lo = F.interpolate(tgt, scale_factor=1 / 8, mode="area")
blur = F.interpolate(lo, size=(S, S), mode="bicubic", align_corners=False).clamp(0, 1)
b0, b1 = psnr(blur, tgt), psnr(match_radiometry(blur, tgt), tgt)
print(f"  raw {b0:6.2f} dB  ->  matched {b1:6.2f} dB   ({b1-b0:+.2f})")
print(f"  sharpness {sharp(blur):.4f} vs target {sharp(tgt):.4f} "
      f"(ratio {sharp(blur)/sharp(tgt):.3f})")

print("\ntest 3 - BOTH defects, as in the real experiment")
both = (blur * gain + bias).clamp(0, 1)
c0, c1 = psnr(both, tgt), psnr(match_radiometry(both, tgt), tgt)
print(f"  raw {c0:6.2f} dB  ->  matched {c1:6.2f} dB   ({c1-c0:+.2f})")
print(f"  the matched control recovers the radiometric part; the residual gap to")
print(f"  the target ({psnr(tgt, tgt) if False else b1:.2f} dB ceiling for pure blur) is the")
print(f"  resolution part, which is what d_psnr should measure")

ok = (r1 - r0 > 10) and (b1 - b0 < 1.0)
print(f"\nradiometric defect largely removed  : {r1-r0:+.2f} dB (want >> 0)")
print(f"blur defect left essentially alone  : {b1-b0:+.2f} dB (want ~0)")
print("RESULT:", "PASS - the fit isolates radiometry, not detail" if ok else "FAIL")
sys.exit(0 if ok else 1)
