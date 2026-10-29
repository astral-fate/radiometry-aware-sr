# -*- coding: utf-8 -*-
r"""Super-resolve one tile with a released checkpoint, and score it if a target is given.

Inputs follow the training normalisation in modal_app/flair_sr_modal.py:
  * `.npy` holding reflectance x 10000 (uint16 SPOT, or an S2 RGB composite),
    shape (3,h,w) or (h,w,3), RGB order -> (x / 10000 / 0.3) clamped to [0,1];
  * `.png`/`.jpg` -> x / 255 (only meaningful for a model trained on 8-bit input).
A target (`.png` aerial at uint8, or `.npy` as above) adds raw and radiometrically
matched PSNR for the model and for the bicubic control, using the paper's
`match_radiometry` (each output fitted to the target separately).

    python scripts/eval/super_resolve.py --ckpt weights/v2_seeds/spot_to_aerial/checkpoints/edsr__spot_to_aerial_last.pt \
        --arch edsr --input spot_tile.npy --size 512 --target aerial_tile.png --out out/

Unguided architectures only (srcnn, edsr, rcan, swinir); the nDSM-guided and DDPM
models need the guide / sampler in modal_app/flair_sr_guided.py.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "modal_app"))


def load(path):
    if path.endswith(".npy"):
        a = np.load(path).astype(np.float32)
        if a.ndim == 3 and a.shape[-1] in (3, 4) and a.shape[0] not in (3, 4):
            a = a.transpose(2, 0, 1)
        return np.clip(a[:3] / 10000.0 / 0.3, 0, 1)
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    return a.transpose(2, 0, 1)


def save(t, path):
    from PIL import Image
    a = (t[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    Image.fromarray(a).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", required=True, choices=["srcnn", "edsr", "rcan", "swinir"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--size", type=int, help="output side in px (default: target's)")
    ap.add_argument("--target")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    import flair_sr_core as C

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lo = torch.from_numpy(load(a.input))[None].to(dev)
    y = torch.from_numpy(load(a.target))[None].to(dev) if a.target else None
    size = a.size or (y.shape[-1] if y is not None else None)
    if size is None:
        sys.exit("give --size or --target")

    model = C.MODELS[a.arch]().to(dev).eval()
    st = torch.load(a.ckpt, map_location=dev)
    model.load_state_dict(st["model"])
    with torch.no_grad():
        pred = model(lo, (size, size)).clamp(0, 1)
        bic = F.interpolate(lo, size=(size, size), mode="bicubic",
                            align_corners=False).clamp(0, 1)

    os.makedirs(a.out, exist_ok=True)
    save(pred, os.path.join(a.out, f"sr_{a.arch}.png"))
    save(bic, os.path.join(a.out, "bicubic.png"))
    print(f"{a.arch} (epoch {st.get('epoch', '?')}): {tuple(lo.shape[-2:])} -> {size}px, "
          f"written to {a.out}")

    if y is not None:
        if y.shape[-1] != size:
            y = F.interpolate(y, size=(size, size), mode="area")
        for name, z in (("bicubic", bic), (a.arch, pred)):
            raw = C.psnr(z, y)
            mat = C.psnr(C.match_radiometry(z, y), y)
            print(f"  {name:8} PSNR raw {raw:6.2f} dB   matched {mat:6.2f} dB   "
                  f"G {C.sharpness(z):.4f}")
        print(f"  target   G {C.sharpness(y):.4f}")


if __name__ == "__main__":
    main()
