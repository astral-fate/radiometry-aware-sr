# -*- coding: utf-8 -*-
r"""The figure that shows what the study is actually doing.

One ground square, every sensor that images it, and what the model makes of it.
The point is to make the problem legible: a 64 px SPOT tile and a 512 px aerial
tile cover the SAME 102.4 m, and the model's job is to get from one to the other.

The preprocessing here mirrors the training path exactly -- SPOT as
`(uint16/10000)/0.3` clamped, aerial as `uint8/255`. Displaying a differently
normalised image next to a model output trained on the other convention would
make the figure a lie even though every panel is real.

    python make_qualitative_figure.py --pack data/flair-parquet-v3 \
        --ckpt <edsr__spot_to_aerial_last.pt> --out <dir>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_DOMAINS = ("D040-2021", "D044-2022")
GRID = 102.4          # metres per tile side


def load_tiles(shard, want, n_scan=400):
    """Return candidate tiles, richest in built structure first.

    nDSM is the selector: a tile with tall objects shows the difference between
    3 px and 512 px far better than a field does, and the figure exists to show
    that difference.
    """
    import pyarrow.parquet as pq
    import zstandard
    from PIL import Image
    D = zstandard.ZstdDecompressor()
    cols = ["stem", "domain", "rgb", "s2", "s2_shape", "spot", "spot_shape", "ndsm"]
    out = []
    for b in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=cols):
        d = b.to_pydict()
        for j in range(len(d["stem"])):
            if len(out) >= n_scan:
                break
            if not (d["s2"][j] and d["spot"][j] and d["ndsm"][j]):
                continue
            nd = (np.frombuffer(D.decompress(d["ndsm"][j]), np.uint16)
                  .astype(np.float32) / 100.0)
            px = int(round(len(nd) ** 0.5))
            nd = nd.reshape(px, px)
            tall = float((nd > 3.0).mean())
            # A closed forest canopy scores ~1.0 on "fraction above 3 m" and is
            # useless here: it has no edges, so nothing distinguishes 1.6 m from
            # 0.2 m. What shows the problem is STRUCTURE WITH GROUND BETWEEN IT.
            # Score tiles that are part-tall, part-flat, and reject the extremes.
            if not (0.05 < tall < 0.55):
                continue
            score = float(nd.std()) * (1.0 - abs(tall - 0.3))
            out.append((score, d["stem"][j], j, b, nd))
        if len(out) >= n_scan:
            break
    out.sort(key=lambda x: -x[0])
    return out[:want]


def decode(batch, j):
    """Every modality for one tile, in the training convention."""
    import zstandard
    from PIL import Image
    D = zstandard.ZstdDecompressor()
    d = batch.to_pydict()
    aerial = np.array(Image.open(io.BytesIO(d["rgb"][j]))).astype(np.float32) / 255.
    spot = (np.frombuffer(D.decompress(d["spot"][j]), np.uint16)
            .reshape(json.loads(d["spot_shape"][j]))[:3].astype(np.float32) / 10000.)
    spot = np.clip(spot / 0.3, 0, 1).transpose(1, 2, 0)
    s2 = (np.frombuffer(D.decompress(d["s2"][j]), np.uint16)
          .reshape(json.loads(d["s2_shape"][j])).astype(np.float32) / 10000.)
    s2 = np.clip(np.median(s2, axis=0)[[2, 1, 0]] / 0.3, 0, 1).transpose(1, 2, 0)
    nd = (np.frombuffer(D.decompress(d["ndsm"][j]), np.uint16).astype(np.float32) / 100.)
    px = int(round(len(nd) ** 0.5))
    return dict(aerial=aerial, spot=spot, s2=s2, ndsm=nd.reshape(px, px),
                stem=d["stem"][j], domain=d["domain"][j])


def run_model(ckpt_path, spot_hwc, out_px, core_dir):
    """Bicubic control and EDSR prediction, both at the target grid."""
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, core_dir)
    import flair_sr_core as C
    lo = torch.from_numpy(spot_hwc.transpose(2, 0, 1))[None].float()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lo = lo.to(dev)
    bic = F.interpolate(lo, size=(out_px, out_px), mode="bicubic",
                        align_corners=False).clamp(0, 1)
    model = C.MODELS["edsr"]().to(dev).eval()
    st = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(st["model"])
    with torch.no_grad():
        pred = model(lo, (out_px, out_px)).clamp(0, 1)
    f = lambda t: t[0].cpu().numpy().transpose(1, 2, 0)
    return f(bic), f(pred), int(st.get("epoch", -1))


def show(x, lo=1.0, hi=99.0):
    """Percentile stretch for DISPLAY ONLY.

    The model is fed the training normalisation and its output is measured in
    that space; but `(reflectance/10000)/0.3` renders a dark scene almost black,
    which hides the very differences this figure exists to show. Every panel
    gets the same treatment so the comparison stays fair, and the caption says
    the stretch was applied.
    """
    a = np.asarray(x, dtype=np.float32)
    p_lo, p_hi = np.percentile(a, lo), np.percentile(a, hi)
    if p_hi - p_lo < 1e-6:
        return np.clip(a, 0, 1)
    return np.clip((a - p_lo) / (p_hi - p_lo), 0, 1)


def sharpness(x):
    g = np.asarray(x).mean(-1) if x.ndim == 3 else np.asarray(x)
    gy, gx = np.gradient(g)
    return float(np.hypot(gx, gy).mean())


def build(t, bic, pred, out, crop=160):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    mpl.rcParams.update({
        "figure.dpi": 220, "savefig.dpi": 220, "font.size": 7.5,
        "axes.titlesize": 8, "text.color": "#27272A", "axes.labelcolor": "#27272A"})

    A, S, S2, ND = t["aerial"], t["spot"], t["s2"], t["ndsm"]
    px = A.shape[0]
    # Pick the crop where the TARGET actually has detail, not where nDSM peaks.
    # The tallest pixel is often a lone tree beside open water, and a crop of
    # flat water makes the target look blurrier than the model - which reads as
    # the model out-sharpening the truth. Maximise the aerial gradient instead:
    # the zoom then shows the content the sharpness ratio is computed over.
    g = np.asarray(A).mean(-1)
    gy, gx = np.gradient(g)
    energy = np.hypot(gx, gy)
    step = max(px // 24, 1)
    best, y0, x0 = -1.0, 0, 0
    for yy in range(0, px - crop + 1, step):
        for xx in range(0, px - crop + 1, step):
            e = float(energy[yy:yy + crop, xx:xx + crop].mean())
            if e > best:
                best, y0, x0 = e, yy, xx
    sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))

    fig, ax = plt.subplots(2, 4, figsize=(9.6, 5.9))
    src = [
        (S2,  f"Sentinel-2\n{S2.shape[0]}$\\times${S2.shape[0]} px  "
              f"{GRID/S2.shape[0]:.2f} m/px"),
        (S,   f"SPOT  (model input)\n{S.shape[0]}$\\times${S.shape[0]} px  "
              f"{GRID/S.shape[0]:.2f} m/px"),
        (ND,  f"nDSM (elevation)\n{ND.shape[0]}$\\times${ND.shape[0]} px  "
              f"{GRID/ND.shape[0]:.2f} m/px"),
        (A,   f"Aerial  (target)\n{px}$\\times${px} px  {GRID/px:.2f} m/px"),
    ]
    for k, (img, title) in enumerate(src):
        a = ax[0, k]
        a.imshow(show(img), cmap="magma" if img.ndim == 2 else None,
                 interpolation="nearest")
        a.set_title(title, loc="left")
        a.add_patch(Rectangle((x0 * img.shape[0] / px, y0 * img.shape[0] / px),
                              crop * img.shape[0] / px, crop * img.shape[0] / px,
                              fill=False, ec="#00E5FF", lw=1.0))
        a.set_xticks([]); a.set_yticks([])

    zoom = [
        (S,    "SPOT input (nearest)", None),
        (bic,  "Bicubic control", sharpness(bic[sl])),
        (pred, "EDSR", sharpness(pred[sl])),
        (A,    "Aerial target", sharpness(A[sl])),
    ]
    for k, (img, title, sh) in enumerate(zoom):
        a = ax[1, k]
        s = img.shape[0]
        if s == px:
            a.imshow(show(img[sl]), interpolation="nearest")
        else:                      # SPOT: same ground window, coarser grid
            f = s / px
            a.imshow(show(img[int(y0 * f):int((y0 + crop) * f),
                              int(x0 * f):int((x0 + crop) * f)]),
                     interpolation="nearest")
        a.set_title(title + (f"\ngradient {sh:.4f}" if sh else "\n "), loc="left")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor("#00E5FF"); sp.set_linewidth(1.0)

    fig.suptitle(f"One {GRID:.1f} m ground square, four sensors "
                 f"({t['domain']} / {t['stem']})", fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.4)
    fig.text(0.5, -0.012, f"Bottom row: {crop} px window = "
             f"{crop*GRID/px:.1f} m on the ground, at the target's densest "
             "detail. All panels percentile-stretched for display; the "
             "model sees the training normalisation.",
             ha="center", va="top", fontsize=7.5, color="#52525B")
    p = os.path.join(out, "fig_qualitative.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
    print("  wrote", p)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=r"data/flair-parquet-v3")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="D040-2021")
    ap.add_argument("--rank", type=int, default=0, help="which candidate tile")
    ap.add_argument("--core", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "modal_app"))
    a = ap.parse_args(argv)
    assert a.domain in TEST_DOMAINS, f"{a.domain} is not a held-out domain"
    os.makedirs(a.out, exist_ok=True)

    shard = os.path.join(a.pack, f"{a.domain}.parquet")
    cands = load_tiles(shard, want=6)
    print(f"  {len(cands)} candidates; structure score "
          f"{[round(c[0], 2) for c in cands]}")
    built, stem, j, batch, _ = cands[a.rank]
    t = decode(batch, j)
    print(f"  tile {t['stem']}  structure score {built:.2f}")

    bic, pred, ep = run_model(a.ckpt, t["spot"], t["aerial"].shape[0], a.core)
    print(f"  model from epoch {ep};  gradient  bicubic {sharpness(bic):.4f}  "
          f"edsr {sharpness(pred):.4f}  target {sharpness(t['aerial']):.4f}")
    build(t, bic, pred, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
