# -*- coding: utf-8 -*-
r"""Per-class recovery: which land covers does super-resolution actually help?

The aggregate sharpness ratio says a model recovers detail. It does not say
*where*. FLAIR ships COSIA land-cover labels on the target's own 0.20 m grid,
so the same gradient-magnitude measure can be taken per class: for every class
c, the mean gradient over pixels labelled c, for the target, the bicubic
control, and each model.

Reading the output: a class where the target's own gradient is high is one with
real high-frequency content to recover (built surfaces, tree crowns). A class
where it is low is intrinsically smooth (water, bare soil) and no model can add
detail that the truth does not contain -- a low ratio there is not a failure.

Runs locally on one GPU; inference only.

    python make_perclass_analysis.py --pack data/flair-parquet-v3 \
        --ckpt-base <edsr.pt> --ckpt-guided <edsr_guided.pt> --out <dir>
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

COSIA = {0: "Building", 1: "Greenhouse", 2: "Swimming pool",
         3: "Impervious surface", 4: "Pervious surface", 5: "Bare soil",
         6: "Water", 7: "Snow", 8: "Herbaceous vegetation",
         9: "Agricultural land", 10: "Plowed land", 11: "Vineyard",
         12: "Deciduous", 13: "Coniferous", 14: "Brushwood", 15: "Clear cut",
         16: "Ligneous", 17: "Mixed", 18: "Undefined"}


def grad_mag(x):
    """Per-pixel gradient magnitude of a (H,W,3) or (H,W) array."""
    g = x.mean(-1) if x.ndim == 3 else x
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=r"data/flair-parquet-v3")
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--ckpt-guided", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiles", type=int, default=250, help="per domain")
    ap.add_argument("--core", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "modal_app"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    import pyarrow.parquet as pq
    import zstandard
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, a.core)
    import flair_sr_core as C
    import flair_sr_guided as G

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device {dev}")
    base = C.MODELS["edsr"]().to(dev).eval()
    base.load_state_dict(torch.load(a.ckpt_base, map_location=dev)["model"])
    guided = None
    if a.ckpt_guided and os.path.exists(a.ckpt_guided):
        guided, _ = G.build("edsr", C, guided=True)
        guided = guided.to(dev).eval()
        guided.load_state_dict(torch.load(a.ckpt_guided, map_location=dev)["model"])
        print("  guided model loaded")

    D = zstandard.ZstdDecompressor()
    keys = ("target", "bicubic", "edsr", "guided")
    tot = {k: np.zeros(19) for k in keys}      # summed gradient per class
    cnt = np.zeros(19)                          # pixel count per class
    n_tiles = 0

    cols = ["stem", "lab", "rgb", "spot", "spot_shape", "ndsm"]
    for dom in TEST_DOMAINS:
        shard = os.path.join(a.pack, f"{dom}.parquet")
        if not os.path.exists(shard):
            continue
        seen = 0
        for b in pq.ParquetFile(shard).iter_batches(batch_size=8, columns=cols):
            d = b.to_pydict()
            for j in range(len(d["stem"])):
                if seen >= a.tiles:
                    break
                if not (d["spot"][j] and d["ndsm"][j]):
                    continue
                lab = np.array(Image.open(io.BytesIO(d["lab"][j])))
                tgt = np.array(Image.open(io.BytesIO(d["rgb"][j]))).astype(np.float32) / 255.
                px = tgt.shape[0]
                sp = (np.frombuffer(D.decompress(d["spot"][j]), np.uint16)
                      .reshape(json.loads(d["spot_shape"][j]))[:3].astype(np.float32) / 10000.)
                sp = np.clip(sp / 0.3, 0, 1)
                nd = (np.frombuffer(D.decompress(d["ndsm"][j]), np.uint16)
                      .astype(np.float32) / 100.).reshape(px, px)

                lo = torch.from_numpy(sp)[None].to(dev)
                gd = torch.from_numpy((nd / 30.).clip(0, 1))[None, None].to(dev)
                with torch.no_grad():
                    bic = F.interpolate(lo, size=(px, px), mode="bicubic",
                                        align_corners=False).clamp(0, 1)
                    pr = base(lo, (px, px)).clamp(0, 1)
                    gp = (guided(lo, (px, px), gd).clamp(0, 1)
                          if guided is not None else None)
                f = lambda t: t[0].cpu().numpy().transpose(1, 2, 0)
                gm = {"target": grad_mag(tgt), "bicubic": grad_mag(f(bic)),
                      "edsr": grad_mag(f(pr))}
                if gp is not None:
                    gm["guided"] = grad_mag(f(gp))

                for c in np.unique(lab):
                    m = lab == c
                    k = int(m.sum())
                    cnt[c] += k
                    for key, g in gm.items():
                        tot[key][c] += float(g[m].sum())
                seen += 1
                n_tiles += 1
            if seen >= a.tiles:
                break
        print(f"  {dom}: {seen} tiles")

    present = [c for c in range(19) if cnt[c] > 0]
    present.sort(key=lambda c: -cnt[c])
    rows = []
    for c in present:
        r = {"class": COSIA[c], "id": c, "share": cnt[c] / cnt.sum()}
        for k in keys:
            if tot[k].any():
                r[k] = tot[k][c] / cnt[c]
        rows.append(r)
    json.dump({"n_tiles": n_tiles, "rows": rows},
              open(os.path.join(a.out, "perclass.json"), "w"), indent=1)

    print(f"\n  {n_tiles} tiles\n")
    hdr = f"{'class':24}{'share':>7}{'target':>9}{'bicubic':>9}{'EDSR':>9}"
    if guided is not None:
        hdr += f"{'+nDSM':>9}"
    print(hdr + f"{'EDSR/bic':>10}")
    for r in rows:
        if r["share"] < 0.005:
            continue
        line = (f"{r['class'][:23]:24}{r['share']*100:6.1f}%"
                f"{r['target']:9.4f}{r['bicubic']:9.4f}{r['edsr']:9.4f}")
        if "guided" in r:
            line += f"{r['guided']:9.4f}"
        line += f"{r['edsr']/max(r['bicubic'],1e-9):9.2f}x"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
