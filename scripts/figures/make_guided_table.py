# -*- coding: utf-8 -*-
r"""Guided vs unguided table, generated from the run logs.

Paired by construction: same architecture, same regime, same seed set, same
test tiles, same bank. The only difference between the two columns is whether
the elevation head was attached, and that head starts at zero, so the rows are
directly comparable.

    python make_guided_table.py --guided <dir> --unguided <dir> \
        --bicubic <bicubic__spot_to_aerial_test.json> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

ARCH = ("edsr", "srcnn")
LABEL = {"edsr": "EDSR", "srcnn": "SRCNN"}


def collect(d, bic):
    """(arch, guided) -> per-metric (mean, sd, n)."""
    out = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(d, "*_test.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("regime") != "spot_to_aerial" or r.get("arch") not in ARCH:
            continue
        k = (r["arch"], bool(r.get("guided")))
        out[k]["dm"].append(r["d_psnr_matched"])
        out[k]["sh"].append(r["sharpness_ratio"] / bic["sharpness_ratio"])
        lp = r.get("lpips")
        # 0.0 is not a legitimate LPIPS here: it is what a container that could
        # not load the metric leaves behind, and it is the best possible value.
        if isinstance(lp, (int, float)) and lp == lp and lp > 0:
            out[k]["lp"].append(lp)
        out[k]["seed"].append(r.get("seed"))
    agg = {}
    for k, v in out.items():
        agg[k] = {m: (st.mean(v[m]), st.stdev(v[m]) if len(v[m]) > 1 else 0.0,
                      len(v[m])) for m in ("dm", "sh", "lp")}
        agg[k]["seeds"] = sorted(x for x in v["seed"] if x is not None)
    return agg


def fmt(t, prec, pm=True):
    m, sd, n = t
    return f"${m:+.{prec}f}$" if not pm or n < 2 else \
           f"${m:+.{prec}f}\\,{{\\pm}}\\,{sd:.{prec}f}$"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--guided", required=True)
    ap.add_argument("--unguided", required=True)
    ap.add_argument("--bicubic", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    bic = json.load(open(a.bicubic))

    g = collect(a.guided, bic)
    u = collect(a.unguided, bic)
    both = {k: v for k, v in list(g.items()) + list(u.items())}

    L = [r"\begin{table*}[t]\centering\footnotesize",
         r"\caption{Elevation guidance on SPOT$\rightarrow$aerial, against the "
         r"same architectures without it. Mean\,$\pm$\,sd over the seed set "
         r"$\{1337, 2026, 7, 41, 913\}$; $n$ given per row. The guided head is "
         r"$11{,}299$ parameters and is zero-initialised, so the two rows of a "
         r"pair start from an identical model. Sharpness is relative to the "
         r"bicubic control.}",
         r"\label{tab:guided}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}llrrr@{}}\toprule",
         r"Model & & $\Delta$matched & Sharpness & LPIPS$\downarrow$\\\midrule"]
    for arch in ARCH:
        for guided in (False, True):
            k = (arch, guided)
            if k not in both:
                continue
            d = both[k]
            tag = r"\,+\,nDSM" if guided else "baseline"
            name = f"\\textsc{{{LABEL[arch].lower()}}}" if not guided else ""
            L.append(
                f"{name} & {tag} & {fmt(d['dm'],3)} & "
                f"${d['sh'][0]:.2f}\\,{{\\pm}}\\,{d['sh'][1]:.2f}\\times$ & "
                f"${d['lp'][0]:.4f}$ \\scriptsize($n{{=}}{d['dm'][2]}$)\\\\")
        if arch != ARCH[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule\end{tabular}\end{table*}"]
    p = os.path.join(a.out, "tab_guided.tex")
    open(p, "w", encoding="utf-8").write("\n".join(L))
    print("  wrote", p)
    for k in sorted(both, key=lambda x: (x[0], x[1])):
        d = both[k]
        print(f"   {k[0]:6} {'guided' if k[1] else 'base  '} n={d['dm'][2]} "
              f"dm {d['dm'][0]:+.4f}+/-{d['dm'][1]:.4f}  sharp {d['sh'][0]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
