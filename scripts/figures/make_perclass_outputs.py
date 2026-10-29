# -*- coding: utf-8 -*-
r"""Per-class table and figure from perclass.json.

Reported as RECOVERY FRACTION -- the model's mean gradient over a class divided
by the target's over the same pixels. That is more interpretable than a ratio
against bicubic, because bicubic's own performance varies by class: it does
comparatively well on buildings, whose edges survive at 1.6 m, and almost
nothing on crop texture, which does not. A ratio against a moving baseline
flatters the classes where the baseline is worst.
"""
from __future__ import annotations

import argparse
import json
import os

ARCH_COLOR = {"bicubic": "#6B7280", "edsr": "#D55E00", "guided": "#0072B2"}
LABEL = {"bicubic": "Bicubic control", "edsr": "EDSR", "guided": "EDSR + nDSM"}
GRID_INK = "#D4D4D8"
TEXT_INK = "#27272A"


SHORT = {"Herbaceous vegetation": "Herbaceous veg.",
         "Impervious surface": "Impervious surf.",
         "Agricultural land": "Agricultural",
         "Pervious surface": "Pervious surf."}


def table(rows, n_tiles, out, top=8):
    sel = [r for r in rows if r["share"] >= 0.005][:top]
    L = [r"\begin{table*}[t]\centering\footnotesize",
         r"\caption{Detail recovery by land-cover class on "
         r"SPOT$\rightarrow$aerial, over " + f"{n_tiles}" + r" held-out tiles. "
         r"\emph{Target} is the mean gradient magnitude of the aerial truth "
         r"over pixels of that class---how much high-frequency content exists "
         r"to recover. The model columns give the fraction of it recovered. "
         r"Classes are ordered by share of test pixels.}",
         r"\label{tab:perclass}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
         r"Class & Share & Target & Bicubic & EDSR & +\,nDSM\\\midrule"]
    for r in sel:
        t = r["target"]
        f = lambda k: (f"{100*r[k]/t:.1f}\\%" if t > 0 and k in r else "--")
        L.append(f"{r['class']} & {100*r['share']:.1f}\\% & {t:.4f} & "
                 f"{f('bicubic')} & {f('edsr')} & \\textbf{{{f('guided')}}}\\\\")
    L += [r"\bottomrule\end{tabular}\end{table*}"]
    p = os.path.join(out, "tab_perclass.tex")
    open(p, "w", encoding="utf-8").write("\n".join(L))
    print("  wrote", os.path.basename(p))
    return sel


def figure(sel, out):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID_INK, "text.color": TEXT_INK,
        "axes.labelcolor": TEXT_INK, "xtick.color": TEXT_INK,
        "ytick.color": TEXT_INK, "grid.color": GRID_INK,
        "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True})

    names = [r["class"] for r in sel][::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 0.42 * len(names) + 1.9),
                           gridspec_kw={"width_ratios": [2.1, 1]})

    h = 0.26
    for i, k in enumerate(("bicubic", "edsr", "guided")):
        v = [100 * r[k] / r["target"] if r["target"] > 0 and k in r else 0
             for r in sel][::-1]
        ax[0].barh(y + (i - 1) * h, v, h * 0.88, color=ARCH_COLOR[k],
                   label=LABEL[k], zorder=3)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names)
    ax[0].set_xlabel("percent of the target's gradient energy recovered")
    ax[0].set_title("a  How much detail is recovered", loc="left")
    ax[0].legend(frameon=False, loc="lower right")
    ax[0].grid(axis="y", visible=False)

    tv = [r["target"] for r in sel][::-1]
    ax[1].barh(y, tv, 0.62, color="#27272A", zorder=3)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel("target gradient magnitude")
    ax[1].set_title("b  How much exists to recover", loc="left")
    ax[1].grid(axis="y", visible=False)

    fig.tight_layout()
    p = os.path.join(out, "fig_perclass.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
    print("  wrote", os.path.basename(p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(argv)
    d = json.load(open(a.json))
    sel = table(d["rows"], d["n_tiles"], a.out, a.top)
    figure(sel, a.out)
    print()
    for r in sel:
        t = r["target"]
        print(f"  {r['class'][:22]:23} target {t:.4f}  bicubic "
              f"{100*r['bicubic']/t:5.1f}%  EDSR {100*r['edsr']/t:5.1f}%  "
              f"+nDSM {100*r.get('guided',0)/t:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
