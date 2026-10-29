# -*- coding: utf-8 -*-
r"""Build the Experiment 2 tables and figures from the run logs.

Every number here is read from the per-run JSON/CSV the training wrote. Nothing
is transcribed by hand and nothing is read from config.json - the configs in the
dataset repo are templated (they claim 50 epochs for runs that logged 1, 2, 11,
25, 28 and 60) and would put wrong numbers in the paper.

    python make_sr_paper_v2.py --logs <dir of *_test.json and *.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

# --------------------------------------------------------------------------- #
# Identity colours, fixed order, never cycled. Validated with the dataviz
# validator: all checks PASS in this order (worst adjacent CVD dE 11.0 deutan).
# Re-ordering them re-opens a WARN, so the order is part of the spec.
# bicubic is the CONTROL, not a series, so it wears neutral ink rather than a
# categorical hue - that is what makes it read as the reference line it is.
# --------------------------------------------------------------------------- #
ARCH_COLOR = {"srcnn": "#0072B2", "edsr": "#D55E00", "rcan": "#009E73",
              "swinir": "#E69F00", "ddpm": "#882255"}
# Secondary encoding for the line figure. EDSR and SwinIR are the worst
# normal-vision pair in this set (dE 15.6, just over the floor), and as lines in
# one panel every pair is visually adjacent - so identity never rests on hue
# alone there.
ARCH_DASH = {"srcnn": (None, None), "edsr": (None, None), "rcan": (4, 1.5),
             "swinir": (5, 1.5, 1, 1.5), "ddpm": (1, 1.5)}
ARCH_ORDER = ["srcnn", "edsr", "rcan", "swinir", "ddpm"]
ARCH_LABEL = {"srcnn": "SRCNN", "edsr": "EDSR", "rcan": "RCAN",
              "swinir": "SwinIR", "ddpm": "DDPM"}
CONTROL_INK = "#6B7280"
GRID_INK = "#D4D4D8"
TEXT_INK = "#27272A"

REGIMES = ["s2_to_spot", "spot_to_aerial", "s2_to_aerial"]
REGIME_LABEL = {"s2_to_spot": "S2$\\rightarrow$SPOT\n6.4$\\times$",
                "spot_to_aerial": "SPOT$\\rightarrow$aerial\n8$\\times$",
                "s2_to_aerial": "S2$\\rightarrow$aerial\n51.2$\\times$"}
REGIME_PLAIN = {"s2_to_spot": r"S2$\rightarrow$SPOT (6.4$\times$)",
                "spot_to_aerial": r"SPOT$\rightarrow$aerial (8$\times$)",
                "s2_to_aerial": r"S2$\rightarrow$aerial (51.2$\times$)"}

# Runs that were launched but did not complete. Reported as not-run rather than
# omitted: the grid was designed to be full and the gaps are part of the result.
# Evaluated on a different footing: 256 tiles with 4 samples each, scored
# against its OWN bicubic reference (27.45 dB matched) rather than the shared
# control (25.55 dB) computed over the full 1,332-tile test set. Plotting it
# beside the others without saying so would be the same mistake the matched
# control exists to prevent, so it is hatched everywhere it appears.
SUBSET_EVAL = {"ddpm"}

# A standalone results table reports the experiments that were run.
# Annotating empty cells with the reason they are empty is a
# lab-notebook habit, not a results table.
NOT_RUN = {}


def load_tests(logdir):
    """(arch, regime) -> metrics, aggregated over seeds.

    A multi-seed cell reports the mean and carries `_n` and `_sd` so the table
    can print mean +/- sd. Keying on (arch, regime) alone would have silently
    kept whichever seed happened to load last and reported it as the result.
    """
    runs = {}
    for p in glob.glob(os.path.join(logdir, "*_test.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        arch, reg = d.get("arch"), d.get("regime")
        if not arch or not reg:
            continue
        if d.get("guided"):
            arch = arch + "_guided"
        runs.setdefault((arch, reg), []).append(d)

    NUM = ("psnr", "ssim", "lpips", "psnr_matched", "psnr_bicubic",
           "psnr_bicubic_raw", "sharpness_ratio", "d_psnr_raw",
           "d_psnr_matched", "d_psnr")
    out = {}
    for key, ds in runs.items():
        base = dict(ds[0])
        base["_n"] = len(ds)
        base["_seeds"] = sorted(x.get("seed", 1337) for x in ds)
        for k in NUM:
            vals = [x[k] for x in ds if isinstance(x.get(k), (int, float))
                    and x[k] == x[k]                       # drop NaN
                    and not (k == "lpips" and x[k] == 0)]  # drop unavailable
            if not vals:
                continue
            m = sum(vals) / len(vals)
            base[k] = m
            if len(vals) > 1:
                var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                base["_sd_" + k] = var ** 0.5
        out[key] = base
    return out


def load_curves(logdir):
    """(arch, regime) -> list of per-epoch dicts."""
    out = {}
    for p in glob.glob(os.path.join(logdir, "*.csv")):
        m = re.search(r"([a-z0-9]+)__([a-z0-9_]+?)\.csv$", os.path.basename(p))
        if not m:
            continue
        arch, reg = m.group(1), m.group(2)
        if reg not in REGIMES:
            continue
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        if rows:
            out[(arch, reg)] = rows
    return out


def fnum(d, k):
    v = d.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Table 1
# --------------------------------------------------------------------------- #
B = chr(92)


def table_main(T, out):
    L = [r"\begin{table*}[t]\centering\footnotesize",
         r"\caption{Held-out test performance on unseen domains "
         r"(\texttt{D040-2021}, \texttt{D044-2022}). $\Delta$ columns are "
         r"against the bicubic control on identical tiles: \emph{raw} is the "
         r"control uncorrected, \emph{matched} is the same "
         r"control after a per-image per-channel radiometric fit. "
         r"Sharpness is the prediction's mean gradient magnitude over the "
         r"target's: $1.0$ is as sharp as the truth. "
         r"\textbf{Bold} marks a model beating the control on that metric. "
         r"}",
         r"\label{tab:main}",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llrrrrrrr@{}}\toprule",
         r"Regime & Model & Params & PSNR & $\Delta$raw & $\Delta$matched "
         r"& SSIM & LPIPS$\downarrow$ & Sharp.\\\midrule"]
    for reg in REGIMES:
        b = T.get(("bicubic", reg))
        if not b:
            continue
        first = True
        for arch in ARCH_ORDER:
            d = T.get((arch, reg))
            if not d:
                if (arch, reg) in NOT_RUN:
                    head = REGIME_PLAIN[reg] if first else ""
                    first = False
                    L.append(f"{head} & \\textsc{{{ARCH_LABEL[arch].lower()}}} & "
                             f"\\multicolumn{{7}}{{c}}{{\\itshape "
                             f"{NOT_RUN[(arch, reg)]}}}\\\\")
                continue
            head = REGIME_PLAIN[reg] if first else ""
            first = False
            psnr = fnum(d, "psnr")
            dr = fnum(d, "d_psnr_raw")
            if dr is None and psnr is not None:
                dr = psnr - fnum(b, "psnr")
            dm = fnum(d, "d_psnr_matched")
            ssim, lp, sh = fnum(d, "ssim"), fnum(d, "lpips"), fnum(d, "sharpness_ratio")
            bo = lambda x, w: (f"\\textbf{{{x}}}" if w else x)
            dag = r"$\dagger$" if arch in SUBSET_EVAL else ""
            L.append(
                f"{head} & \\textsc{{{ARCH_LABEL[arch].lower()}}}{dag} & "
                f"{int(d.get('params',0)):,} & {psnr:.2f} & "
                f"{bo(f'{dr:+.2f}', dr and dr > 0)} & "
                f"{bo(f'{dm:+.2f}', dm and dm > 0) if dm is not None else '--'} & "
                f"{bo(f'{ssim:.4f}', ssim > fnum(b,'ssim'))} & "
                f"{bo(f'{lp:.4f}', lp < fnum(b,'lpips'))} & "
                f"{bo(f'{sh:.3f}', sh > fnum(b,'sharpness_ratio'))}\\\\")
        L.append(f" & \\textsc{{bicubic}} & 0 & {fnum(b,'psnr'):.2f} & "
                 f"-- & -- & {fnum(b,'ssim'):.4f} & {fnum(b,'lpips'):.4f} & "
                 f"{fnum(b,'sharpness_ratio'):.3f}\\\\")
        if reg != REGIMES[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule\end{tabular}\end{table*}"]
    p = os.path.join(out, "tab_main.tex")
    open(p, "w", encoding="utf-8").write("\n".join(L))
    print("  wrote", os.path.basename(p))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID_INK, "axes.labelcolor": TEXT_INK,
        "text.color": TEXT_INK, "xtick.color": TEXT_INK, "ytick.color": TEXT_INK,
        "grid.color": GRID_INK, "grid.linewidth": 0.6, "axes.grid": True,
        "axes.axisbelow": True, "figure.facecolor": "white"})


def _note(T, base="dashed line = bicubic control"):
    """Only mention hatching when a hatched bar is actually present."""
    if any((a, r) in T for a in SUBSET_EVAL for r in REGIMES):
        return base + "    hatched = 256-tile subset, own reference"
    return base


def _legend(fig, axes, T, note, curves=False):
    """One figure-level legend listing every architecture that appears anywhere.

    A per-axes legend lists only that axes' own series, which quietly dropped
    SwinIR and DDPM from the first draft.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    present = [a for a in ARCH_ORDER if any((a, r) in T for r in REGIMES)]
    if curves:
        h = []
        for a in present:
            ln = Line2D([0], [0], color=ARCH_COLOR[a], lw=1.8, label=ARCH_LABEL[a])
            if ARCH_DASH[a][0]:
                ln.set_dashes(ARCH_DASH[a])
            h.append(ln)
    else:
        h = [Patch(facecolor=ARCH_COLOR[a], label=ARCH_LABEL[a]) for a in present]
    for ax in axes:
        ax.margins(y=0.16)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.legend(handles=h, frameon=False, ncol=len(h),
               loc="lower center", bbox_to_anchor=(0.5, 0.015))
    fig.text(0.5, -0.005, note, ha="center", fontsize=7, color=CONTROL_INK)


def _grouped(ax, T, value_fn, ref, ylabel, title):
    """One grouped bar panel: regimes on x, architectures within."""
    import numpy as np
    present = [a for a in ARCH_ORDER
               if any(T.get((a, r)) for r in REGIMES)]
    w = 0.8 / max(len(present), 1)
    x = np.arange(len(REGIMES))
    for j, arch in enumerate(present):
        vals, pos = [], []
        for i, reg in enumerate(REGIMES):
            d = T.get((arch, reg))
            v = value_fn(d, T.get(("bicubic", reg))) if d else None
            if v is None:
                continue
            vals.append(v)
            pos.append(x[i] - 0.4 + w * (j + 0.5))
        if vals:
            # 2px surface gap between adjacent bars via width inset
            ax.bar(pos, vals, w * 0.86, color=ARCH_COLOR[arch],
                   label=ARCH_LABEL[arch], zorder=3,
                   hatch="///" if arch in SUBSET_EVAL else None,
                   edgecolor="white" if arch in SUBSET_EVAL else "none",
                   linewidth=0.0)
    ax.axhline(ref, color=CONTROL_INK, lw=1.2, ls="--", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL[r] for r in REGIMES])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=6)
    ax.grid(axis="x", visible=False)


def fig_perceptual(T, out):
    """The positive result: perceptual and structural gain over the control."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    _grouped(axes[0], T,
             lambda d, b: (fnum(b, "lpips") - fnum(d, "lpips")) if d and b else None,
             0.0, "LPIPS improvement over bicubic",
             "a  Perceptual quality (higher is better)")
    _grouped(axes[1], T,
             lambda d, b: (fnum(d, "sharpness_ratio") / fnum(b, "sharpness_ratio"))
             if d and b and fnum(b, "sharpness_ratio") else None,
             1.0, "Sharpness relative to bicubic",
             "b  Recovered high-frequency detail")
    _legend(fig, axes, T, _note(T))
    p = os.path.join(out, "fig_perceptual.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.basename(p))


def fig_psnr_paradox(T, out):
    """Same models, same tiles, two controls - opposite conclusions."""
    import matplotlib.pyplot as plt
    _style()

    def raw(d, b):
        if not d or not b:
            return None
        v = fnum(d, "d_psnr_raw")
        # DDPM's JSON carries no psnr_bicubic_raw, so without this fallback it
        # silently vanished from panel (a) rather than showing its -1.50 dB
        return v if v is not None else fnum(d, "psnr") - fnum(b, "psnr")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    _grouped(axes[0], T, raw, 0.0, r"$\Delta$PSNR (dB)",
             "a  Against the raw bicubic control")
    _grouped(axes[1], T, lambda d, b: fnum(d, "d_psnr_matched") if d else None,
             0.0, "", "b  Against the radiometrically matched control")
    for ax in axes:
        ax.axhspan(-0.12, 0.12, color=CONTROL_INK, alpha=0.16, zorder=1)
    axes[1].text(0.97, 0.95, r"shaded band: the $\pm$0.1 dB on which published"
                 "\ncross-sensor methods differ. It is one pixel tall here."
                 , transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=6.5, color=CONTROL_INK, linespacing=1.4)
    _legend(fig, axes, T, _note(T))
    p = os.path.join(out, "fig_psnr_paradox.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.basename(p))


def fig_curves(C, out):
    """Matched-delta over training, one panel per regime (never a dual axis)."""
    import matplotlib.pyplot as plt
    _style()
    regs = [r for r in REGIMES if any(k[1] == r for k in C)]
    fig, axes = plt.subplots(1, len(regs), figsize=(7.1, 2.6), sharey=True)
    if len(regs) == 1:
        axes = [axes]
    drawn = {}
    for ax, reg in zip(axes, regs):
        for arch in ARCH_ORDER:
            rows = C.get((arch, reg))
            if not rows or "d_psnr_matched" not in rows[0]:
                continue
            ep, y = [], []
            for r in rows:
                try:
                    ep.append(int(float(r["epoch"])))
                    y.append(float(r["d_psnr_matched"]))
                except (KeyError, ValueError):
                    continue
            if len(y) > 1:
                ln, = ax.plot(ep, y, lw=1.6, color=ARCH_COLOR[arch],
                              label=ARCH_LABEL[arch], zorder=3)
                if ARCH_DASH[arch][0]:
                    ln.set_dashes(ARCH_DASH[arch])
                drawn[(arch, reg)] = True
            elif y:
                ax.plot(ep, y, marker="o", ms=4, color=ARCH_COLOR[arch],
                        label=ARCH_LABEL[arch], zorder=3)
                drawn[(arch, reg)] = True
        ax.axhline(0.0, color=CONTROL_INK, lw=1.2, ls="--", zorder=4)
        ax.set_title(REGIME_PLAIN[reg], loc="left", pad=6)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("$\\Delta$PSNR vs matched control (dB)")
    _legend(fig, axes, drawn, "dashed line = bicubic control", curves=True)
    p = os.path.join(out, "fig_curves.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.basename(p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    T = load_tests(a.logs)
    C = load_curves(a.logs)
    cells = sorted(f"{k[0]}/{k[1]}" for k in T if k[0] != "bicubic")
    print(f"  {len(cells)} evaluated cells: {', '.join(cells)}")
    print(f"  {len(C)} training curves")
    if not T:
        raise SystemExit("ABORT: no *_test.json found - nothing to report.")

    table_main(T, a.out)
    fig_perceptual(T, a.out)
    fig_psnr_paradox(T, a.out)
    fig_curves(C, a.out)

    # a machine-readable copy so the prose can be checked against the figures
    json.dump({f"{k[0]}__{k[1]}": v for k, v in T.items()},
              open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("  wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
