# -*- coding: utf-8 -*-
r"""Regenerate the camera-ready Fig. 2 and Fig. 3 from results/results.json.

Wraps make_sr_paper_v2.py with the two camera-ready changes: no "256-tile subset"
hatching on DDPM (its entry in results.json is the full 1,332-tile run scored
against the shared control), and no hard-coded, uncited "+-0.1 dB" band.

    python scripts/figures/regen_paper_figs.py --out paper/
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_sr_paper_v2 as M  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "..", "..", "results", "results.json"))
    ap.add_argument("--out", default="figures_out")
    a = ap.parse_args(argv)

    M.SUBSET_EVAL = set()
    T = {tuple(k.split("__")): v for k, v in json.load(open(a.results)).items()}
    ddpm = T[("ddpm", "s2_to_spot")]
    assert ddpm["n_eval"] == 1332
    assert abs(ddpm["psnr_bicubic"] - T[("bicubic", "s2_to_spot")]["psnr_matched"]) < 1e-9
    os.makedirs(a.out, exist_ok=True)
    M.fig_perceptual(T, a.out)

    import matplotlib.pyplot as plt
    M._style()

    def raw(d, b):
        if not d or not b:
            return None
        v = M.fnum(d, "d_psnr_raw")
        return v if v is not None else M.fnum(d, "psnr") - M.fnum(b, "psnr")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    M._grouped(axes[0], T, raw, 0.0, r"$\Delta$PSNR (dB)", "a  Against the raw bicubic control")
    M._grouped(axes[1], T, lambda d, b: M.fnum(d, "d_psnr_matched") if d else None,
               0.0, "", "b  Against the radiometrically matched control")
    M._legend(fig, axes, T, M._note(T))
    p = os.path.join(a.out, "fig_psnr_paradox.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight")
    print("  wrote fig_psnr_paradox.pdf")


if __name__ == "__main__":
    main()
