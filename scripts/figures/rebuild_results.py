# -*- coding: utf-8 -*-
r"""Rebuild results/ from the run logs on the Hub, and check it against the committed copy.

    python scripts/figures/rebuild_results.py              # -> build/results/, then diff
    python scripts/figures/rebuild_results.py --logs weights   # reuse a local download

Merge order matters: v2_seeds holds the five-seed sweep, v2_512 the seed-1337 runs
of every architecture (including the single-seed DDPM and S2->aerial rows). Both
contain files named `<arch>__<regime>_test.json` for seed 1337, and the paper uses
the v2_512 copy, so v2_512 is laid over v2_seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def flatten(logs, run, dst, regime="*"):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for p in glob.glob(os.path.join(logs, run, regime, "logs", "*")):
        shutil.copy2(p, dst)
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="", help="local copy of the weights repo (logs only is enough)")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "results"))
    a = ap.parse_args(argv)
    py = sys.executable

    logs = a.logs or os.path.join(ROOT, "build", "hub_logs")
    if not a.logs:
        subprocess.run([py, os.path.join(ROOT, "scripts", "hub", "download_weights.py"),
                        "--logs-only", "--out", logs], check=True)

    flat = os.path.join(ROOT, "build", "flat")
    shutil.rmtree(flat, ignore_errors=True)
    main_dir = os.path.join(flat, "main")
    flatten(logs, "v2_seeds", main_dir)
    flatten(logs, "v2_512", main_dir)          # seed-1337 files win
    subprocess.run([py, os.path.join(HERE, "make_sr_paper_v2.py"),
                    "--logs", main_dir, "--out", a.out], check=True)

    g, u = os.path.join(flat, "guided"), os.path.join(flat, "unguided")
    flatten(logs, "v3_guided", g, "spot_to_aerial")
    flatten(logs, "v2_seeds", u, "spot_to_aerial")
    subprocess.run([py, os.path.join(HERE, "make_guided_table.py"), "--guided", g,
                    "--unguided", u, "--bicubic",
                    os.path.join(logs, "v2_512", "spot_to_aerial", "logs",
                                 "bicubic__spot_to_aerial_test.json"), "--out", a.out],
                   check=True)

    ref = json.load(open(os.path.join(ROOT, "results", "results.json")))
    new = json.load(open(os.path.join(a.out, "results.json")))
    bad = [(k, f) for k in ref for f, v in ref[k].items()
           if isinstance(v, (int, float)) and not isinstance(v, bool)
           and abs(v - (new.get(k, {}).get(f) or 0)) > 1e-6]
    if bad:
        sys.exit(f"results.json differs in {len(bad)} values, e.g. {bad[:3]}")
    print(f"results.json: {len(ref)} rows reproduced exactly from the logs")
    for t in ("tab_guided.tex",):
        same = open(os.path.join(a.out, t)).read() == open(os.path.join(ROOT, "results", t)).read()
        print(f"{t}: {'identical' if same else 'DIFFERS from results/' + t}")


if __name__ == "__main__":
    main()
