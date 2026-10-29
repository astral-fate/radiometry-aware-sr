# -*- coding: utf-8 -*-
r"""Watch Modal training runs from this machine.

The Modal functions push a checkpoint and a log row to the Hub every epoch. That
is what makes them resumable; this reads the same files so the compute can sit on
a rented GPU while the inspection stays local.

    python watch_runs.py --repo USER/flair-sr-512-runs
    python watch_runs.py --repo USER/flair-sr-512-runs --follow --every 120
    python watch_runs.py --repo USER/flair-sr-512-runs --pull runs

`--pull` mirrors the logs locally so they survive the repo being rewritten, and
so make_sr_paper_v2.py can regenerate the paper from them directly.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import time


def fetch(api, repo, pull_to):
    """Return {(arch,regime): rows} for every log CSV currently in the repo."""
    from huggingface_hub import hf_hub_download
    files = [f for f in api.list_repo_files(repo, repo_type="model")
             if f.endswith(".csv") or f.endswith("_test.json")]
    runs, tests = {}, {}
    for f in files:
        try:
            p = hf_hub_download(repo, f, repo_type="model")
        except Exception:
            continue
        if pull_to:
            dst = os.path.join(pull_to, f.replace("/", "_"))
            os.makedirs(pull_to, exist_ok=True)
            try:
                open(dst, "wb").write(open(p, "rb").read())
            except Exception:
                pass
        base = os.path.basename(f)
        if f.endswith("_test.json"):
            import json
            try:
                d = json.load(open(p))
                # same filter as the CSVs: a segmentation _test.json parses and
                # would otherwise pad the summary line with dozens of names
                if "sharpness_ratio" in d or "d_psnr_matched" in d:
                    tests[base] = d
            except Exception:
                pass
            continue
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        # only SR logs: a segmentation CSV parses fine and then renders as a
        # row of dashes, which buries the runs actually being watched
        if rows and any(k in rows[0] for k in ("psnr", "d_psnr_matched")):
            runs[base[:-4]] = rows
    return runs, tests


def show(runs, tests):
    if not runs and not tests:
        print("  no logs in the repo yet - the first epoch has not been pushed")
        return
    print(f"  {'run':28}{'ep':>4}{'l1':>9}{'psnr':>8}{'Draw':>8}{'Dmatch':>8}"
          f"{'sharp':>8}{'s/ep':>7}")
    for name in sorted(runs):
        r = runs[name]
        last = r[-1]
        def g(k, f="{:.3f}"):
            v = last.get(k)
            try:
                return f.format(float(v))
            except (TypeError, ValueError):
                return "-"
        # the sign of the matched delta is the number that matters; flag a run
        # that has gone positive, because that would be the first time any of
        # these models beat the matched control
        flag = ""
        try:
            if float(last.get("d_psnr_matched", -1)) > 0:
                flag = "  <-- BEATS MATCHED CONTROL"
        except (TypeError, ValueError):
            pass
        print(f"  {name[:27]:28}{last.get('epoch','?'):>4}{g('l1','{:.4f}'):>9}"
              f"{g('psnr','{:.2f}'):>8}{g('d_psnr_raw','{:+.2f}'):>8}"
              f"{g('d_psnr_matched','{:+.2f}'):>8}{g('sharpness_ratio'):>8}"
              f"{g('secs','{:.0f}'):>7}{flag}")
    if tests:
        print(f"\n  finished ({len(tests)}): "
              + ", ".join(sorted(t.replace('_test.json', '') for t in tests)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=int, default=120)
    ap.add_argument("--pull", default="")
    ap.add_argument("--all", action="store_true",
                    help="include logs without SR metric columns")
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF")
    api = HfApi(token=tok)
    while True:
        try:
            runs, tests = fetch(api, a.repo, a.pull)
            print(f"\n[{time.strftime('%H:%M:%S')}] {a.repo}")
            show(runs, tests)
        except Exception as e:
            print(f"  poll failed ({type(e).__name__}: {str(e)[:80]}) - retrying")
        if not a.follow:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
