# -*- coding: utf-8 -*-
r"""Download checkpoints and run logs from the Hugging Face weights repo.

The weights repo is private; access is granted on request. Log in once
(`hf auth login`) or set HF_TOKEN, then select what to pull:

    python scripts/hub/download_weights.py --list
    python scripts/hub/download_weights.py --run v2_seeds --regime spot_to_aerial --arch edsr
    python scripts/hub/download_weights.py --run v3_guided --logs-only
    python scripts/hub/download_weights.py --all

Layout of the repo (one folder per sweep):
    v2_512/<regime>/{checkpoints,logs}     seed 1337, every architecture
    v2_seeds/<regime>/{checkpoints,logs}   five seeds, s2_to_spot and spot_to_aerial
    v3_guided/spot_to_aerial/...           nDSM-guided EDSR and SRCNN, five seeds
Checkpoints are `<arch>[_s<seed>]__<regime>_last.pt`; seed 1337 has no suffix.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys

REPO = "FatimahEmadEldin/flair-sr-512-runs"
RUNS = ("v2_512", "v2_seeds", "v3_guided")
REGIMES = ("s2_to_spot", "spot_to_aerial", "s2_to_aerial")


def patterns(a):
    runs = [a.run] if a.run else list(RUNS)
    regs = [a.regime] if a.regime else list(REGIMES)
    arch = a.arch or "*"
    seed = "" if a.seed in (None, 1337) else f"_s{a.seed}"
    if a.seed is None:
        stem = f"{arch}*__"            # every seed
    else:
        stem = f"{arch}{seed}__"
    out = []
    for r in runs:
        for g in regs:
            out.append(f"{r}/{g}/logs/{stem}*")
            out.append(f"{r}/{g}/logs/bicubic__*")
            if not a.logs_only:
                out.append(f"{r}/{g}/checkpoints/{stem}*")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--run", choices=RUNS)
    ap.add_argument("--regime", choices=REGIMES)
    ap.add_argument("--arch", help="srcnn, edsr, rcan, swinir, ddpm, edsr_guided, srcnn_guided")
    ap.add_argument("--seed", type=int, help="1337, 2026, 7, 41 or 913 (default: all)")
    ap.add_argument("--logs-only", action="store_true")
    ap.add_argument("--all", action="store_true", help="everything (~all checkpoints)")
    ap.add_argument("--list", action="store_true", help="list matching files, download nothing")
    ap.add_argument("--out", default="weights")
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN") or None
    try:
        files = HfApi(token=token).list_repo_files(a.repo, repo_type="model")
    except (RepositoryNotFoundError, GatedRepoError):
        sys.exit(f"No access to {a.repo}. The weights are private: request access "
                 "(see README), then `hf auth login` or set HF_TOKEN.")

    pats = ["*"] if a.all else patterns(a)
    sel = sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in pats))
    if not sel:
        sys.exit("nothing matches; try --list without filters")
    if a.list:
        print("\n".join(sel))
        print(f"\n{len(sel)} files")
        return
    path = snapshot_download(a.repo, repo_type="model", allow_patterns=sel,
                             local_dir=a.out, token=token)
    print(f"{len(sel)} files -> {path}")


if __name__ == "__main__":
    main()
