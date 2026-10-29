# -*- coding: utf-8 -*-
r"""Cross-sensor super-resolution on Modal, at native 512 px.

Design notes that are not obvious:

* **Architectures are imported, not rewritten.** `flair_sr_core.py` is lifted
  verbatim from the notebook that produced the published runs; its parameter
  counts match the paper's table exactly (188,163 / 926,723 / 1,013,731 /
  241,283). Orchestration lives here so that the science does not move.

* **Resume is two-layered.** A Modal container can vanish mid-run. Every epoch
  writes the checkpoint and the log row to the Volume (fast, survives a retry in
  the same region) *and* pushes them to the Hub (durable, and what makes local
  monitoring possible). On start we prefer the Volume and fall back to the Hub,
  so a cold container in a new region still resumes rather than restarting.

* **Effective batch is held constant by accumulation.** Native 512 px is 4x the
  pixels of the 256 px runs, so the physical batch has to fall on a small card.
  Lowering it outright would change the optimisation schedule and make the runs
  incomparable to each other; accumulation keeps `batch * accum` fixed instead.

* **LPIPS is recomputed here.** It appears in the published results but in none
  of the published notebooks, so its provenance is unverifiable. Recomputing it
  from the released checkpoints makes it reproducible. Values may differ from
  the published ones if those used a different backbone; this uses AlexNet,
  the reference default.

    modal run  modal_app/flair_sr_modal.py::prepare
    modal run  modal_app/flair_sr_modal.py::main --arch edsr --regime s2_to_spot --gpu T4
    modal run --detach modal_app/flair_sr_modal.py::sweep --big-gpu H100
"""
import pathlib

import modal

APP = "flair-sr"
PACK_REPO = "FatimahEmadEldin/flair-guided-sr-512"      # dataset: the v2 pack
OUT_REPO = "FatimahEmadEldin/flair-sr-512-runs"          # model: results land here

app = modal.App(APP)
vol = modal.Volume.from_name("flair-sr", create_if_missing=True)
hf = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy", "pyarrow", "zstandard", "pillow",
        "huggingface_hub", "lpips", "scipy",
    )
    .add_local_file(pathlib.Path(__file__).parent / "flair_sr_core.py",
                    "/root/flair_sr_core.py")
    .add_local_file(pathlib.Path(__file__).parent / "flair_sr_guided.py",
                    "/root/flair_sr_guided.py")
)

VOL = "/vol"
TEST_DOMAINS = ["D040-2021", "D044-2022"]

# 512 px is 4x the pixels of 256. Physical batch drops, effective batch does not.
EFFECTIVE_BATCH = 32
PHYS_BATCH = {       # (target px) -> physical batch that fits ~16 GB in fp16
    64: 32,
    256: 8,
    512: 2,
}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _build_bank(shard_dir, cap, aerial_px, want_guide, tag="data"):
    """Decode the parquet shards into memmaps once, then reuse.

    Decoding 24k 512 px tiles into RAM needs far more than a container has, so
    the bank is disk-backed and RAM stays at one batch. Returns train/test index
    arrays split by DOMAIN, never at random: neighbouring tiles inside one
    French department share acquisition conditions and a random split leaks them.
    """
    import os, io, json, glob
    import numpy as np, pyarrow.parquet as pq, zstandard
    from PIL import Image

    mm = os.path.join(VOL, "bank",
                      f"{tag}_px{aerial_px}_cap{cap}_g{int(want_guide)}")
    meta_p = os.path.join(mm, "meta.json")
    if os.path.exists(meta_p):
        m = json.load(open(meta_p))
        if m.get("complete"):
            print(f"  reusing bank at {mm} ({m['n']} tiles)")
            return m, mm
    os.makedirs(mm, exist_ok=True)

    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    assert files, f"no shards in {shard_dir}"
    DEC = zstandard.ZstdDecompressor()

    hi = np.lib.format.open_memmap(f"{mm}/hi.npy", mode="w+", dtype=np.uint8,
                                   shape=(cap, 3, aerial_px, aerial_px))
    sp = np.lib.format.open_memmap(f"{mm}/sp.npy", mode="w+", dtype=np.float16,
                                   shape=(cap, 3, 64, 64))
    s2 = np.lib.format.open_memmap(f"{mm}/s2.npy", mode="w+", dtype=np.float16,
                                   shape=(cap, 12, 10, 10, 10))
    gd = (np.lib.format.open_memmap(f"{mm}/gd.npy", mode="w+", dtype=np.float16,
                                    shape=(cap, 1, aerial_px, aerial_px))
          if want_guide else None)

    # Per-shard quota, not first-come. Shards are read in sorted order and
    # D004+D006 alone exceed a 6000 cap, so a global cap fills the bank before
    # it ever reaches D040/D044 - the held-out domains - and the split assert
    # below fires on an empty test set. Every domain contributes its share.
    per = max(1, cap // len(files))
    dom, i, skipped = [], 0, 0
    for f in files:
        if i >= cap:
            break
        got = 0
        cols = ["stem", "domain", "rgb", "s2", "s2_shape", "spot", "spot_shape"]
        if want_guide:
            cols += ["ndsm"]
        for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=cols):
            if got >= per:
                break
            d = b.to_pydict()
            for j in range(len(d["stem"])):
                if i >= cap or got >= per:
                    break
                if not d["s2"][j] or not d["spot"][j]:
                    skipped += 1
                    continue
                rgb = np.array(Image.open(io.BytesIO(d["rgb"][j])))
                if rgb.shape[0] != aerial_px:
                    rgb = np.array(Image.fromarray(rgb).resize(
                        (aerial_px, aerial_px), Image.BOX))
                hi[i] = rgb.transpose(2, 0, 1)
                spot = np.frombuffer(DEC.decompress(d["spot"][j]), np.uint16)
                spot = spot.reshape(json.loads(d["spot_shape"][j]))[:3]
                sp[i] = (spot.astype(np.float32) / 10000.).astype(np.float16)
                s2[i] = (np.frombuffer(DEC.decompress(d["s2"][j]), np.uint16)
                         .reshape(json.loads(d["s2_shape"][j]))
                         .astype(np.float32) / 10000.).astype(np.float16)
                if want_guide:
                    nd = d.get("ndsm", [b""])[j]
                    if nd:
                        h = (np.frombuffer(DEC.decompress(nd), np.uint16)
                             .reshape(aerial_px, aerial_px).astype(np.float32) / 100.)
                        gd[i, 0] = (h / 30.).clip(0, 1).astype(np.float16)  # ~30 m full scale
                    else:
                        gd[i, 0] = 0
                dom.append(d["domain"][j])
                i += 1
                got += 1
    n = i
    assert n > 0, "bank is empty - the shards have no paired spot/s2"
    dom = np.array(dom)
    te = np.where(np.isin(dom, TEST_DOMAINS))[0]
    tr = np.where(~np.isin(dom, TEST_DOMAINS))[0]
    assert len(te) and len(tr), (
        f"domain split empty: train {len(tr)} test {len(te)} over domains "
        f"{sorted(set(dom.tolist()))} - the cap starved the held-out domains")
    meta = {"complete": True, "n": int(n), "aerial_px": aerial_px,
            "cap": cap, "guide": bool(want_guide), "skipped": int(skipped),
            "train": tr.tolist(), "test": te.tolist(),
            "domains": sorted(set(dom.tolist()))}
    json.dump(meta, open(meta_p, "w"))
    print(f"  bank built: {n} tiles, train {len(tr)} / test {len(te)}, "
          f"{skipped} skipped for missing spot/s2")
    return meta, mm


# --------------------------------------------------------------------------- #
# Modal functions
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={VOL: vol}, secrets=[hf], timeout=60 * 60 * 3)
def prepare(pack_repo: str = PACK_REPO):
    """Pull the pack into the Volume once. Idempotent; safe to re-run."""
    import os
    from huggingface_hub import snapshot_download
    dst = os.path.join(VOL, "data")
    os.makedirs(dst, exist_ok=True)
    p = snapshot_download(pack_repo, repo_type="dataset", local_dir=dst,
                          allow_patterns=["*.parquet", "*.json"],
                          token=os.environ["HF_TOKEN"])
    vol.commit()
    files = [f for f in os.listdir(dst) if f.endswith(".parquet")]
    tot = sum(os.path.getsize(os.path.join(dst, f)) for f in files)
    print(f"pack in volume: {len(files)} shards, {tot/1e9:.2f} GB at {p}")
    return {"shards": len(files), "gb": round(tot / 1e9, 2)}


def _push(api, repo, local, remote, tries=3, _state={"blocked": False}):
    """Mirror one file to the Hub, and give up on the repo once it says 429.

    A 429 here is a repo-wide rate limit that lasts minutes, not a transient
    collision, so retrying it is not just useless but actively harmful: an
    earlier version of this function backed off 30/60/120/240s per push, which
    would have left a 25-epoch run sleeping longer than it trained.

    So the first 429 trips a breaker for the whole container: every later push
    becomes a no-op. Nothing is lost - the Volume is written every epoch and is
    the source of truth - and `modal run ::push_results` mirrors it afterwards.
    """
    import random, time as _t
    if _state["blocked"]:
        return False
    for k in range(tries):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id=repo, repo_type="model")
            return True
        except Exception as e:
            if "429" in str(e):
                _state["blocked"] = True
                print("  hub rate-limited (429): mirroring disabled for this "
                      "run. The volume has everything; run ::push_results later.")
                return False
            if k == tries - 1:
                print(f"  hub push failed ({type(e).__name__}) - volume has it")
                return False
            _t.sleep((2 ** k) + random.random())
    return False


def _train_impl(arch, regime, epochs, aerial_px, cap, lr, run_id,
                out_repo, guide, push_every=5, data_dir="data",
                seed=1337):
    import os, json, csv, time, math
    import numpy as np, torch, torch.nn.functional as F
    import sys
    sys.path.insert(0, "/root")
    import flair_sr_core as C
    import flair_sr_guided as G
    from huggingface_hub import HfApi

    tok = os.environ["HF_TOKEN"]
    api = HfApi(token=tok)
    api.create_repo(out_repo, repo_type="model", private=True, exist_ok=True)

    meta, mm = _build_bank(os.path.join(VOL, data_dir), cap, aerial_px,
                           guide, tag=data_dir)
    hi = np.load(f"{mm}/hi.npy", mmap_mode="r")
    sp = np.load(f"{mm}/sp.npy", mmap_mode="r")
    s2 = np.load(f"{mm}/s2.npy", mmap_mode="r")
    gd = np.load(f"{mm}/gd.npy", mmap_mode="r") if guide else None
    tr = np.array(meta["train"]); te = np.array(meta["test"])

    out_px = 64 if regime == "s2_to_spot" else aerial_px
    src_kind = "s2" if regime.startswith("s2") else "spot"

    def make_batch(idx, _regime=regime, augment=False):
        idx = np.asarray(idx)
        if src_kind == "s2":
            x = torch.from_numpy(np.ascontiguousarray(s2[idx])).cuda().float()
            lo = (x.median(dim=1).values[:, [2, 1, 0]] / 0.3).clamp(0, 1)
        else:
            lo = (torch.from_numpy(np.ascontiguousarray(sp[idx])).cuda().float()
                  / 0.3).clamp(0, 1)
        if regime == "s2_to_spot":
            y = (torch.from_numpy(np.ascontiguousarray(sp[idx])).cuda().float()
                 / 0.3).clamp(0, 1)
        else:
            y = torch.from_numpy(np.ascontiguousarray(hi[idx])).cuda().float() / 255.
        if y.shape[-1] != out_px:
            y = F.interpolate(y, size=(out_px,) * 2, mode="area")
        g = None
        if guide:
            g = torch.from_numpy(np.ascontiguousarray(gd[idx])).cuda().float()
            if g.shape[-1] != y.shape[-1]:
                g = F.interpolate(g, size=y.shape[-2:], mode="area")
        if augment and torch.rand(1).item() < 0.5:
            lo, y = lo.flip(-1), y.flip(-1)
            if g is not None:
                g = g.flip(-1)
        return (lo, y, g) if guide else (lo, y)

    phys = PHYS_BATCH.get(out_px, 4)
    accum = max(1, EFFECTIVE_BATCH // phys)
    print(f"{arch}/{regime}: target {out_px}px, physical batch {phys} x accum "
          f"{accum} = effective {phys*accum}")

    # Seed everything that draws. Without this the shuffle, the init and the
    # diffusion noise all run off global state, so a run cannot be repeated -
    # which makes "single seed" an understatement and a reproducibility claim
    # unsupportable. The segmentation study used 5 seeds; this at least makes
    # each SR run addressable.
    import random as _r
    _r.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    model, takes_guide = G.build(arch, C, guided=guide)
    model = model.cuda()
    own_loss = hasattr(model, "loss")
    costly = bool(getattr(model, "expensive_eval", False))
    if own_loss:
        print(f"  {arch}: own objective, evaluation deferred to the final epoch")

    if arch == "bicubic":
        # The control has no parameters, so AdamW would raise on an empty list.
        # It also has nothing to learn: evaluate once and write the same test
        # JSON shape as a trained arm, so the tables can treat it as a row.
        lp = None
        try:
            import lpips as _l
            lp = _l.LPIPS(net="alex").cuda()
        except Exception as e:
            print(f"  LPIPS unavailable ({type(e).__name__})")
        final = _eval(model, make_batch, te, C, lp, phys, False)
        final.update(tag=f"bicubic__{regime}", arch="bicubic", regime=regime,
                     params=0, synthetic=False, test_domains=TEST_DOMAINS,
                     aerial_px=aerial_px, epochs=0, guided=False)
        rdir = os.path.join(VOL, "runs", run_id, regime, "logs")
        os.makedirs(rdir, exist_ok=True)
        tp = os.path.join(rdir, f"bicubic__{regime}_test.json")
        json.dump(final, open(tp, "w"), indent=1)
        vol.commit()
        _push(api, out_repo, tp,
              f"{run_id}/{regime}/logs/bicubic__{regime}_test.json")
        print(json.dumps(final, indent=1)[:500])
        return final
    if takes_guide:
        nh = sum(p.numel() for p in model.head_parameters())
        print(f"  guided: +{nh:,} head params, zero-initialised "
              f"(guided == unguided at step 0)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")

    rdir = os.path.join(VOL, "runs", run_id, regime)
    os.makedirs(os.path.join(rdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(rdir, "logs"), exist_ok=True)
    sfx = f"{arch}{'_guided' if guide else ''}"
    if seed != 1337:                     # 1337 keeps the existing filenames
        sfx = f"{sfx}_s{seed}"
    ck = os.path.join(rdir, "checkpoints", f"{sfx}__{regime}_last.pt")
    logp = os.path.join(rdir, "logs", f"{sfx}__{regime}.csv")

    ep0 = 0
    if os.path.exists(ck):                       # volume resume
        st = torch.load(ck, map_location="cuda")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        ep0 = st["epoch"]
        print(f"  resumed from volume at epoch {ep0}")
    else:                                        # hub resume (cold container)
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(out_repo, f"{run_id}/{regime}/checkpoints/"
                                f"{sfx}__{regime}_last.pt", token=tok)
            st = torch.load(p, map_location="cuda")
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            ep0 = st["epoch"]
            print(f"  resumed from hub at epoch {ep0}")
        except Exception as e:
            print(f"  fresh start ({type(e).__name__})")

    lpips_fn = None
    try:
        import lpips as _l
        lpips_fn = _l.LPIPS(net="alex").cuda()
    except Exception as e:
        print(f"  LPIPS unavailable ({type(e).__name__}) - column will be blank")

    for ep in range(ep0 + 1, epochs + 1):
        model.train(); t0 = time.time()
        perm = np.random.permutation(tr); tot = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, len(perm), phys)):
            bt = make_batch(perm[i:i + phys], augment=True)
            lo, y, g = (bt if takes_guide else (bt[0], bt[1], None))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if own_loss:
                    # diffusion optimises noise prediction, not pixel L1
                    loss = model.loss(lo, y) / accum
                else:
                    pred = (model(lo, y.shape[-2:], g) if takes_guide
                            else model(lo, y.shape[-2:]))
                    loss = F.l1_loss(pred, y) / accum
            scaler.scale(loss).backward()
            if (bi + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += float(loss) * accum; nb += 1
        if costly:
            # Sampling is `steps` sequential passes per batch; evaluating every
            # epoch would cost more than the training does. Scored once at the
            # end instead, on the full test set.
            m = {"psnr": float("nan"), "d_psnr_matched": float("nan"),
                 "sharpness_ratio": float("nan")}
        else:
            m = _eval(model, make_batch, te, C, lpips_fn, phys, takes_guide)
        m.update(epoch=ep, l1=tot / max(nb, 1), secs=round(time.time() - t0, 1))

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep}, ck)
        new = not os.path.exists(logp)
        with open(logp, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(m))
            if new:
                w.writeheader()
            w.writerow(m)
        vol.commit()
        # The volume is the durable store and is written every epoch above.
        # The Hub is a mirror: ten jobs committing per-epoch rate-limited the
        # repo (429) and froze the mirror while training ran on fine. Mirror on
        # a cadence, and always on the last epoch.
        if ep % push_every == 0 or ep == epochs:
            for local, remote in (
                    (ck, f"{run_id}/{regime}/checkpoints/{sfx}__{regime}_last.pt"),
                    (logp, f"{run_id}/{regime}/logs/{sfx}__{regime}.csv")):
                _push(api, out_repo, local, remote)
        print(f"  ep {ep}/{epochs} l1 {m['l1']:.4f} psnr {m['psnr']:.2f} "
              f"dmatched {m['d_psnr_matched']:+.2f} sharp {m['sharpness_ratio']:.3f} "
              f"{m['secs']:.0f}s")

    final = _eval(model, make_batch, te, C, lpips_fn, phys, takes_guide)
    final.update(tag=f"{sfx}__{regime}", arch=arch, regime=regime,
                 params=sum(p.numel() for p in model.parameters()),
                 synthetic=False, test_domains=TEST_DOMAINS,
                 aerial_px=aerial_px, epochs=epochs,
                 effective_batch=phys * accum, guided=bool(guide),
                 seed=seed)
    # sfx, not arch: the checkpoint and CSV are seed-suffixed, and a bare
    # arch name here made every seed overwrite the previous one's metrics.
    tp = os.path.join(rdir, "logs", f"{sfx}__{regime}_test.json")
    json.dump(final, open(tp, "w"), indent=1)
    vol.commit()
    _push(api, out_repo, tp,
          f"{run_id}/{regime}/logs/{sfx}__{regime}_test.json")
    print(json.dumps(final, indent=1)[:600])
    return final


def _eval(model, make_batch, idx, C, lpips_fn, bs, takes_guide=False):
    """Metrics against both controls. The affine fit is an ORACLE and is applied
    to BOTH the model and the bicubic control - giving it to one side only would
    let that side calibrate on the answer."""
    import numpy as np, torch, torch.nn.functional as F
    model.eval() if hasattr(model, "eval") else None
    acc = dict(psnr=0., ssim=0., lpips=0., psnr_matched=0., psnr_bicubic=0.,
               psnr_bicubic_raw=0., sharpness_ratio=0.)
    n = 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            bt = make_batch(idx[i:i + bs])
            lo, y, g = (bt if takes_guide else (bt[0], bt[1], None))
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = (model(lo, y.shape[-2:], g) if takes_guide
                        else model(lo, y.shape[-2:])).float().clamp(0, 1)
            bic = F.interpolate(lo, size=y.shape[-2:], mode="bicubic",
                                align_corners=False).clamp(0, 1)
            pm = C.match_radiometry(pred, y)
            bm = C.match_radiometry(bic, y)
            k = len(lo); n += k
            acc["psnr"] += float(C.psnr(pred, y)) * k
            acc["ssim"] += float(C.ssim(pred, y)) * k
            acc["psnr_matched"] += float(C.psnr(pm, y)) * k
            acc["psnr_bicubic"] += float(C.psnr(bm, y)) * k
            acc["psnr_bicubic_raw"] += float(C.psnr(bic, y)) * k
            acc["sharpness_ratio"] += float(C.sharpness(pred) /
                                            max(float(C.sharpness(y)), 1e-8)) * k
            if lpips_fn is not None:
                acc["lpips"] += float(lpips_fn(pred * 2 - 1, y * 2 - 1).mean()) * k
    m = {k: v / max(n, 1) for k, v in acc.items()}
    # An unavailable metric must not report as a perfect score. LPIPS is
    # lower-is-better, so leaving the accumulator at 0 makes a container that
    # failed to import lpips look like the best run in the study - and it is
    # then silently averaged into a mean. NaN propagates instead.
    if lpips_fn is None:
        m["lpips"] = float("nan")
    m["d_psnr_raw"] = m["psnr"] - m["psnr_bicubic_raw"]
    m["d_psnr_matched"] = m["psnr_matched"] - m["psnr_bicubic"]
    m["d_psnr"] = m["psnr"] - m["psnr_bicubic"]
    m["n_eval"] = n
    return m


@app.function(image=image, gpu="T4", volumes={VOL: vol}, secrets=[hf],
              timeout=60 * 60 * 24, retries=modal.Retries(max_retries=2))
def train(**kw):
    return _train_impl(**kw)


# Modal validates these server-side, so the names are the ones its docs list.
GPU_ALIAS = {"T4": "T4", "L4": "L4", "A10G": "A10G", "A100": "A100-40GB",
             "A100-80GB": "A100-80GB", "L40S": "L40S", "H100": "H100",
             "H200": "H200", "B200": "B200"}


def _on(gpu):
    if gpu not in GPU_ALIAS:
        raise SystemExit(f"unknown gpu {gpu!r}; pick one of {sorted(GPU_ALIAS)}")
    return train.with_options(gpu=GPU_ALIAS[gpu])


@app.local_entrypoint()
def main(arch: str = "edsr", regime: str = "s2_to_spot", gpu: str = "T4",
         epochs: int = 25, aerial_px: int = 512, cap: int = 6000,
         lr: float = 2e-4, run_id: str = "v2_512", out_repo: str = OUT_REPO,
         guide: bool = False, push_every: int = 5,
         data_dir: str = "data", seed: int = 1337):
    r = _on(gpu).remote(arch=arch, regime=regime, epochs=epochs,
                        aerial_px=aerial_px, cap=cap, lr=lr, run_id=run_id,
                        out_repo=out_repo, guide=guide,
                        push_every=push_every, data_dir=data_dir,
                        seed=seed)
    print(r)


@app.local_entrypoint()
def sweep(run_id: str = "v2_512", out_repo: str = OUT_REPO,
          cheap_only: bool = False, big_gpu: str = "H100",
          epochs: int = 25, guided: bool = False, push_every: int = 5,
          data_dir: str = "data", archs: str = "", seeds: str = "1337",
          regimes: str = "", batch: int = 8):
    """Fan out. s2_to_spot is seconds/epoch so it goes on a T4; the 512 px aerial
    regimes go on an A100. RCAN on aerial is excluded by default: it measured
    1037-1104 s/epoch at 256 px, so ~30 h per run at 512 - about half the credit
    for the architecture that scored worst everywhere it finished."""
    sel = [a.strip() for a in archs.split(",") if a.strip()]
    keep = [x.strip() for x in regimes.split(",") if x.strip()]
    jobs = []
    if not guided and (not keep or "s2_to_spot" in keep):
        for a in (sel or ["srcnn", "edsr", "rcan", "swinir"]):
            jobs.append((a, "s2_to_spot", "T4"))
    if not cheap_only:
        for a in (sel or ["srcnn", "edsr", "swinir"]):
            for r in ["spot_to_aerial", "s2_to_aerial"]:
                if keep and r not in keep:
                    continue
                jobs.append((a, r, big_gpu))
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    plan = [(a, r, g, k) for a, r, g in jobs for k in sd]
    print(f"launching {len(jobs)} runs x {len(sd)} seeds = {len(plan)}")

    # Batch under the workspace GPU cap and BLOCK on each wave.
    #
    # spawn() is fire-and-forget and the entrypoint returning tears the
    # ephemeral app down with it: a previous 25-run sweep had 20 calls silently
    # discarded because they had not started before the entrypoint exited. They
    # were not queued and they were not billed - they simply never happened.
    # Waiting on each wave keeps the app alive until its work is really done.
    done = []
    for w in range(0, len(plan), batch):
        wave = plan[w:w + batch]
        print("")
        print(f"-- wave {w // batch + 1}: {len(wave)} runs")
        handles = [_on(g).spawn(arch=a, regime=r, epochs=epochs, aerial_px=512,
                                cap=6000, lr=2e-4, run_id=run_id,
                                out_repo=out_repo, guide=guided,
                                push_every=push_every, data_dir=data_dir,
                                seed=k)
                   for a, r, g, k in wave]
        for (a, r, g, k), h in zip(wave, handles):
            print(f"   {a:7} {r:16} {g:6} seed {k:5} -> {h.object_id}")
        for (a, r, g, k), h in zip(wave, handles):
            try:
                h.get()
                done.append((a, r, k))
            except Exception as e:
                print(f"   FAILED {a}/{r}/seed {k}: {type(e).__name__}")
    print("")
    print(f"completed {len(done)}/{len(plan)}")
    print("\nwatch: modal app logs flair-sr   |   results stream to "
          f"https://huggingface.co/{out_repo}")


@app.function(image=image, volumes={VOL: vol}, secrets=[hf], timeout=60 * 60 * 2)
def sync(run_id: str = "v2_512", out_repo: str = OUT_REPO, logs_only: bool = True):
    """Mirror the volume's run artefacts to the Hub in one pass.

    The volume is the source of truth: it is written every epoch and never
    rate-limited. This exists so a run whose mirror was throttled can be
    published afterwards, rather than the results living only inside Modal.
    """
    import os, glob
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(out_repo, repo_type="model", private=True, exist_ok=True)
    root = os.path.join(VOL, "runs", run_id)
    pats = ["logs/*.csv", "logs/*_test.json"] + ([] if logs_only
                                                 else ["checkpoints/*.pt"])
    sent = failed = 0
    for regime in sorted(os.listdir(root)):
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(root, regime, pat))):
                rel = f"{run_id}/{regime}/{pat.split('/')[0]}/{os.path.basename(f)}"
                if _push_patient(api, out_repo, f, rel):
                    sent += 1
                else:
                    failed += 1
    print(f"synced {sent} files, {failed} failed -> "
          f"https://huggingface.co/{out_repo}")
    return {"sent": sent, "failed": failed}


@app.local_entrypoint()
def push_results(run_id: str = "v2_512", out_repo: str = OUT_REPO,
                 checkpoints: bool = False):
    print(sync.remote(run_id=run_id, out_repo=out_repo,
                      logs_only=not checkpoints))


def _push_patient(api, repo, local, remote, tries=6):
    """Sync-only upload: waits out a 429 instead of tripping the breaker.

    The breaker in _push exists because a training container must not sleep.
    This runs alone, after the fact, and waiting is exactly what it should do.
    """
    import random, time as _t
    for k in range(tries):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id=repo, repo_type="model")
            return True
        except Exception as e:
            if k == tries - 1:
                print(f"  sync failed for {remote} ({type(e).__name__})")
                return False
            _t.sleep((20 if "429" in str(e) else 1) * (2 ** k) + random.random() * 3)
    return False
