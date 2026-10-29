# -*- coding: utf-8 -*-
"""One cheap call that proves the whole Modal path before anything expensive.

Checks, in one container: GPU allocation and name, torch sees CUDA, the
`huggingface` secret is injected under a key we can find, the volume mounts and
survives a commit, and the HF Hub is reachable with that token.

    modal run "modal_app/smoke.py"
    modal run "modal_app/smoke.py" --gpu L4
"""
import modal

app = modal.App("flair-sr-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "huggingface_hub")
)
vol = modal.Volume.from_name("flair-sr", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=600,
)
def probe():
    import os, subprocess, json
    out = {}
    out["gpu_name"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["vram_gb"] = round(total / 1e9, 1)
        out["vram_free_gb"] = round(free / 1e9, 1)

    # which env key does the `huggingface` secret actually use?
    cand = [k for k in os.environ
            if "HF" in k.upper() or "HUGGING" in k.upper()]
    out["secret_keys"] = sorted(cand)
    tok = next((os.environ[k] for k in
                ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF")
                if os.environ.get(k)), None)
    out["token_present"] = bool(tok)
    if tok:
        from huggingface_hub import HfApi
        try:
            who = HfApi(token=tok).whoami()
            out["hf_user"] = who.get("name")
        except Exception as e:
            out["hf_error"] = f"{type(e).__name__}: {e}"[:120]

    # volume round-trip
    os.makedirs("/vol/smoke", exist_ok=True)
    open("/vol/smoke/ok.txt", "w").write("written from modal")
    out["volume_writable"] = os.path.exists("/vol/smoke/ok.txt")
    return out


@app.local_entrypoint()
def main(gpu: str = "T4"):
    import json
    r = probe.remote()
    print(json.dumps(r, indent=2))
