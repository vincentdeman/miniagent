#!/usr/bin/env python3
"""Compare generation throughput across llama-server models.

Brings up each model one at a time (they don't co-fit in 8GB VRAM), warms it,
then times several fixed-length generations and reports median tokens/sec from
the server's own timings (wall-clock fallback). Speed only; correctness is
minibench.py's job. Prints the token count so a short (early-stop) run is visible.

    python benchmarks/speedbench.py --models ornith:9b,qwen3.6 --tokens 512 --trials 3
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import serve  # noqa: E402  launch, wait_health, load_models

PROMPT = ("Write a thorough, technical explanation of how a transformer neural network "
          "processes a sequence: tokenization, self-attention, and the feed-forward "
          "layers. Be detailed and precise.")


def one(base_url, model, max_tokens):
    """One generation; return (gen_tokens_per_sec, n_generated). Prefer server timings."""
    body = json.dumps({"model": model, "temperature": 0.3, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    wall = time.monotonic() - t0
    n = d["usage"]["completion_tokens"]
    tps = d.get("timings", {}).get("predicted_per_second") or (n / wall if wall else 0)
    return tps, n


def bench(label, cfg, tokens, trials):
    log = f"/tmp/speedbench-{cfg['port']}.log"
    proc, url = serve.launch(label, cfg, log=log)
    try:
        if not serve.wait_health(url):
            print(f"{label}: server failed to start (see {log})")
            return None
        one(url, label, 64)                       # warmup: page cache + CUDA context
        tps, ns = [], []
        for _ in range(trials):
            t, n = one(url, label, tokens)
            tps.append(t)
            ns.append(n)
        return {"tps": median(tps), "tps_all": tps, "n": int(median(ns))}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)                  # ensure VRAM/port free before next model
        except Exception:
            proc.kill()


def ctx_of(cfg):
    a = cfg.get("args", [])
    return a[a.index("-c") + 1] if "-c" in a else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    models = serve.load_models()
    print(f"tokens={a.tokens}  trials={a.trials}\n")
    print(f"{'model':16} {'ctx':>8} {'gen_tok/s':>10} {'tokens':>7}   per-trial")
    for label in [m.strip() for m in a.models.split(",") if m.strip()]:
        r = bench(label, models[label], a.tokens, a.trials)
        if r:
            per = ", ".join(f"{x:.1f}" for x in r["tps_all"])
            print(f"{label:16} {ctx_of(models[label]):>8} {r['tps']:>10.1f} {r['n']:>7}   [{per}]")


if __name__ == "__main__":
    main()
