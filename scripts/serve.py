#!/usr/bin/env python3
"""Launch a llama-server for a model defined in models.json.

    python scripts/serve.py qwen3.6            # run in foreground
    LLAMA_SERVER_BIN=/path/to/llama-server python scripts/serve.py ornith:9b

Config lives in models.json (label -> gguf, port, extra llama-server args). Tuned
for an 8GB GPU: ornith:9b fits fully in VRAM; qwen3.6 offloads MoE experts to RAM
via --n-cpu-moe (see hardware notes). Also importable: load_models(),
launch(label) -> (Popen, base_url), wait_health(base_url).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get("LLAMA_SERVER_BIN",
                     str(Path.home() / "llama.cpp-src" / "build" / "bin" / "llama-server"))


def load_models(path=ROOT / "models.json"):
    return json.loads(Path(path).read_text())


def build_cmd(label, cfg):
    gguf = os.path.expanduser(cfg["gguf"])
    return [BIN, "-m", gguf, "-a", label, "--host", "127.0.0.1",
            "--port", str(cfg["port"])] + cfg.get("args", [])


def launch(label, cfg=None, log=None):
    """Start llama-server in the background; return (Popen, base_url)."""
    cfg = cfg or load_models()[label]
    out = open(log, "w") if log else subprocess.DEVNULL
    p = subprocess.Popen(build_cmd(label, cfg), stdout=out, stderr=out)
    return p, f"http://127.0.0.1:{cfg['port']}"


def wait_health(base_url, timeout=400):
    """Block until /health returns 200, or timeout; return True/False."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    models = load_models()
    if len(sys.argv) != 2 or sys.argv[1] not in models:
        sys.exit(f"usage: serve.py <label>  (labels: {', '.join(models)})")
    os.execv(BIN, build_cmd(sys.argv[1], models[sys.argv[1]]))  # foreground


if __name__ == "__main__":
    main()
