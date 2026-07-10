#!/usr/bin/env python3
"""Minimal Claude-Code-style local agent. Fully local, closed loop, stdlib only.

Ollama backend (native /api/chat). Tools: run_bash, read_file, write_file.
Memory via memory.py: recall before each turn, extract after.

    ollama pull ornith:9b
    python miniagent.py
"""

import argparse
import json
import os
import subprocess
import urllib.request
import uuid
from pathlib import Path

import memory

MODEL = os.environ.get("MINIAGENT_MODEL", "ornith:9b")
BASE_URL = os.environ.get("MINIAGENT_BASE_URL", "http://localhost:11434")
NUM_CTX = int(os.environ.get("MINIAGENT_NUM_CTX", "32768"))  # context window; Ollama defaults to 4096
MAX_HISTORY = 20      # live window; durable content persists in memory
MAX_TOOL_ROUNDS = int(os.environ.get("MINIAGENT_MAX_ROUNDS", "10"))  # cap tool loops so a misbehaving model can't spin forever
AUTO_CONFIRM = os.environ.get("MINIAGENT_AUTO_CONFIRM") == "1"       # skip the run_bash prompt (non-interactive use)
THINK = os.environ.get("MINIAGENT_THINK", "1") != "0"               # off → tell the model to skip chain-of-thought

SYSTEM = """You are a local assistant that handles small tasks directly.
Prefer acting over explaining. Use tools to do the work; don't narrate first.
Tools: run_bash, read_file, write_file. Keep answers terse.
Durable facts are remembered automatically across sessions; don't manage memory yourself.

--- RELEVANT MEMORIES ---
{memblock}
--- END MEMORIES ---
"""


def llm(messages, tools=None, temperature=0.2, max_tokens=None):
    """POST to the chat endpoint; return the assistant message dict."""
    opts = {"temperature": temperature, "num_ctx": NUM_CTX}
    if max_tokens:
        opts["num_predict"] = max_tokens                # native's name for the output cap
    body = {"model": MODEL, "messages": messages, "stream": False, "options": opts}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]


def run_bash(command: str) -> str:
    print(f"\n  \033[33m$ {command}\033[0m")
    if not AUTO_CONFIRM and input("  run this? [y/N] ").strip().lower() != "y":
        return "User declined to run the command."
    try:
        p = subprocess.run(command, shell=True, capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "error: command timed out after 120s"
    return ((p.stdout or "") + (p.stderr or "")).strip()[:8000] \
        or f"(exit {p.returncode}, no output)"


def read_file(path: str) -> str:
    try:
        return Path(path).expanduser().read_text()[:12000]
    except Exception as e:
        return f"error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        p = Path(path).expanduser()
        if p.exists():                                 # creating is safe; clobbering needs consent
            print(f"\n  \033[33moverwrite {p} ({len(content)} bytes)\033[0m")
            if not AUTO_CONFIRM and input("  overwrite? [y/N] ").strip().lower() != "y":
                return "User declined to overwrite the file."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} bytes to {p}"
    except Exception as e:
        return f"error: {e}"


TOOLS = {"run_bash": run_bash, "read_file": read_file, "write_file": write_file}

SCHEMAS = [
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Run a shell command on the local machine. Output is returned.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file and return its contents (truncated).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write (overwrite) a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
]


def system_msg(user_text: str) -> dict:
    """Build the system message: recalled memories, plus the think toggle for this run."""
    content = SYSTEM.format(memblock=memory.format_block(memory.recall(user_text)))
    if not THINK:
        content += "\n/no_think"                       # Qwen3 convention; harmless to models without it
    return {"role": "system", "content": content}


def turn(system: dict, history: list):
    """Run one user turn to completion; return (reply, updated_history)."""
    msgs = [system] + history
    for _ in range(MAX_TOOL_ROUNDS):
        m = llm(msgs, tools=SCHEMAS)
        calls = m.get("tool_calls")
        if not calls:
            msgs.append(m)
            return m.get("content") or "", msgs[1:]
        msgs.append(m)
        for call in calls:
            cid = call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            call["id"] = cid                           # sync assistant msg + tool reply
            name = call["function"]["name"]
            fn = TOOLS.get(name)
            raw = call["function"].get("arguments") or {}
            args = json.loads(raw) if isinstance(raw, str) else raw  # native sends an object

            result = fn(**args) if fn else f"unknown tool: {name}"
            msgs.append({"role": "tool", "tool_call_id": cid, "content": result})
    return "(stopped: tool-call limit reached)", msgs[1:]  # ends on a complete group


def main():
    ap = argparse.ArgumentParser(description="Minimal local agent.")
    ap.add_argument("--prompt", help="run one prompt non-interactively, then exit")
    args = ap.parse_args()

    if args.prompt:                                    # one-shot: run once, then exit
        reply, _ = turn(system_msg(args.prompt),
                        [{"role": "user", "content": args.prompt}])
        print(reply)
        memory.extract_and_store(llm, args.prompt, reply)
        return

    print(f"miniagent · {MODEL} · type 'exit' to quit\n")
    history = []
    while True:
        try:
            user = input("\033[36myou ›\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        system = system_msg(user)                      # recall + think toggle
        history.append({"role": "user", "content": user})

        try:
            reply, history = turn(system, history)
        except Exception as e:
            print(f"\033[31m! {e}\033[0m  (is `ollama serve` running?)\n")
            history.pop()
            continue
        history = history[-MAX_HISTORY:]              # trim live window...
        while history and history[0].get("role") == "tool":
            history.pop(0)                            # ...never lead with an orphaned tool reply
        print(f"\033[32m›\033[0m {reply}\n")

        memory.extract_and_store(llm, user, reply)     # capture after


if __name__ == "__main__":
    main()