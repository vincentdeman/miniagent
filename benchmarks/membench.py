"""Memory benchmark for miniagent: plant facts in one session, probe in a fresh one.

Each scenario runs the agent once (or twice) to plant facts, then probes with a
separate process sharing only the memory DB — the answer can only come from memory.
Three checks per attempt attribute failures to a pipeline stage:

  extracted  the fact landed in the DB (extraction LLM call + JSON parsing)
  recalled   memory.recall(probe) surfaces it (deterministic: BM25 + weights)
  used       the probe reply states the expected answer (model trusts its memories)

`noisy` scenarios pre-seed 40 background memories: recall must rank, and BM25 gets
the corpus it needs for supersede to fire (relevance ~0 in a near-empty DB).

    python benchmarks/membench.py --models ornith:9b --n 3 --think --csv benchmarks/membench_results.csv
"""

import argparse, os, signal, sqlite3, subprocess, sys, tempfile, shutil
from pathlib import Path

from minibench import AGENT, PY

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import memory

MAX_ROUNDS = "6"   # plant/probe turns are plain Q&A; no long tool loops needed

# (id, plant prompts, probe prompt, expect in reply, forbid in reply, noisy)
SCENARIOS = [
    # baseline: probe shares the planted wording, BM25 match is easy
    ("verbatim",
     ["Remember this for later: the deploy bucket is named zx-flumen-47."],
     "What is the name of the deploy bucket?",
     ["zx-flumen-47"], [], False),

    # probe shares no content words with the plant — measures the lexical-recall
    # ceiling (the embeddings upgrade memory.py reserves space for)
    ("paraphrase",
     ["Remember this for later: the API rate limit is 4400 requests per minute."],
     "How hard can we hammer the external endpoint before throttling kicks in?",
     ["4400"], [], False),

    # recall has to rank against 40 background memories, not just find one row
    ("noise",
     ["Remember this for later: the staging gateway hostname is kumquat-lantern.internal."],
     "What is the staging gateway hostname?",
     ["kumquat-lantern"], [], True),

    # belief revision across sessions: only the corrected value should survive
    ("supersede",
     ["Remember this for later: we deploy to region eu-west-1.",
      "Correction, remember this instead: we now deploy to region ap-south-2, not eu-west-1."],
     "Which region do we deploy to?",
     ["ap-south-2"], ["eu-west-1"], True),
]


def _adopt(db):
    """Point the in-process memory module at an attempt's DB, matching the
    scope the agent subprocess uses (its cwd, i.e. the work dir)."""
    memory.DB = Path(db)
    memory.SCOPE = os.path.realpath(memory.DB.parent.parent)  # agent's getcwd() is resolved


def seed_distractors(db, n=40):
    _adopt(db)
    for i in range(n):
        memory.store(f"service worker-{i} listens on port {7000 + i} and logs to /var/log/w{i}.log")


def db_contains(db, token):
    """True if a live memory mentions `token` (extraction stage evidence)."""
    if not os.path.exists(db):
        return False
    c = sqlite3.connect(db)
    try:
        n = c.execute("SELECT COUNT(*) FROM memories WHERE superseded=0 AND text LIKE ?",
                      (f"%{token}%",)).fetchone()[0]
    except sqlite3.OperationalError:                   # agent never created the schema
        n = 0
    c.close()
    return n > 0


def recalled(db, probe, expect):
    """True if recall(probe) — exactly what system_msg runs — surfaces every token."""
    _adopt(db)
    mems = memory.recall(probe)
    return all(any(tok in txt for _, txt in mems) for tok in expect)


def run_capture(work, model, prompt, think, timeout, db):
    """Run the agent once and capture stdout; return (output, timed_out).

    Own session so a timeout kills grandchildren; SIGKILL closes their pipe ends,
    letting the draining communicate() return.
    """
    env = {**os.environ, "MINIAGENT_MODEL": model, "MINIAGENT_THINK": str(int(think)),
           "MINIAGENT_AUTO_CONFIRM": "1", "MINIAGENT_MAX_ROUNDS": MAX_ROUNDS,
           "MINIAGENT_MEMORY_DB": db}
    p = subprocess.Popen([PY, AGENT, "--prompt", prompt], cwd=work, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return out or "", False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return out or "", True


def attempt(model, scen, think, timeout):
    """Run one scenario end to end; return (extracted, recalled, used, timed_out)."""
    sid, plants, probe, expect, forbid, noisy = scen
    work = tempfile.mkdtemp()
    db = os.path.join(work, ".miniagent", "memory.db")
    if noisy:
        seed_distractors(db)
    timed_out = False
    for p in plants:
        _, to = run_capture(work, model, p, think, timeout, db)
        timed_out |= to
    ext = all(db_contains(db, tok) for tok in expect)
    reply, to = run_capture(work, model, probe, think, timeout, db)
    timed_out |= to
    rec = recalled(db, probe, expect)                  # after probe: don't perturb its recall
    used = all(t in reply for t in expect) and not any(t in reply for t in forbid)
    shutil.rmtree(work, ignore_errors=True)
    return ext, rec, used, timed_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--only", help="comma-separated scenario ids to run")
    ap.add_argument("--timeout", type=int, default=300, help="per agent invocation")
    ap.add_argument("--csv", help="append per-attempt rows to this CSV file")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    want = {s.strip() for s in a.only.split(",")} if a.only else None
    scens = [s for s in SCENARIOS if s[0] in want] if want else SCENARIOS
    think = int(a.think)
    print(f"thinking={'on' if a.think else 'off'}  n={a.n}  scenarios={len(scens)}  models={','.join(models)}\n")

    rows = []                                          # (model, scenario, ext, rec, used, timed_out)
    for m in models:
        for scen in scens:
            for _ in range(a.n):
                ext, rec, used, to = attempt(m, scen, a.think, a.timeout)
                rows.append((m, scen[0], ext, rec, used, to))
                stages = f"extracted={'y' if ext else 'n'} recalled={'y' if rec else 'n'} used={'y' if used else 'n'}"
                tag = "TIMEOUT " if to else ""
                print(f"{m:16} {scen[0]:12} {tag}{stages}")

    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a") as f:
            if new: f.write("model,scenario,think,extracted,recalled,used,timed_out\n")
            for m, sid, ext, rec, used, to in rows:
                f.write(f"{m},{sid},{think},{int(ext)},{int(rec)},{int(used)},{int(to)}\n")

    # stage funnel per model x scenario: where does the pipeline lose the fact?
    print(f"\n{'model':16} {'scenario':12} {'extracted':>9} {'recalled':>9} {'used':>9}")
    for m in models:
        for sid, *_ in scens:
            v = [(e, r, u) for mm, s, e, r, u, _ in rows if mm == m and s == sid]
            e, r, u = (sum(col) for col in zip(*v))
            n = len(v)
            print(f"{m:16} {sid:12} {e:>7}/{n} {r:>7}/{n} {u:>7}/{n}")


if __name__ == "__main__":
    main()
