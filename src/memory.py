#!/usr/bin/env python3
"""Persistent memory for miniagent. Stdlib only: SQLite FTS5 + BM25, no network.

API: store, recall, format_block, extract_and_store.
One shared DB; memories are scoped to the working directory they were made in,
except GLOBAL_TYPES (user preferences), which surface everywhere. recall() sees
the current scope plus globals.
To add semantic recall later, touch only store() (fill `emb`) and recall()
(cosine over `emb`, or merge with BM25); schema and API stay put. Get embeddings
from the backend already running — llama.cpp `/v1/embeddings` or Ollama
`/api/embeddings` with a small embed model — no separate server needed.
"""

import json
import math
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

DB = Path(os.environ.get("MINIAGENT_MEMORY_DB",
                         Path(__file__).resolve().parent.parent / ".miniagent" / "memory.db"))  # project root; override to relocate
SCOPE = os.environ.get("MINIAGENT_SCOPE") or os.getcwd()  # project = where the agent runs
DEBUG = bool(os.environ.get("MINIAGENT_DEBUG"))

# tunables
TYPE_WEIGHT = {"preference": 1.2, "error_pattern": 1.15, "decision": 1.1, "fact": 1.0}
NO_DECAY = {"preference", "error_pattern"}   # exempt from recency decay
GLOBAL_TYPES = {"preference"}                # stored scope='global': visible from every project
HALFLIFE_DAYS = 30.0
DECAY_FLOOR = 0.35
FTS_CANDIDATES = 4        # fetch k*this from FTS, then re-rank
SUPERSEDE_REL = 6.0       # min BM25 relevance to retire a same-type belief; corpus-dependent, tune
MIN_TERM_LEN = 3          # drop shorter query tokens


def _conn():
    DB.parent.mkdir(parents=True, exist_ok=True)       # lazy: honors post-import DB reassignment
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS memories(
        id INTEGER PRIMARY KEY,
        type TEXT NOT NULL,
        text TEXT NOT NULL,
        norm TEXT NOT NULL,
        scope TEXT NOT NULL DEFAULT 'global',
        emb BLOB,                          -- reserved: embeddings upgrade
        created REAL NOT NULL,
        accessed REAL NOT NULL,
        hits INTEGER NOT NULL DEFAULT 0,
        superseded INTEGER NOT NULL DEFAULT 0)""")
    try:                                   # pre-scope DBs: migrate, old rows become global
        c.execute("ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'global'")
    except sqlite3.OperationalError:
        pass
    c.execute("CREATE INDEX IF NOT EXISTS idx_norm ON memories(norm)")
    c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(text)")
    return c


def _norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def _fts_query(text):
    """OR of quoted terms; None if nothing usable."""
    terms = [t for t in re.findall(r"\w+", text.lower()) if len(t) >= MIN_TERM_LEN][:20]
    return " OR ".join(f'"{t}"' for t in terms) if terms else None


def store(text, mtype="fact"):
    text = text.strip()
    if not text:
        return
    mtype = mtype if mtype in TYPE_WEIGHT else "fact"
    scope = "global" if mtype in GLOBAL_TYPES else SCOPE
    norm = _norm(text)
    now = time.time()
    c = _conn()

    dup = c.execute("SELECT id FROM memories WHERE norm=? AND superseded=0 "
                    "AND scope IN (?, 'global')", (norm, SCOPE)).fetchone()
    if dup:                                            # exact restatement -> touch
        c.execute("UPDATE memories SET accessed=?, hits=hits+1 WHERE id=?", (now, dup[0]))
        c.commit()
        c.close()
        return

    if mtype in ("preference", "decision"):            # retire a contradicted belief
        fq = _fts_query(text)
        if fq:
            cand = c.execute(
                "SELECT m.id, bm25(mem_fts) FROM mem_fts JOIN memories m "
                "ON m.id=mem_fts.rowid WHERE mem_fts MATCH ? AND m.type=? "
                "AND m.superseded=0 AND m.scope IN (?, 'global') "
                "ORDER BY bm25(mem_fts) LIMIT 1",
                (fq, mtype, SCOPE)).fetchone()
            if cand and -cand[1] >= SUPERSEDE_REL:
                c.execute("UPDATE memories SET superseded=1 WHERE id=?", (cand[0],))

    mid = c.execute("INSERT INTO memories(type,text,norm,scope,created,accessed) "
                    "VALUES(?,?,?,?,?,?)", (mtype, text, norm, scope, now, now)).lastrowid
    c.execute("INSERT INTO mem_fts(rowid, text) VALUES(?,?)", (mid, text))
    c.commit()
    c.close()


def recall(query, k=6):
    fq = _fts_query(query)
    if not fq:
        return []
    now = time.time()
    c = _conn()
    rows = c.execute(
        "SELECT m.id, m.type, m.text, m.accessed, bm25(mem_fts) FROM mem_fts "
        "JOIN memories m ON m.id=mem_fts.rowid WHERE mem_fts MATCH ? "
        "AND m.superseded=0 AND m.scope IN (?, 'global') "
        "ORDER BY bm25(mem_fts) LIMIT ?",
        (fq, SCOPE, k * FTS_CANDIDATES)).fetchall()

    scored = []
    for mid, mtype, text, accessed, bm in rows:
        rel = -bm                                      # higher = better
        if mtype in NO_DECAY:
            recency = 1.0
        else:
            age_days = (now - accessed) / 86400.0
            recency = max(DECAY_FLOOR, math.exp(-age_days / HALFLIFE_DAYS))
        scored.append((rel * recency * TYPE_WEIGHT.get(mtype, 1.0), mid, mtype, text))

    scored.sort(reverse=True)
    top = scored[:k]
    if top:                                            # bump access on what we surfaced
        ids = [t[1] for t in top]
        c.execute(f"UPDATE memories SET accessed=?, hits=hits+1 "
                  f"WHERE id IN ({','.join('?' * len(ids))})", (now, *ids))
        c.commit()
    c.close()
    return [(t[2], t[3]) for t in top]                 # (type, text)


def format_block(mems):
    return "\n".join(f"- [{t}] {txt}" for t, txt in mems) if mems else "(none yet)"


_EXTRACT_SYS = (
    "Extract durable, reusable facts from the exchange. Return ONLY a JSON array, "
    "no prose, no code fences. Each item: {\"type\": one of "
    "[\"preference\",\"decision\",\"fact\",\"error_pattern\"], "
    "\"text\": one short atomic statement}. Keep only things worth remembering across "
    "sessions: stable preferences, decisions made, durable facts, recurring errors and "
    "their fixes. Skip ephemeral chatter. Return [] if nothing is durable."
)


def extract_and_store(chat, user_msg, assistant_msg):
    """Best-effort; never raises into the loop. `chat` returns an assistant msg dict."""
    try:
        m = chat([{"role": "system", "content": _EXTRACT_SYS},
                  {"role": "user",
                   "content": f"USER: {user_msg}\nASSISTANT: {assistant_msg}"}],
                 temperature=0.0, max_tokens=400)
        raw = (m.get("content") or "").strip()
        lo, hi = raw.find("["), raw.rfind("]")
        items = json.loads(raw[lo:hi + 1]) if lo != -1 and hi != -1 else []
        if DEBUG and not items:
            print(f"[memory] nothing extracted from: {raw[:200]!r}", file=sys.stderr)
        for it in items:
            if isinstance(it, dict) and it.get("text"):
                store(it["text"], it.get("type", "fact"))
    except Exception as e:
        if DEBUG:
            print(f"[memory] extraction error: {e}", file=sys.stderr)