# Status

"Implemented" requires cited evidence; queue items carry the measurement that
proves them done. All LLM-dependent results are model-specific — here
ornith 1.0 9B (`ornith:9b`, think-on); only the unit tests are model-independent.

## Implemented

| Capability | Evidence |
|---|---|
| Agent loop: Ollama tool-calling, tool-round cap, history trim | `minibench.py` pass@1 0.83, pass@k 1.0 (ornith:9b think-on, 2026-07) |
| Open-ended skills (refactor, diagnose, review, document, test-design) | `judge.py` bundle 5/5, scored by a separate strong model |
| Tool gating: `run_bash` always confirms, `write_file` on overwrite, fails closed without stdin | `tests/test_miniagent.py` |
| Memory: BM25 recall, type weights, recency decay with floor, dedup | `tests/test_memory.py` |
| Belief revision: contradicted same-type preference/decision retired | unit tests + membench supersede 2/2 |
| Scoping: facts per working dir, preferences global, pre-scope DBs migrate | unit tests |
| Cross-session extract → recall → use, robust to 40 distractors | membench verbatim/noise 2/2 |
| Bench isolation: attempts never touch the real memory DB | scratch `MINIAGENT_MEMORY_DB` per attempt |

## Measured limits

- Paraphrase recall: extracted 2/2 but recalled 0/2 — BM25 needs lexical overlap.
- Cold-start supersede: BM25 ≈ 0 below ~5 memories, so nothing retires in a fresh DB.
- Supersede matches similarity, not contradiction; false-positive rate untested.
- `recall()` writes (`hits`/`accessed`): analyze real DBs on a copy.
- membench `used` stage is model-stochastic: trust only at `--n 3`+.
- No pruning: stale facts decay to a floor but live forever.

## Improvement queue

| # | Improvement | Acceptance gate |
|---|---|---|
| 1 | Semantic recall: fill reserved `emb` column, merge cosine with BM25 (touch only `store()`/`recall()`) | membench paraphrase recalled 0/2 → ≥2/3, no regression elsewhere |
| 2 | Contradiction-aware supersede: log retirements, then tighten the trigger | similar-but-compatible preferences both survive; contradictions still retire |
| 3 | Cold-start supersede: corpus-size-aware threshold | contradiction retires in an otherwise-empty DB |
| 4 | Pruning of superseded/floor-decayed rows | deferred until a real DB nears ~10k rows |
