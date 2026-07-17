# Status

"Implemented" requires cited evidence; queue items carry the measurement that
proves them done. All LLM-dependent results are model-specific. The default model
is now Qwen3.6-35B-A3B (results below); the tables here are the `ornith:9b`
(think-on) baseline, which remains the lighter dense alternative. Only the unit
tests are model-independent.

## Implemented

| Capability | Evidence |
|---|---|
| Agent loop: tool-calling, tool-round cap, history trim | `minibench.py` pass@1 0.83, pass@k 1.0 (ornith:9b think-on, 2026-07) |
| Open-ended skills (refactor, diagnose, review, document, test-design) | `judge.py` bundle 5/5, scored by a separate strong model |
| Tool gating: `run_bash` always confirms, `write_file` on overwrite, fails closed without stdin | `tests/test_miniagent.py` |
| Memory: BM25 recall, type weights, recency decay with floor, dedup | `tests/test_memory.py` |
| Belief revision: contradicted same-type preference/decision retired | unit tests + membench supersede 2/2 |
| Scoping: facts per working dir, preferences global, pre-scope DBs migrate | unit tests |
| Cross-session extract → recall → use, robust to 40 distractors | membench verbatim/noise 2/2 |
| Bench isolation: attempts never touch the real memory DB | scratch `MINIAGENT_MEMORY_DB` per attempt |

## Second model: Qwen3.6-35B-A3B (MoE) — 2026-07-17

35B-A3B (3B active) run via the CUDA llama-server backend, experts offloaded to
RAM on the 8GB GPU (`--n-cpu-moe 35`, 128K ctx, think-on; see `models.json`).

| Capability | Evidence |
|---|---|
| Agent loop / minibench | pass@1 0.86, pass@k 0.92, p50 38.8s (n=3, 12 tasks). Only miss: `regex-engine` 0/3 all TIMEOUT@300s — latency, not capability (`json-roundtrip`, comparable difficulty, 3/3). `timefmt` 1/3 is a real edge-case miss. |
| Open-ended skills (judge) | 5/5, scored by a separate strong model (`reviews/qwen3.6-thinking.md`) |
| Throughput | ~29 tok/s @128K, matching dense ornith @32K despite the RAM offload (A3B) |

vs `ornith:9b`: comparable correctness (0.86 vs 0.83 pass@1; both 5/5 judge) at 4×
the context and the same speed. The lower pass@k (0.92 vs 1.0) is a single latency
timeout, not a reasoning gap.

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
