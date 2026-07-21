# miniagent

A stdlib-only local coding agent for small tasks, run offline. The CLI drives a
local model through tool calls (`run_bash`, `read_file`, `write_file`) and keeps
a persistent, project-scoped SQLite memory across sessions. Default model:
`qwen3.6` (35B-A3B MoE, experts offloaded to RAM); `ornith:9b` (dense 9B) is the
lighter alternative. Eval in `STATUS.md`. Backend: a local
[llama-server](https://github.com/ggml-org/llama.cpp) if one is up, otherwise
[Ollama](https://ollama.com) as a fallback.

## Directory structure

- `src/`: agent (`miniagent.py`) and persistent memory (`memory.py`).
- `tests/`: unit tests for memory and tool gating; no LLM or network.
- `benchmarks/`: `minibench.py` (oracle-graded), `judge.py` (model-judged),
  `membench.py` (memory), `speedbench.py` (throughput).
- `scripts/serve.py` + `models.json`: launch a llama-server from a model config.
- `reviews/`: judge bundles for a strong model to score.
- `STATUS.md`: verified capabilities, measured limits, improvement queue.

## Setup

```bash
conda env create -f environment.yml
conda activate miniagent
```

Then bring up a model backend. The agent prefers a local **llama-server** and
falls back to **Ollama**; force either with `MINIAGENT_BACKEND=llama|ollama`.

**llama-server (preferred).** Launch configs live in `models.json` (label → GGUF
+ flags); `scripts/serve.py` runs one:

```bash
python scripts/serve.py qwen3.6        # default: 35B-A3B MoE, experts offloaded to RAM, 128K ctx (:8081)
python scripts/serve.py ornith:9b      # lighter: dense 9B, whole model on GPU, 32K ctx (:8080)
```

Set `LLAMA_SERVER_BIN` if `llama-server` isn't at `~/llama.cpp-src/build/bin/`.
The two configs are tuned for an 8 GB GPU so VRAM stays ~7.2 GB (with
flash-attention on): ornith fits whole; qwen keeps ~5 of its 40 expert-layers on
GPU (`--n-cpu-moe 35`), the rest in system RAM. Context and expert-offload trade
off against the same VRAM — see the hardware notes for the math.

**Ollama (fallback).** When no llama-server is up, the agent falls back to Ollama
serving `ornith:9b` automatically — the model is coupled to the backend
(llama → `qwen3.6`, ollama → `ornith:9b`; override with `MINIAGENT_MODEL`), since
the MoE default isn't practical in Ollama. Just `ollama pull ornith:9b`. Ollama
defaults to a 4096-token context, which truncates tool schemas and breaks
tool-calling, so miniagent raises it per request (`MINIAGENT_NUM_CTX`, default 32768).

`opencode` points at the same llama-server (`opencode.json`, OpenAI `/v1`).

## Usage

Independent steps; each needs a backend up — a local llama-server (preferred)
or `ollama serve` (fallback).

1. **Chat interactively**

   ```bash
   python src/miniagent.py
   ```

   - `run_bash` confirms every command, `write_file` confirms overwrites;
     `MINIAGENT_AUTO_CONFIRM=1` skips both.
   - Env overrides: `MINIAGENT_MODEL`, `_BACKEND`, `_LLAMA_URL`, `_BASE_URL`
     (Ollama), `_NUM_CTX`, `_TEMP` (llama sampling temp), `_THINK`,
     `_MAX_ROUNDS`, `_MEMORY_DB`, `_SCOPE`.

2. **Run one prompt**

   ```bash
   python src/miniagent.py --prompt "rename every .txt file in ./notes to .md"
   ```

3. **Benchmark models** — oracle-graded pass@1 / pass@k / latency:

   ```bash
   python benchmarks/minibench.py --models qwen3.6 --n 3 --think --csv benchmarks/minibench_results.csv
   ```

4. **Judge open-ended skills** — writes `reviews/<model>.md` for a strong
   model (never the model under test) to score:

   ```bash
   python benchmarks/judge.py --model qwen3.6 --think
   ```

5. **Benchmark memory** — plants facts in one session, probes in a fresh one;
   reports per stage (`extracted`/`recalled`/`used`). Stochastic: use `--n 3`+.

   ```bash
   python benchmarks/membench.py --models qwen3.6 --n 3 --think
   ```

6. **Benchmark throughput** — brings up each model (one at a time), warms it, and
   reports median generation tok/s over several fixed-length runs:

   ```bash
   python benchmarks/speedbench.py --models ornith:9b,qwen3.6 --tokens 512 --trials 3
   ```

## Memory

Each turn injects the top BM25 matches into the system prompt; after each
exchange an LLM call extracts durable typed facts into `.miniagent/memory.db`.
Facts and decisions are scoped to the agent's working directory, preferences
are global. Ranking applies type weights and recency decay; a contradicted
preference or decision retires its predecessor. Limits: `STATUS.md`.

## Tests

No LLM or network, ~1 s:

```bash
python -m unittest discover tests
```
