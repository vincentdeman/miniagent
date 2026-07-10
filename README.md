# miniagent

A stdlib-only local coding agent for small tasks, run offline against
[Ollama](https://ollama.com). The CLI drives a local model through tool calls
(`run_bash`, `read_file`, `write_file`) and keeps a persistent, project-scoped
SQLite memory across sessions. Default model: `ornith:9b` (eval in `STATUS.md`).

## Directory structure

- `src/`: agent (`miniagent.py`) and persistent memory (`memory.py`).
- `tests/`: unit tests for memory and tool gating; no LLM or network.
- `benchmarks/`: `minibench.py` (oracle-graded), `judge.py` (model-judged),
  `membench.py` (memory).
- `reviews/`: judge bundles for a strong model to score.
- `STATUS.md`: verified capabilities, measured limits, improvement queue.

## Setup

```bash
conda env create -f environment.yml
conda activate miniagent
ollama pull ornith:9b
```

## Usage

Independent steps; all need `ollama serve` running.

1. **Chat interactively**

   ```bash
   python src/miniagent.py
   ```

   - `run_bash` confirms every command, `write_file` confirms overwrites;
     `MINIAGENT_AUTO_CONFIRM=1` skips both.
   - Env overrides: `MINIAGENT_MODEL`, `_BASE_URL`, `_NUM_CTX`, `_THINK`,
     `_MAX_ROUNDS`, `_MEMORY_DB`, `_SCOPE`.

2. **Run one prompt**

   ```bash
   python src/miniagent.py --prompt "rename every .txt file in ./notes to .md"
   ```

3. **Benchmark models** — oracle-graded pass@1 / pass@k / latency:

   ```bash
   python benchmarks/minibench.py --models ornith:9b,qwen3:4b --n 3 --think --csv benchmarks/minibench_results.csv
   ```

4. **Judge open-ended skills** — writes `reviews/<model>.md` for a strong
   model (never the model under test) to score:

   ```bash
   python benchmarks/judge.py --model ornith:9b --think
   ```

5. **Benchmark memory** — plants facts in one session, probes in a fresh one;
   reports per stage (`extracted`/`recalled`/`used`). Stochastic: use `--n 3`+.

   ```bash
   python benchmarks/membench.py --models ornith:9b --n 3 --think
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
