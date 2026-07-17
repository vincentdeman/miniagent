"""Task benchmark for miniagent: pass@1 / pass@k / p50 latency across models.

Runs the agent as a black box in a scratch cwd and scores whether each task's
artifact actually works. Agent contract: `python $AGENT --prompt P` in the
scratch dir, with env MINIAGENT_MODEL, MINIAGENT_THINK (0/1),
MINIAGENT_AUTO_CONFIRM, MINIAGENT_MAX_ROUNDS. The agent disables its own
chain-of-thought when THINK is 0.

    python benchmarks/minibench.py --models ornith:9b --n 3 --csv benchmarks/minibench_results.csv
"""

import argparse, os, signal, subprocess, sys, tempfile, time, shutil
from statistics import median

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = os.path.abspath(os.environ.get("AGENT_PATH", os.path.join(_ROOT, "src", "miniagent.py")))
PY = sys.executable   # same interpreter we're run with; `python` may not be on PATH
MAX_ROUNDS = "12"

def _run(work, script, t=30):
    """Run a script in the scratch dir; return (returncode, stdout)."""
    p = subprocess.run([PY, script], cwd=work, capture_output=True, text=True, timeout=t)
    return p.returncode, p.stdout

def _ok(work, script):
    """True if `script` runs to a clean exit (shipped test files print 'ok')."""
    try: rc, _ = _run(work, script)
    except Exception: return False
    return rc == 0

def _prints(work, script, expected):
    """True if `script` runs clean and its stdout equals `expected`."""
    try: rc, out = _run(work, script)
    except Exception: return False
    return rc == 0 and out.strip() == expected

def fizzbuzz(work):
    exp = "\n".join("FizzBuzz" if i%15==0 else "Fizz" if i%3==0 else
                    "Buzz" if i%5==0 else str(i) for i in range(1, 101))
    return _prints(work, "solution.py", exp)

def _src(work, f):
    try: return open(f"{work}/{f}").read()
    except Exception: return None

def _ok_banning(work, script, test, *banned):
    """Test passes AND `script`'s source contains none of `banned` (forbid shortcuts)."""
    src = _src(work, script)
    return src is not None and not any(b in src for b in banned) and _ok(work, test)

def _ok_requiring(work, script, test, *required):
    """Test passes AND `script`'s source contains every `required` string (force real use)."""
    src = _src(work, script)
    return src is not None and all(r in src for r in required) and _ok(work, test)

# Differential grading: seed the RNG, generate many cases, and compare the candidate
# against a stdlib reference the solution itself may not import. One wrong case fails.
_REGEX_TEST = r'''
import random, re
from matcher import match
assert match('a','a') and not match('a','b')
assert match('a*','') and match('a*','aaa')
assert match('(ab)+','abab') and not match('(ab)+','aba')
assert match('(a|ab)*','aab') and match('(a|ab)+c','abac')
assert match('[a-c]+','abcabc') and not match('(a|b)*c','abab')

random.seed(7)
A = 'ab'                       # small alphabet keeps matches ambiguous
def cls(): return '[' + ''.join(sorted(set(random.sample(A, random.randint(1,2))))) + ']'
def prim(): return random.choice([random.choice(A), '.', cls()])
def group():                   # one level: a plain alternation, no nesting or inner quantifiers
    branch = lambda: ''.join(prim() for _ in range(random.randint(1,3)))
    return '(' + '|'.join(branch() for _ in range(random.randint(1,3))) + ')'
def atom(): return group() if random.random() < 0.4 else prim()
def quant(p): return p + random.choice(['', '*', '+', '?'])
def concat(): return ''.join(quant(atom()) for _ in range(random.randint(1,3)))
def pattern(): return '|'.join(concat() for _ in range(random.randint(1,2)))

for _ in range(300):
    p = pattern()
    s = ''.join(random.choice(A) for _ in range(random.randint(0,6)))
    assert bool(match(p,s)) == (re.fullmatch(p,s) is not None), (p,s)
print('ok')
'''

_JSON_TEST = r'''
import random, json
from jsonparse import parse
assert parse('true') is True and parse('null') is None and parse('-3.5')==-3.5
assert parse('"☃"')=='☃' and parse('"a\\nb"')=='a\nb'
assert parse('{"a":[1,2],"b":"x"}')=={'a':[1,2],'b':'x'}

random.seed(99)
def rnd(d):
    t = random.random()
    if d < 3 and t < 0.25: return {('k%d'%i): rnd(d+1) for i in range(random.randint(0,3))}
    if d < 3 and t < 0.50: return [rnd(d+1) for _ in range(random.randint(0,3))]
    if t < 0.62: return random.choice([0,1,-5,42,1000])
    if t < 0.74: return random.choice([0.0,-3.5,1e3,2.25,-0.5])
    if t < 0.88: return random.choice([True,False,None])
    return ''.join(random.choice('ab "\\\n\t/') for _ in range(random.randint(0,5)))

for _ in range(300):
    text = json.dumps(rnd(0))
    assert parse(text) == json.loads(text), text
print('ok')
'''

TASKS = [
    # --- smoke: plumbing works end to end ---
    ("fizzbuzz",
     "Write solution.py that prints 1 to 100 one per line, Fizz for multiples of 3, "
     "Buzz for 5, FizzBuzz for both.", {}, fizzbuzz),

    # ===== mid tier: each isolates ONE distinct capability =====

    # subtle bug: spot the Python gotcha (mutable default argument)
    ("subtlebug",
     "acc.py has a subtle bug: on a fresh call add(2) should return [2], but it returns [1,2]. "
     "Read acc.py, find the cause, fix it. Make test_acc.py pass.",
     {"acc.py": "def add(x, into=[]):\n    into.append(x)\n    return into\n",
      "test_acc.py": "from acc import add\nassert add(1)==[1]\nassert add(2)==[2]\n"
                     "assert add(3)==[3]\nprint('ok')\n"},
     lambda w: _ok(w, "test_acc.py")),

    # multi-file wiring: create the module main.py imports
    ("multifile",
     "main.py imports tokenize() from a module named lexer. Create lexer.py with tokenize(s) "
     "returning the list of tokens in s split on whitespace or commas, with empty strings dropped. "
     "Make test_main.py pass.",
     {"main.py": "from lexer import tokenize\n\ndef run(s):\n    return len(tokenize(s))\n",
      "test_main.py": "from main import run\nassert run('a,b,c')==3\nassert run('')==0\n"
                      "assert run('one two three four')==4\nassert run('  x  ')==1\nprint('ok')\n"},
     lambda w: _ok(w, "test_main.py")),

    # API comprehension: use a given class correctly (methods mutate + return None)
    ("api-use",
     "Using the Ledger class in ledger.py unchanged (credit(x)/debit(x) change the balance and "
     "return None; read the current total from the .balance property), create solve.py with "
     "apply(ops): ops is a list of (kind, amount) where kind is 'credit' or 'debit'; return the "
     "final balance. Make test_solve.py pass.",
     {"ledger.py": "class Ledger:\n    def __init__(self):\n        self._b = 0\n"
                   "    def credit(self, x):\n        self._b += x\n"
                   "    def debit(self, x):\n        self._b -= x\n"
                   "    @property\n    def balance(self):\n        return self._b\n",
      "test_solve.py": "from solve import apply\nassert apply([('credit',100),('debit',30)])==70\n"
                       "assert apply([])==0\nassert apply([('credit',5),('credit',5),('debit',3)])==7\n"
                       "print('ok')\n"},
     lambda w: _ok_requiring(w, "solve.py", "test_solve.py", "Ledger(")),

    # edge-case correctness: careful modular arithmetic + formatting
    ("timefmt",
     "Create timefmt.py with fmt(seconds) returning the duration as 'HH:MM:SS', hours zero-padded "
     "to two digits but allowed to exceed 24. Make test_timefmt.py pass.",
     {"test_timefmt.py": "from timefmt import fmt\nassert fmt(0)=='00:00:00'\n"
                         "assert fmt(3665)=='01:01:05'\nassert fmt(86399)=='23:59:59'\n"
                         "assert fmt(90061)=='25:01:01'\nprint('ok')\n"},
     lambda w: _ok(w, "test_timefmt.py")),

    # non-trivial algorithm: sort + merge with touching/unsorted edges
    ("intervals",
     "Create merge.py with merge(intervals): a list of [start, end] pairs; return the "
     "non-overlapping intervals sorted by start, merging any that overlap or touch. "
     "Make test_merge.py pass.",
     {"test_merge.py": "from merge import merge\n"
                       "assert merge([[1,3],[2,6],[8,10]])==[[1,6],[8,10]]\nassert merge([])==[]\n"
                       "assert merge([[1,4],[4,5]])==[[1,5]]\n"
                       "assert merge([[5,6],[1,3],[2,4]])==[[1,4],[5,6]]\nprint('ok')\n"},
     lambda w: _ok(w, "test_merge.py")),

    # constraint adherence: implement a sort without the built-ins
    ("constrained-sort",
     "Create mysort.py with mysort(xs) returning xs sorted ascending. You may not use sorted() or "
     "list.sort() — implement the ordering yourself. Make test_mysort.py pass.",
     {"test_mysort.py": "from mysort import mysort\nassert mysort([3,1,2])==[1,2,3]\n"
                        "assert mysort([])==[]\nassert mysort([5,5,1])==[1,5,5]\n"
                        "assert mysort([-1,-3,2])==[-3,-1,2]\nprint('ok')\n"},
     lambda w: _ok_banning(w, "mysort.py", "test_mysort.py", "sorted(", ".sort(")),

    # efficiency: the idiomatic one-liner is O(n^2) and times out; needs O(n)
    ("efficiency",
     "Create freq.py with most_common(nums) returning the value that appears most often. "
     "It must handle 200000 elements quickly. Make test_freq.py pass.",
     {"test_freq.py": "from freq import most_common\nassert most_common([1,1,2])==1\n"
                      "assert most_common([3])==3\nbig=[7]*100000+list(range(100000))\n"
                      "assert most_common(big)==7\nprint('ok')\n"},
     lambda w: _ok(w, "test_freq.py")),

    # ===== hard tier: multi-step and algorithmic =====

    # long-horizon debugging: 3 interacting bugs, must fix all (run/read/fix loop)
    ("multibug",
     "calc3.py has several bugs across its functions. Run test_calc3.py, then fix calc3.py until "
     "every test passes. Iterate: run, read the failure, fix.",
     {"calc3.py": "def add_all(nums):\n    t=1\n    for n in nums:\n        t+=n\n    return t\n\n"
                  "def average(nums):\n    return add_all(nums)//len(nums)\n\n"
                  "def maximum(nums):\n    m=0\n    for n in nums:\n        if n>m:\n            m=n\n    return m\n",
      "test_calc3.py": "from calc3 import add_all, average, maximum\n"
                       "assert add_all([])==0 and add_all([1,2,3])==6\n"
                       "assert average([1,2,3,4])==2.5\n"
                       "assert maximum([-3,-1,-2])==-1 and maximum([1,5,2])==5\nprint('ok')\n"},
     lambda w: _ok(w, "test_calc3.py")),

    # graph algorithm: topological order, cycle -> None
    ("topo-sort",
     "Create topo.py with topo(graph): graph maps each node to the list of its prerequisite nodes. "
     "Return a list of all nodes ordered so every node comes after its prerequisites, or None if a "
     "cycle makes that impossible. Make test_topo.py pass.",
     {"test_topo.py": "from topo import topo\n"
                      "def ok(o,g):\n    p={n:i for i,n in enumerate(o)}\n"
                      "    return set(o)==set(g) and all(p[q]<p[n] for n in g for q in g[n])\n"
                      "g={'a':[],'b':['a'],'c':['a'],'d':['b','c']}\nassert ok(topo(g),g)\n"
                      "assert topo({'a':['b'],'b':['a']}) is None\n"
                      "assert topo({})==[]\n"
                      "assert topo({'x':[],'y':['x'],'z':['y']})==['x','y','z']\nprint('ok')\n"},
     lambda w: _ok(w, "test_topo.py")),

    # ===== frontier tier: differential fuzz vs a stdlib reference (300 seeded cases) =====

    # full regex engine: literals, . [] () | and * + ? — graded against `re.fullmatch`
    ("regex-engine",
     "Create matcher.py with match(pattern, s) returning True iff pattern matches the ENTIRE "
     "string s. Support literals, '.' (any char), character classes like [abc] and [a-z], "
     "grouping (...), alternation |, and the quantifiers * + ? on the preceding element or group. "
     "Do not import re. Make test_matcher.py pass.",
     {"test_matcher.py": _REGEX_TEST},
     lambda w: _ok_banning(w, "matcher.py", "test_matcher.py", "import re")),

    # full JSON parse matching json.loads: escapes incl. \\uXXXX, floats with exponents, nesting
    ("json-roundtrip",
     "Create jsonparse.py with parse(text) parsing JSON into the equivalent Python value, matching "
     "json.loads: objects, arrays, strings with the standard escapes including \\uXXXX, integers, "
     "floats with exponents, true/false/null, and arbitrary whitespace. Do not use json, ast, or "
     "eval. Make test_jsonparse.py pass.",
     {"test_jsonparse.py": _JSON_TEST},
     lambda w: _ok_banning(w, "jsonparse.py", "test_jsonparse.py",
                           "import json", "import ast", "eval(", "exec(")),
]

def run_agent(work, model, prompt, think, timeout):
    """Run the agent once in `work`; return True if it was killed on timeout.

    DEVNULL (output is unused) + own session so a timeout kills any grandchildren
    the agent spawned; capturing pipes can hang past `timeout` while a child is open.
    """
    env = {**os.environ, "MINIAGENT_MODEL": model, "MINIAGENT_THINK": str(int(think)),
           "MINIAGENT_AUTO_CONFIRM": "1", "MINIAGENT_MAX_ROUNDS": MAX_ROUNDS,
           # scratch memory DB: no recall from earlier attempts, no writes to the real DB
           "MINIAGENT_MEMORY_DB": os.path.join(work, ".miniagent", "memory.db")}
    p = subprocess.Popen([PY, AGENT, "--prompt", prompt], cwd=work, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    try:
        p.wait(timeout=timeout); return False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); return True

def once(model, prompt, setup, check, think, timeout):
    """Run one attempt in a fresh scratch dir; return (passed, ms, timed_out)."""
    work = tempfile.mkdtemp()
    for name, content in setup.items():
        open(f"{work}/{name}", "w").write(content)
    t0 = time.monotonic()                              # monotonic: immune to wall-clock jumps
    timed_out = run_agent(work, model, prompt, think, timeout)
    ms = (time.monotonic() - t0) * 1000
    try: ok = check(work)
    except Exception: ok = False
    shutil.rmtree(work, ignore_errors=True)
    return ok, ms, timed_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--only", help="comma-separated task ids to run")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--csv", help="append per-attempt rows to this CSV file")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    want = {s.strip() for s in a.only.split(",")} if a.only else None
    tasks = [t for t in TASKS if t[0] in want] if want else TASKS
    think = int(a.think)
    print(f"thinking={'on' if a.think else 'off'}  n={a.n}  tasks={len(tasks)}  models={','.join(models)}\n")

    rows = []                                          # (model, task, passed, ms, timed_out)
    for m in models:
        for tid, prompt, setup, check in tasks:
            for _ in range(a.n):
                ok, ms, to = once(m, prompt, setup, check, a.think, a.timeout)
                rows.append((m, tid, ok, ms, to))
                tag = "TIMEOUT" if to else ("PASS" if ok else "FAIL")
                print(f"{m:16} {tid:16} {tag:7} {ms:7.0f}ms")

    if a.csv:                                          # structured output; append across runs
        new = not os.path.exists(a.csv)
        with open(a.csv, "a") as f:
            if new: f.write("model,task,think,passed,ms,timed_out\n")
            for m, tid, ok, ms, to in rows:
                f.write(f"{m},{tid},{think},{int(ok)},{ms:.0f},{int(to)}\n")

    def cell(m, tid):                                  # (passes, attempts, timeouts) for a model+task
        v = [(ok, to) for mm, t, ok, _, to in rows if mm == m and t == tid]
        return sum(ok for ok, _ in v), len(v), sum(to for _, to in v)

    print(f"\n{'model':16} {'pass@1':>7} {'pass@k':>7} {'p50_ms':>8} {'timeouts':>9}")
    for m in models:
        mr = [r for r in rows if r[0] == m]
        p1 = sum(r[2] for r in mr) / len(mr)
        bytask = {}
        for _, tid, ok, _, _ in mr: bytask.setdefault(tid, []).append(ok)
        pk = sum(any(v) for v in bytask.values()) / len(bytask)
        print(f"{m:16} {p1:7.2f} {pk:7.2f} {median(r[3] for r in mr):8.0f} {sum(r[4] for r in mr):9d}")

    # per-task pass counts (k/n, +Nt timeouts); `split` = pass-rate gap -> does the task discriminate
    print(f"\n{'task':16} " + " ".join(f"{m:>14}" for m in models) + "   split")
    for tid, *_ in tasks:
        counts = [cell(m, tid) for m in models]
        rates = [k / n if n else 0 for k, n, _ in counts]
        split = f"{max(rates) - min(rates):.2f}" if len(models) > 1 and max(rates) != min(rates) else ""
        cells = " ".join((f"{k}/{n}" + (f"+{t}t" if t else "")).rjust(14) for k, n, t in counts)
        print(f"{tid:16} {cells}   {split}")

if __name__ == "__main__":
    main()
