"""Model-judged tasks for miniagent: capabilities with no cheap oracle.

Assert-based grading (bench.py) only covers tasks with one computable answer. These
tasks test open-ended skills — refactoring, diagnosis, review — where a strong model
must verify the artifact. Each task runs the agent, applies a deterministic GATE where
one exists (e.g. tests still pass), then writes the produced artifact into a markdown
bundle. A strong model then reads the bundle and scores each artifact against its rubric.

Rule: never judge with the model under test — self-grading is biased.

    python judge.py --model ornith:9b --out reviews/
    # then have Claude read reviews/<model>.md and score each artifact
"""

import argparse, os, subprocess, tempfile, shutil
from minibench import PY, run_agent   # reuse the agent runner + launch contract


def tests_pass(script):
    """Deterministic gate: `script` runs to a clean exit in the work dir."""
    def gate(work):
        try:
            return subprocess.run([PY, script], cwd=work, capture_output=True,
                                  text=True, timeout=30).returncode == 0
        except Exception:
            return False
    return gate


# (id, prompt, setup, gate | None, [artifact files], rubric for the judge)
TASKS = [
    ("refactor",
     "Refactor shape.py to remove duplication and improve clarity WITHOUT changing behavior. "
     "Keep test_shape.py passing.",
     {"shape.py": "def area(kind, a, b=0):\n"
                  "    if kind == 'rect': return a * b\n"
                  "    if kind == 'square': return a * a\n"
                  "    if kind == 'circle': return 3.14159 * a * a\n\n"
                  "def perimeter(kind, a, b=0):\n"
                  "    if kind == 'rect': return a + b + a + b\n"
                  "    if kind == 'square': return a + a + a + a\n"
                  "    if kind == 'circle': return 2 * 3.14159 * a\n",
      "test_shape.py": "from shape import area, perimeter\n"
                       "assert area('rect',3,4)==12 and area('square',5)==25\n"
                       "assert abs(area('circle',2)-12.56636)<1e-9\n"
                       "assert perimeter('rect',3,4)==14 and perimeter('square',5)==20\n"
                       "assert abs(perimeter('circle',2)-12.56636)<1e-9\nprint('ok')\n"},
     tests_pass("test_shape.py"), ["shape.py"],
     "Gate already confirms behavior is unchanged. Is shape.py MEANINGFULLY cleaner than the "
     "original — duplication removed (single dispatch/shared pi), clearer structure — rather than "
     "cosmetically shuffled? PASS only if a reviewer would prefer it."),

    ("bug-explain",
     "Read buggy.py and write explanation.md explaining the bug it contains and why the result is "
     "wrong. Do NOT fix the code.",
     {"buggy.py": "def average_gap(nums):\n"
                  "    # average difference between consecutive values (sorted)\n"
                  "    nums = sorted(nums)\n"
                  "    gaps = []\n"
                  "    for i in range(len(nums)):\n"
                  "        gaps.append(nums[i] - nums[i-1])\n"
                  "    return sum(gaps) / len(gaps)\n"},
     None, ["explanation.md"],
     "The real bug: the loop starts at i=0, so nums[i-1] is nums[-1] (the largest element) instead "
     "of being skipped; the range should start at 1. It is silent (no crash), just a wrong number. "
     "PASS only if the explanation identifies THIS mechanism, not a vague or wrong diagnosis."),

    ("code-review",
     "Review changes.py and write review.md listing the problems you find, most important first.",
     {"changes.py": "def load(path, cache={}):\n"
                    "    if path in cache:\n"
                    "        return cache[path]\n"
                    "    try:\n"
                    "        data = open(path).read()\n"
                    "    except:\n"
                    "        return None\n"
                    "    cache[path] = data\n"
                    "    return data\n"},
     None, ["review.md"],
     "Planted issues: (1) mutable default arg `cache={}` shared across calls; (2) file handle from "
     "open() never closed; (3) bare `except:` swallows all errors incl. KeyboardInterrupt. Score by "
     "how many of the three it catches AND signal-to-noise (penalize invented/wrong findings)."),

    ("docstring",
     "Read rle.py and add a clear docstring to encode() describing what it does, its parameter, its "
     "return value, and its behavior on empty input. Do not change the code's behavior.",
     {"rle.py": "def encode(s):\n    if not s:\n        return ''\n    out = []\n"
                "    prev = s[0]\n    count = 1\n    for c in s[1:]:\n"
                "        if c == prev:\n            count += 1\n"
                "        else:\n            out.append(prev + str(count))\n            prev = c\n            count = 1\n"
                "    out.append(prev + str(count))\n    return ''.join(out)\n",
      "test_rle.py": "from rle import encode\n"
                     "assert encode('aaabbc')=='a3b2c1' and encode('')=='' and encode('x')=='x1'\nprint('ok')\n"},
     tests_pass("test_rle.py"), ["rle.py"],
     "Behavior is gated. Does the docstring accurately and completely describe encode(): run-length "
     "encoding to a string like 'a3b2c1', the string parameter, the string return, and empty input "
     "-> ''? PASS only if accurate and genuinely informative, not a vague one-liner."),

    ("test-design",
     "is_leap.py contains a correct is_leap(year). Write test_leap.py whose assertions thoroughly "
     "test it, including the tricky century rules. Make test_leap.py pass.",
     {"is_leap.py": "def is_leap(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n"},
     tests_pass("test_leap.py"), ["test_leap.py"],
     "Gate confirms the tests run and pass on the correct code. Do they cover the DISCRIMINATING "
     "cases — a common leap year (2004), a common non-leap (2001), a century non-leap (1900), and a "
     "400-divisible leap (2000)? PASS only if the century/400 edges are tested, not just multiples of 4."),
]


def run_task(model, prompt, setup, gate, artifacts, think, timeout):
    """Run the agent once; return (gate_result_or_None, {filename: contents})."""
    work = tempfile.mkdtemp()
    for name, content in setup.items():
        open(f"{work}/{name}", "w").write(content)
    run_agent(work, model, prompt, think, timeout)     # timeout attribution unused here
    gres = None if gate is None else gate(work)
    files = {}
    for a in artifacts:
        try: files[a] = open(f"{work}/{a}").read()
        except Exception: files[a] = "(missing — agent did not produce this file)"
    shutil.rmtree(work, ignore_errors=True)
    return gres, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="reviews")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    md = [f"# Judge bundle — {a.model} (think={'on' if a.think else 'off'})",
          "Score each artifact against its rubric: PASS / FAIL + one-line reason.", ""]
    for tid, prompt, setup, gate, artifacts, rubric in TASKS:
        gres, files = run_task(a.model, prompt, setup, gate, artifacts, a.think, a.timeout)
        tag = "n/a" if gres is None else ("PASS" if gres else "FAIL")
        print(f"{tid:12} gate={tag}")
        md += [f"## {tid}", "", f"**Prompt:** {prompt}", "", f"**Rubric:** {rubric}", "",
               f"**Deterministic gate:** {tag}", ""]
        for name, content in files.items():
            md += [f"### artifact: {name}", "```", content.rstrip(), "```", ""]
    suffix = "-thinking" if a.think else ""
    path = os.path.join(a.out, f"{a.model.replace(':', '-')}{suffix}.md")
    open(path, "w").write("\n".join(md))
    print(f"\nbundle -> {path}\nHave a STRONG model (not {a.model}) read it and score each artifact.")


if __name__ == "__main__":
    main()
