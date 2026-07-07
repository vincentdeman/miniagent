# Judge bundle — ornith:9b (think=on)
Score each artifact against its rubric: PASS / FAIL + one-line reason.

## refactor

**Prompt:** Refactor shape.py to remove duplication and improve clarity WITHOUT changing behavior. Keep test_shape.py passing.

**Rubric:** Gate already confirms behavior is unchanged. Is shape.py MEANINGFULLY cleaner than the original — duplication removed (single dispatch/shared pi), clearer structure — rather than cosmetically shuffled? PASS only if a reviewer would prefer it.

**Deterministic gate:** PASS

### artifact: shape.py
```
# Shape definitions: each entry is (area_formula, perimeter_formula).
_SHAPES = {
    'rect':   ((lambda a, b: a * b),       (lambda a, b: 2 * (a + b))),
    'square': ((lambda a, _=None: a * a),   (lambda a, _=None: 4 * a)),
    'circle': ((lambda a, _=None: 3.14159 * a * a),(lambda a, _=None: 2 * 3.14159 * a)),
}

def area(kind, a, b=0):
    return _SHAPES[kind][0](a, b)

def perimeter(kind, a, b=0):
    return _SHAPES[kind][1](a, b)
```

## bug-explain

**Prompt:** Read buggy.py and write explanation.md explaining the bug it contains and why the result is wrong. Do NOT fix the code.

**Rubric:** The real bug: the loop starts at i=0, so nums[i-1] is nums[-1] (the largest element) instead of being skipped; the range should start at 1. It is silent (no crash), just a wrong number. PASS only if the explanation identifies THIS mechanism, not a vague or wrong diagnosis.

**Deterministic gate:** n/a

### artifact: explanation.md
```
# Bug Explanation for `buggy.py`

## The Bug

In the function `average_gap`, the loop iterates over all indices from `0` to `len(nums) - 1`:

```python
for i in range(len(nums)):
    gaps.append(nums[i] - nums[i-1])
```

When `i = 0`, this accesses `nums[-1]`, which — due to Python's negative indexing — refers to the **last** element of the sorted list, not a "previous" element. This produces an incorrect first gap: `nums[0] - nums[len(nums)-1]`.

## Why the Result Is Wrong

The function is supposed to compute the average difference between *consecutive* values in the sorted sequence (i.e., `nums[1]-nums[0]`, `nums[2]-nums[1]`, …). But because the loop starts at index 0, it computes one extra gap that wraps around from the last element back to the first. This spurious value is included in both the sum and the count of gaps, so:

- The numerator (`sum(gaps)`) includes an incorrect term.
- The denominator (`len(gaps)`) counts one too many entries.

The returned average therefore does not represent the true mean gap between consecutive sorted values.

## Correct Approach

The loop should start at index 1:

```python
for i in range(1, len(nums)):
    gaps.append(nums[i] - nums[i-1])
```

This ensures only genuine consecutive differences are computed. (Note: this explanation does not modify the original code.)
```

## code-review

**Prompt:** Review changes.py and write review.md listing the problems you find, most important first.

**Rubric:** Planted issues: (1) mutable default arg `cache={}` shared across calls; (2) file handle from open() never closed; (3) bare `except:` swallows all errors incl. KeyboardInterrupt. Score by how many of the three it catches AND signal-to-noise (penalize invented/wrong findings).

**Deterministic gate:** n/a

### artifact: review.md
```
# Review of `changes.py`

## 1. Mutable default argument (`cache={}`) — critical bug

The parameter `cache={}` is a mutable default that persists across calls. Every invocation shares the same dict, so:

- Cached values from one call leak into subsequent calls (wrong data returned).
- The cache grows unbounded and never gets cleared.

**Fix:** Use `None` as the default and instantiate inside the function body:

```python
def load(path, cache=None):
    if cache is None:
        cache = {}
    ...
```

## 2. Bare `except:` swallows all exceptions — critical bug

`except:` catches every exception including `KeyboardInterrupt`, `SystemExit`, and unexpected errors. The function silently returns `None`, hiding failures from the caller. This makes debugging nearly impossible.

**Fix:** Catch specific exceptions (e.g., `OSError`) or at least re-raise after logging:

```python
except OSError as e:
    print(f"Failed to read {path}: {e}")
    return None
```

## 3. No input validation — minor issue

If `path` is not a string (or any other non-string type), the function will still attempt to call it, which may raise an exception that gets swallowed by #2. A simple type check would help:

```python
if not isinstance(path, str):
    raise TypeError("path must be a string")
```

## 4. Resource leak — minor issue

`open(path).read()` does not close the file handle explicitly (no `with` statement). While Python's garbage collector eventually reclaims it, this is bad practice and can cause issues in long-running processes or under memory pressure.

**Fix:** Use a context manager:

```python
data = open(path).read()  # or better: with open(path) as f: data = f.read()
```

## 5. Returning `None` on failure — minor issue

Returning `None` makes it impossible for callers to distinguish between "file not found", "permission denied", and other errors. A more informative approach is to raise an exception or return a tuple with status + data.

**Fix:** Raise `FileNotFoundError` explicitly, or use a custom result type:

```python
if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")
```
```

## docstring

**Prompt:** Read rle.py and add a clear docstring to encode() describing what it does, its parameter, its return value, and its behavior on empty input. Do not change the code's behavior.

**Rubric:** Behavior is gated. Does the docstring accurately and completely describe encode(): run-length encoding to a string like 'a3b2c1', the string parameter, the string return, and empty input -> ''? PASS only if accurate and genuinely informative, not a vague one-liner.

**Deterministic gate:** PASS

### artifact: rle.py
```
def encode(s):
    """Run-length encode the input string *s*.

    Each maximal run of identical consecutive characters in *s* is replaced
    by that character followed by its length. The runs are concatenated and
    returned as a single string.

    Parameters
    ----------
    s : str
        The input string to encode.

    Returns
    -------
    str
        The run-length encoded representation of *s*. For example,
        ``encode("aaabbc")`` returns ``"a3b2c1"``.

    Behavior on empty input
    -----------------------
    If *s* is the empty string (or any falsy value), an empty string is
    returned immediately without further processing.
    """
    if not s:
        return ''
    out = []
    prev = s[0]
    count = 1
    for c in s[1:]:
        if c == prev:
            count += 1
        else:
            out.append(prev + str(count))
            prev = c
            count = 1
    out.append(prev + str(count))
    return ''.join(out)
```

## test-design

**Prompt:** is_leap.py contains a correct is_leap(year). Write test_leap.py whose assertions thoroughly test it, including the tricky century rules. Make test_leap.py pass.

**Rubric:** Gate confirms the tests run and pass on the correct code. Do they cover the DISCRIMINATING cases — a common leap year (2004), a common non-leap (2001), a century non-leap (1900), and a 400-divisible leap (2000)? PASS only if the century/400 edges are tested, not just multiples of 4.

**Deterministic gate:** PASS

### artifact: test_leap.py
```
from is_leap import is_leap


def test_normal_leap_years():
    """Years divisible by 4 (non-century) are leap years."""
    assert is_leap(1996) is True
    assert is_leap(2000) is True
    assert is_leap(2004) is True
    assert is_leap(2008) is True
    assert is_leap(2012) is True
    assert is_leap(2016) is True
    assert is_leap(2020) is True
    assert is_leap(2024) is True


def test_normal_non_leap_years():
    """Years not divisible by 4 are not leap years."""
    assert is_leap(1997) is False
    assert is_leap(1998) is False
    assert is_leap(1999) is False
    assert is_leap(2001) is False
    assert is_leap(2002) is False
    assert is_leap(2003) is False
    assert is_leap(2005) is False
    assert is_leap(2006) is False


def test_century_non_leap_years():
    """Years divisible by 100 but NOT 400 are NOT leap years (the tricky rule)."""
    # 1700: div by 100, not 400 -> not leap
    assert is_leap(1700) is False
    # 1800: div by 100, not 400 -> not leap
    assert is_leap(1800) is False
    # 1900: div by 100, not 400 -> not leap
    assert is_leap(1900) is False
    # 2100: div by 100, not 400 -> not leap
    assert is_leap(2100) is False
    # 2200: div by 100, not 400 -> not leap
    assert is_leap(2200) is False
    # 2300: div by 100, not 400 -> not leap
    assert is_leap(2300) is False


def test_century_leap_years():
    """Years divisible by 400 ARE leap years (the tricky rule)."""
    # 1600: div by 400 -> leap
    assert is_leap(1600) is True
    # 2000: div by 400 -> leap
    assert is_leap(2000) is True
    # 2400: div by 400 -> leap
    assert is_leap(2400) is True


def test_negative_years():
    """Negative years should also follow the same rules."""
    assert is_leap(-4) is True
    assert is_leap(-8) is True
    assert is_leap(-100) is False
    assert is_leap(-400) is True


def test_boundary_years():
    """Test around century boundaries for thoroughness."""
    # Just before 1700
    assert is_leap(1699) is False
    assert is_leap(1696) is True
    # Around 1800
    assert is_leap(1799) is False
    assert is_leap(1796) is True
    assert is_leap(1804) is True
    # Around 2000
    assert is_leap(1999) is False
    assert is_leap(2004) is True


def test_year_zero():
    """Year 0 (or 1 BC in astronomical year numbering)."""
    assert is_leap(0) is True


if __name__ == "__main__":
    # Run all tests manually if executed directly
    import sys

    failed = []
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS: {name}")
            except AssertionError as e:
                print(f"  FAIL: {name}: {e}")
                failed.append(name)

    if failed:
        print(f"\n{len(failed)} test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
```
