"""Test memory.py: store/recall/supersede/decay on isolated temp databases.

    ~/miniforge3/envs/miniagent/bin/python -m unittest tests.test_memory -v
"""

import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import memory

DAY = 86400


def backdate(text_like, days):
    """Shift a memory's accessed time into the past (decay is age-based)."""
    c = sqlite3.connect(memory.DB)
    c.execute("UPDATE memories SET accessed = accessed - ? WHERE text LIKE ?",
              (days * DAY, f"%{text_like}%"))
    c.commit()
    c.close()


def rows(sql, *args):
    c = sqlite3.connect(memory.DB)
    out = c.execute(sql, args).fetchall()
    c.close()
    return out


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        memory.DB = Path(self.tmp) / "memory.db"
        memory._conn().close()                         # create schema; no-op tests query it too

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- store / recall basics ---

    def test_roundtrip(self):
        memory.store("the deploy bucket is zx-flumen-47", "fact")
        got = memory.recall("what is the deploy bucket?")
        self.assertEqual(got, [("fact", "the deploy bucket is zx-flumen-47")])

    def test_relevant_ranks_first_among_distractors(self):
        for i in range(20):
            memory.store(f"service worker-{i} listens on port {7000 + i}", "fact")
        memory.store("the staging gateway hostname is kumquat-lantern.internal", "fact")
        got = memory.recall("staging gateway hostname", k=3)
        self.assertIn("kumquat-lantern", got[0][1])

    def test_duplicate_touches_instead_of_inserting(self):
        memory.store("use black for formatting", "preference")
        memory.store("  Use   BLACK for formatting ", "preference")
        r = rows("SELECT hits FROM memories")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], 1)

    def test_empty_store_is_noop(self):
        memory.store("   ")
        self.assertEqual(rows("SELECT COUNT(*) FROM memories")[0][0], 0)

    def test_unknown_type_coerced_to_fact(self):
        memory.store("gravity points down", "banana")
        self.assertEqual(rows("SELECT type FROM memories")[0][0], "fact")

    def test_recall_bumps_hits_on_surfaced_memories(self):
        memory.store("the answer is fortytwo", "fact")
        memory.recall("what is the answer")
        self.assertEqual(rows("SELECT hits FROM memories")[0][0], 1)

    # --- query hygiene ---

    def test_unusable_queries_return_empty(self):
        memory.store("something is stored here", "fact")
        self.assertEqual(memory.recall("!!! ???"), [])       # no word chars
        self.assertEqual(memory.recall("a an of to"), [])    # all below MIN_TERM_LEN

    def test_fts_metacharacters_do_not_raise(self):
        memory.store('do not use eval() or exec("*")', "error_pattern")
        got = memory.recall('NEAR "eval*" (col:x) AND NOT')
        self.assertTrue(any("eval" in txt for _, txt in got))

    # --- supersede ---
    # BM25 relevance scales with corpus size (IDF ~0 in a tiny DB), so contradiction
    # detection needs background memories before it can cross SUPERSEDE_REL.

    def _seed(self, n=10):
        for i in range(n):
            memory.store(f"service worker-{i} listens on port {7000 + i}", "fact")

    def test_contradicted_preference_is_retired(self):
        self._seed()
        memory.store("always use tabs for indentation in python source files", "preference")
        memory.store("always use four spaces for indentation in python source files", "preference")
        sup = rows("SELECT superseded FROM memories WHERE text LIKE '%tabs%'")
        self.assertEqual(sup, [(1,)])
        got = memory.recall("indentation preference for python files")
        texts = [txt for _, txt in got]
        self.assertTrue(any("four spaces" in t for t in texts))
        self.assertFalse(any("tabs" in t for t in texts))

    def test_supersede_only_same_type(self):
        self._seed()
        memory.store("the project uses tabs for indentation in python files", "fact")
        memory.store("always use four spaces for indentation in python files", "preference")
        self.assertEqual(rows("SELECT COUNT(*) FROM memories WHERE superseded=1")[0][0], 0)

    # --- recency decay ---

    def test_stale_fact_sinks_below_fresh_fact(self):
        memory.store("the build server hostname is alpha", "fact")
        memory.store("the build server hostname is omega", "fact")
        backdate("alpha", days=60)
        got = memory.recall("build server hostname", k=2)
        self.assertIn("omega", got[0][1])
        self.assertIn("alpha", got[1][1])

    def test_preferences_do_not_decay(self):
        # equal BM25; if decay applied to the old preference its score would drop
        # to 1.2*0.35 = 0.42 of the fresh fact's and it would lose
        memory.store("run linting with the ruff checker", "preference")
        memory.store("run linting with the flake checker", "fact")
        backdate("ruff", days=60)
        got = memory.recall("linting checker", k=2)
        self.assertIn("ruff", got[0][1])

    def test_decayed_fact_still_recallable_at_floor(self):
        memory.store("the legacy cluster name is borealis", "fact")
        backdate("borealis", days=365)
        got = memory.recall("legacy cluster name")
        self.assertTrue(any("borealis" in txt for _, txt in got))

    # --- extract_and_store (stubbed chat; no LLM) ---

    def test_extract_stores_typed_items(self):
        stub = lambda *a, **k: {"content":
            '[{"type": "preference", "text": "prefers vim keybindings"},'
            ' {"type": "fact", "text": "repo lives on gitlab"}]'}
        memory.extract_and_store(stub, "u", "a")
        self.assertEqual(sorted(rows("SELECT type, text FROM memories")),
                         [("fact", "repo lives on gitlab"),
                          ("preference", "prefers vim keybindings")])

    def test_extract_tolerates_prose_wrapped_json(self):
        stub = lambda *a, **k: {"content":
            'Sure! Here you go: [{"type": "fact", "text": "port is 9090"}] Hope that helps.'}
        memory.extract_and_store(stub, "u", "a")
        self.assertEqual(rows("SELECT text FROM memories"), [("port is 9090",)])

    def test_extract_ignores_garbage_and_errors(self):
        memory.extract_and_store(lambda *a, **k: {"content": "no json here"}, "u", "a")
        memory.extract_and_store(lambda *a, **k: {"content": None}, "u", "a")

        def boom(*a, **k):
            raise RuntimeError("llm down")
        memory.extract_and_store(boom, "u", "a")
        self.assertEqual(rows("SELECT COUNT(*) FROM memories")[0][0], 0)

    def test_extract_skips_items_without_text(self):
        stub = lambda *a, **k: {"content": '[{"type": "fact"}, "loose string", '
                                           '{"type": "fact", "text": "kept item"}]'}
        memory.extract_and_store(stub, "u", "a")
        self.assertEqual(rows("SELECT text FROM memories"), [("kept item",)])


if __name__ == "__main__":
    unittest.main()
