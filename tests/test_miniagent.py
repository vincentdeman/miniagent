"""Test miniagent.py tool gating: run_bash always confirms, write_file on overwrite.

    ~/miniforge3/envs/miniagent/bin/python -m unittest tests.test_miniagent -v
"""

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import miniagent

NO_PROMPT = AssertionError("tool prompted when it should not have")


def quiet(fn, *args):
    """Run a tool swallowing its confirmation banner prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


class GatingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.target = self.tmp / "target.txt"
        miniagent.AUTO_CONFIRM = False

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- run_bash: every command needs consent ---

    def test_run_bash_declined_does_not_execute(self):
        with patch("builtins.input", return_value="n"):
            out = quiet(miniagent.run_bash, f"touch {self.target}")
        self.assertIn("declined", out)
        self.assertFalse(self.target.exists())

    def test_run_bash_accepted_executes(self):
        with patch("builtins.input", return_value="y"):
            quiet(miniagent.run_bash, f"touch {self.target}")
        self.assertTrue(self.target.exists())

    def test_run_bash_auto_confirm_skips_prompt(self):
        miniagent.AUTO_CONFIRM = True
        with patch("builtins.input", side_effect=NO_PROMPT):
            quiet(miniagent.run_bash, f"touch {self.target}")
        self.assertTrue(self.target.exists())

    # --- write_file: creating is silent, overwriting needs consent ---

    def test_write_new_file_does_not_prompt(self):
        with patch("builtins.input", side_effect=NO_PROMPT):
            out = quiet(miniagent.write_file, str(self.target), "created")
        self.assertIn("wrote", out)
        self.assertEqual(self.target.read_text(), "created")

    def test_overwrite_declined_preserves_file(self):
        self.target.write_text("original")
        with patch("builtins.input", return_value="n"):
            out = quiet(miniagent.write_file, str(self.target), "clobbered")
        self.assertIn("declined", out)
        self.assertEqual(self.target.read_text(), "original")

    def test_overwrite_accepted_replaces_file(self):
        self.target.write_text("original")
        with patch("builtins.input", return_value="y"):
            quiet(miniagent.write_file, str(self.target), "replaced")
        self.assertEqual(self.target.read_text(), "replaced")

    def test_overwrite_auto_confirm_skips_prompt(self):
        miniagent.AUTO_CONFIRM = True
        self.target.write_text("original")
        with patch("builtins.input", side_effect=NO_PROMPT):
            quiet(miniagent.write_file, str(self.target), "replaced")
        self.assertEqual(self.target.read_text(), "replaced")

    def test_overwrite_without_stdin_fails_closed(self):
        self.target.write_text("original")
        with patch("builtins.input", side_effect=EOFError):
            out = quiet(miniagent.write_file, str(self.target), "clobbered")
        self.assertIn("error", out)
        self.assertEqual(self.target.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
