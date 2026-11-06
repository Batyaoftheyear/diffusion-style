"""Диагностика subprocess и восстановление настоящего окружения без pip."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv

from src.bootstrap import DEFAULT_CACHE, VIRTUALENV_VERSION, ensure_environment, environment_python, run_command


class BootstrapTests(unittest.TestCase):
    def test_stderr_is_visible_and_included_in_error(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "real-bootstrap-error"):
            run_command([sys.executable, "-c", "import sys; print('real-bootstrap-error', file=sys.stderr); sys.exit(7)"])
        self.assertIn("real-bootstrap-error", output.getvalue())

    def test_system_environment_is_rejected(self):
        with self.assertRaises(ValueError):
            ensure_environment(sys.prefix)

    @unittest.skipUnless((DEFAULT_CACHE / f"virtualenv-{VIRTUALENV_VERSION}.pyz").exists(), "requires checksum-verified bootstrap archive in the local cache")
    def test_real_partial_environment_is_repaired_and_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "env"
            venv.EnvBuilder(with_pip=False).create(directory)
            python = environment_python(directory)
            missing = subprocess.run([str(python), "-I", "-m", "pip", "--version"], capture_output=True)
            self.assertNotEqual(missing.returncode, 0)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ensure_environment(directory), python)
            check = subprocess.run([str(python), "-I", "-c", "import sys,pip; assert sys.prefix != sys.base_prefix; print(pip.__version__)"], capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            with patch("src.bootstrap.get_virtualenv", side_effect=AssertionError("Usable environment should be reused")):
                self.assertEqual(ensure_environment(directory), python)


if __name__ == "__main__":
    unittest.main()
