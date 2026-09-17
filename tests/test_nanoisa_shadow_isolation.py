"""I execute my facade shadows concurrently in private fixture directories."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NanoisaShadowIsolation(unittest.TestCase):
    def test_concurrent_import_shadows_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="nanoisa-shadow-check-") as tmp:
            directory = Path(tmp)
            fixtures = directory / "fixtures"
            fixtures.mkdir()
            sentinel = fixtures / "unrelated"
            sentinel.write_text("preserve me")
            source = directory / "probe.nano"
            source.write_text('''from "modules/nanoisa/nanoisa.nano" import last_error
fn main() -> int { return 0 }
shadow main { assert true }
''')
            env = dict(os.environ, TMPDIR=str(fixtures))

            def compile_probe(index):
                result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source),
                                         "-o", str(directory / f"probe-{index}")],
                                        cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            # Warm module cache before concurrent compilation isolates shadow races.
            compile_probe("warm")
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(compile_probe, range(4)))
            self.assertEqual(sentinel.read_text(), "preserve me")
            self.assertEqual(list(fixtures.iterdir()), [sentinel])
            # An unusable requested temp root must reject the shadow run, rather
            # than silently using shared legacy paths elsewhere.
            unavailable = dict(env, TMPDIR=str(directory / "missing-temp-root"))
            rejected = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source),
                                       "-o", str(directory / "rejected")],
                                      cwd=ROOT, env=unavailable, capture_output=True,
                                      text=True, timeout=120)
            self.assertNotEqual(rejected.returncode, 0, rejected.stdout + rejected.stderr)
            self.assertFalse((directory / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
