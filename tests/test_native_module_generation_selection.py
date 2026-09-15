"""I select one foreign generation per module directory and native link."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeModuleGenerationSelection(unittest.TestCase):
    def test_multiple_sources_share_one_uncacheable_foreign_build(self):
        with tempfile.TemporaryDirectory(prefix="nano-link-generation-") as tmp:
            work = Path(tmp)
            module = work / "foreign"
            module.mkdir()
            # A shell-form compiler deliberately has no reusable tool identity.
            # Every redundant build publishes a different immutable generation.
            (module / "module.json").write_text(json.dumps({
                "name": "foreign", "c_compiler": "cc ",
                "c_sources": ["foreign.c"], "headers": ["foreign.h"],
            }))
            (module / "foreign.h").write_text(
                "#include <stdint.h>\nint64_t first(void);\nint64_t second(void);\n")
            (module / "foreign.c").write_text(
                '#include "foreign.h"\nint64_t first(void) { return 20; }\n'
                'int64_t second(void) { return 22; }\n')
            (module / "first.nano").write_text("pub extern fn first() -> int\n")
            (module / "second.nano").write_text("pub extern fn second() -> int\n")
            source = work / "main.nano"
            source.write_text(
                'unsafe module "foreign/first.nano" as a\n'
                'unsafe module "foreign/second.nano" as b\n'
                'fn main() -> int { unsafe { assert (== (+ (a.first) (b.second)) 42) } return 0 }\n'
                'shadow main { assert true }\n')
            env = os.environ.copy()
            env.pop("NANO_CC", None)
            env.pop("CC", None)
            env["NANO_BUILD_CACHE"] = str(work / "cache")
            for attempt in range(2):
                result = subprocess.run(
                    [str(ROOT / "bin/nanoc_c"), str(source), "-o", str(work / "program")],
                    cwd=ROOT, env=env, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                subprocess.run([str(work / "program")], check=True, timeout=10)
                generations = list((work / "cache").glob("*/.nano-gen-*"))
                self.assertEqual(len(generations), attempt + 1)


if __name__ == "__main__":
    unittest.main()
