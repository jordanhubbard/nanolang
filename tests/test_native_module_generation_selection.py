"""I select one foreign generation per module directory and native link."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeModuleGenerationSelection(unittest.TestCase):
    def test_long_link_closure_keeps_every_foreign_object(self):
        with tempfile.TemporaryDirectory(prefix="nano-long-link-") as tmp:
            work = Path(tmp)
            imports = []
            calls = []
            for index in range(8):
                module = work / f"foreign{index}"
                module.mkdir()
                name = f"link_value_{index}"
                (module / "module.json").write_text(json.dumps({
                    "name": f"foreign{index}", "c_sources": ["foreign.c"],
                }))
                (module / "foreign.c").write_text(
                    f"#include <stdint.h>\nint64_t {name}(void) {{ return {index + 1}; }}\n")
                (module / f"api{index}.nano").write_text(f"pub extern fn {name}() -> int\n")
                imports.append(f'unsafe module "foreign{index}/api{index}.nano" as m{index}')
                calls.append(f"assert (== (m{index}.{name}) {index + 1})")
            source = work / "main.nano"
            source.write_text("\n".join(imports) + "\nfn main() -> int { unsafe { " +
                              " ".join(calls) + " } return 0 }\nshadow main { assert true }\n")
            env = os.environ.copy()
            env.pop("NANO_CC", None)
            env.pop("CC", None)
            cache = work / ("cache_" + "x" * 150) / ("nested_" + "y" * 150)
            env["NANO_BUILD_CACHE"] = str(cache)
            # I exercise both initial publication and the cached generation path.
            for _ in range(2):
                result = subprocess.run(
                    [str(ROOT / "bin/nanoc_c"), str(source), "-o", str(work / "program")],
                    cwd=ROOT, env=env, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                objects = list(cache.glob("*/.nano-gen-*/*.o"))
                self.assertEqual(len(objects), 8)
                self.assertGreater(sum(len(str(obj)) + 3 for obj in objects), 2048)
                subprocess.run([str(work / "program")], check=True, timeout=10)

    def test_oversized_compile_flags_fail_without_silent_truncation(self):
        with tempfile.TemporaryDirectory(prefix="nano-long-cflags-") as tmp:
            work = Path(tmp)
            module = work / "foreign"
            module.mkdir()
            # A shell-form driver keeps flags inline instead of response transport.
            (module / "module.json").write_text(json.dumps({
                "name": "long_flags", "c_compiler": "cc ", "c_sources": ["foreign.c"],
                "cflags": [f"-DLONG_COMPILE_FLAG_{index:04d}=1" for index in range(120)],
            }))
            (module / "foreign.c").write_text("#include <stdint.h>\nint64_t probe(void) { return 42; }\n")
            (module / "foreign.nano").write_text("pub extern fn probe() -> int\n")
            source = work / "main.nano"
            source.write_text('unsafe module "foreign/foreign.nano" as f\n'
                              'fn main() -> int { unsafe { assert (== (f.probe) 42) } return 0 }\nshadow main { assert true }\n')
            env = os.environ.copy()
            env["NANO_BUILD_CACHE"] = str(work / "cache")
            result = subprocess.run(
                [str(ROOT / "bin/nanoc_c"), str(source), "-o", str(work / "program")],
                cwd=ROOT, env=env, capture_output=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"I could not represent all module compile flags.", result.stderr)
            self.assertFalse((work / "program").exists())

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
