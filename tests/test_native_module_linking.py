"""I link manifest objects once, before the libraries that satisfy them."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve()


class NativeModuleLinking(unittest.TestCase):
    def compile_fixture(self, duplicate_runtime):
        with tempfile.TemporaryDirectory(prefix="native_link_", dir=ROOT / "modules") as tmp:
            module = Path(tmp)
            source = module / "api.nano"
            source.write_text('module LinkFixture\nextern fn fixture_value() -> int\n'
                              'pub fn value() -> int { unsafe { return (fixture_value) } }\n'
                              'shadow value { assert (== (value) 188) }\n')
            bridge = '#include <openssl/sha.h>\n'
            if duplicate_runtime:
                bridge += '#include "cJSON.h"\n'
                expression = ('cJSON *value = cJSON_Parse("[1,2]"); '
                              'long long extra = cJSON_GetArraySize(value); cJSON_Delete(value);')
                extra_source = "../../src/./cJSON.c"
            else:
                bridge += 'long long fixture_extra(void);\n'
                expression = 'long long extra = fixture_extra();'
                extra_source = "cJSON.c"
                (module / extra_source).write_text('long long fixture_extra(void) { return 2; }\n')
            bridge += ('long long fixture_value(void) { unsigned char digest[SHA256_DIGEST_LENGTH]; '
                       'SHA256((const unsigned char *)"abc", 3, digest); ' + expression +
                       ' return digest[0] + extra; }\n')
            (module / "bridge.c").write_text(bridge)
            (module / "module.json").write_text(json.dumps({
                "name": module.name, "c_sources": ["bridge.c", extra_source],
                "include_dirs": ["src"], "pkg_config": ["openssl"],
            }))
            with tempfile.TemporaryDirectory(prefix="native-link-program-", dir=ROOT / "tests") as work:
                root = Path(work)
                program, binary = root / "main.nano", root / "main"
                program.write_text('module "' + str(source) + '" as fixture\n'
                                   'fn main() -> int { assert (== (fixture.value) 188) return 0 }\n'
                                   'shadow main { assert (== (main) 0) }\n')
                result = subprocess.run([COMPILER, program, "-o", binary], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                run = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_runtime_source_is_not_linked_twice_and_crypto_follows_objects(self):
        self.compile_fixture(True)

    def test_different_source_with_runtime_basename_is_retained(self):
        self.compile_fixture(False)


if __name__ == "__main__":
    unittest.main()
