"""I retain the requested corpus compiler when bootstrap changes bin/nanoc."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def recipe(target):
    lines = (ROOT / "Makefile.gnu").read_text().splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith(target + ":"))
    result = []
    for line in lines[start + 1:]:
        if not line.startswith("\t"):
            break
        result.append(line)
    return "".join(result)


class TestCompilerSelection(unittest.TestCase):
    def check_selection(self, target, backend="c", fail=False):
        with tempfile.TemporaryDirectory(prefix="nano compiler selection ") as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "tests/unit").mkdir(parents=True)
            (root / "tests/user_guide").mkdir()
            shutil.copy2(ROOT / "tests/run_all_tests.sh", root / "tests/run_all_tests.sh")
            corpus = {"tests/nl_types_probe.nano", "tests/test_probe.nano",
                      "tests/unit/probe.nano", "tests/user_guide/probe.nano"}
            for source in corpus:
                (root / source).write_text("fn main() -> int { return 0 }\n")
            fake = '''#!/usr/bin/env python3
from pathlib import Path
import os, sys
name = Path(sys.argv[0]).name
with open("calls", "a") as log:
    log.write(name + " " + sys.argv[1] + "\\n")
if name == os.environ.get("FAIL_COMPILER"):
    sys.exit(17)
output = Path(sys.argv[sys.argv.index("-o") + 1])
output.write_text("#!/bin/sh\\nexit 0\\n")
output.chmod(0o700)
'''
            for name in ("nanoc_c", "nanoc_stage2", "nano_virt"):
                compiler = root / "bin" / name
                compiler.write_text(fake)
                compiler.chmod(0o700)
            vm = root / "bin/nano_vm"
            vm.write_text('#!/bin/sh\nfor arg do artifact="$arg"; done\nexec "$artifact"\n')
            vm.chmod(0o700)
            (root / ".bootstrap-ready").touch()
            expected = "nanoc_c" if target == "test" else "nanoc_stage2"
            mutated = "nanoc_stage2" if expected == "nanoc_c" else "nanoc_c"
            runner = next(line for line in recipe("test-impl").splitlines(keepends=True)
                          if "./tests/run_all_tests.sh" in line)
            makefile = '''COMPILER = bin/nanoc
COMPILER_C = bin/nanoc_c
NANOC_STAGE2 = bin/nanoc_stage2
SENTINEL_BOOTSTRAP3 = .bootstrap-ready
TEST_TIMEOUT = 20
TIMEOUT_CMD = perl -e 'alarm 20; exec @ARGV; die "I could not run make: $$!\\n"'
.PHONY: build shadow-check userguide-export bootstrap test-units test-impl test test-selfhosted test-bootstrap
build shadow-check userguide-export bootstrap:
\t@:
test-units:
\t@ln -sf MUTATED bin/nanoc
test-impl: test-units
'''.replace("MUTATED", mutated) + runner
            for name in ("test", "test-selfhosted", "test-bootstrap"):
                makefile += name + ": build\n" + recipe(name) + "\n"
            (root / "Makefile").write_text(makefile)
            env = dict(os.environ, NANOLANG_BACKEND=backend)
            env.pop("NANOLANG_COMPILER", None)
            env.pop("MAKEFLAGS", None)
            env.pop("MFLAGS", None)
            env.pop("MAKEOVERRIDES", None)
            if fail:
                env["FAIL_COMPILER"] = expected
            result = subprocess.run(["make", "--no-print-directory", target], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=25)
            self.assertEqual(result.returncode == 0, not fail, result.stdout + result.stderr)
            calls = (root / "calls").read_text().splitlines()
            wanted = "nano_virt" if backend in ("vm", "daemon") else expected
            self.assertEqual(set(calls), {wanted + " " + source for source in corpus})
            self.assertEqual(len(calls), len(corpus))

    def test_bootstrap_link_changes_do_not_select_the_corpus_compiler(self):
        for target in ("test", "test-selfhosted", "test-bootstrap"):
            with self.subTest(target=target):
                self.check_selection(target)

    def test_backend_override_retains_full_corpus(self):
        for backend in ("vm", "daemon"):
            with self.subTest(backend=backend):
                self.check_selection("test", backend)

    def test_selected_compiler_failure_is_not_hidden_by_the_link(self):
        for target in ("test", "test-selfhosted"):
            with self.subTest(target=target):
                self.check_selection(target, fail=True)


if __name__ == "__main__":
    unittest.main()
