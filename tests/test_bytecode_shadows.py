"""I execute shadows before publishing bytecode, not from production main."""
from pathlib import Path
import json
import hashlib
import os
import shutil
import signal
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BytecodeShadows(unittest.TestCase):
    def test_string_search_shadows_and_production(self):
        with tempfile.TemporaryDirectory(prefix="nano-search-vm-") as tmp:
            result, output = self.compile((ROOT / "tests/string_search.nano").read_text(), Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            product = self.execute(output)
            self.assertEqual(product.returncode, 0, product.stderr)
            self.assertEqual(product.stdout, b"search-ok\n")

    def test_import_selection_failures_preserve_output(self):
        for transitive in (False, True):
            for failure in ("assert false", 'let wrong: int = "no"', "assert (== (at [1] 9) 0)"):
                with self.subTest(transitive=transitive, failure=failure), tempfile.TemporaryDirectory(prefix="nano-import-shadows-") as tmp:
                    directory = Path(tmp)
                    leaf = directory / "leaf.nano"
                    leaf.write_text(f"pub fn answer() -> int {{ return 42 }}\nshadow answer {{ {failure} }}\n")
                    imported = leaf
                    if transitive:
                        imported = directory / "wrapper.nano"
                        imported.write_text(f'module "{leaf}" as lib\npub fn answer() -> int {{ return (lib.answer) }}\n'
                                            'shadow answer { assert (== (answer) 42) }\n')
                    source = f'module "{imported}" as lib\nfn main() -> int {{ return (lib.answer) }}\n'
                    ordinary, output = self.compile(source, directory, "--root-shadows-only")
                    self.assertEqual(ordinary.returncode, 0, ordinary.stderr)
                    prior = output.read_bytes()
                    selected, output = self.compile(source, directory)
                    self.assertNotEqual(selected.returncode, 0, selected.stderr)
                    self.assertEqual(output.read_bytes(), prior)

    def test_import_diamond_order_owners_and_product_separation(self):
        for declared in (False, True):
            with self.subTest(declared=declared), tempfile.TemporaryDirectory(prefix="nano-import-shadows-") as tmp:
                directory = Path(tmp)
                leaf = directory / "leaf.nano"
                leaf.write_text(('module Leaf\n' if declared else '') +
                                'pub fn answer() -> int { return 11 }\n'
                                'shadow answer { assert (== (answer) 11) (println "leaf-shadow") }\n')
                for name, value in (("left", 12), ("right", 13)):
                    (directory / f"{name}.nano").write_text(
                        (f'module {name.title()}\n' if declared else '') +
                        f'module "{leaf}" as lib\npub fn answer() -> int {{ return (+ (lib.answer) {value - 11}) }}\n'
                        f'shadow answer {{ assert (== (answer) {value}) assert (== (lib.answer) 11) (println "{name}-shadow") }}\n')
                source = (f'module "{directory / "left.nano"}" as left\n'
                          f'module "{directory / "right.nano"}" as right\n'
                          'fn main() -> int { return (+ (left.answer) (right.answer)) }\n'
                          'shadow main { assert (== (main) 25) (println "root-shadow") }\n')
                result, output = self.compile(source, directory, "--test-imports")
                self.assertEqual(result.returncode, 0, result.stderr)
                markers = [line for line in result.stderr.splitlines() if line.endswith(b"-shadow")]
                self.assertEqual(markers, [b"leaf-shadow", b"left-shadow", b"right-shadow", b"root-shadow"])
                self.assertNotIn(b"$shadow_", output.read_bytes())
                product = self.execute(output)
                self.assertEqual(product.returncode, 25, product.stderr)
                self.assertEqual(product.stdout, b"")

    def test_import_path_spelling_does_not_repeat_shadows(self):
        with tempfile.TemporaryDirectory(prefix="nano-import-shadows-") as tmp:
            directory = Path(tmp)
            leaf = directory / "leaf.nano"
            leaf.write_text('pub fn answer() -> int { return 42 }\n'
                            'shadow answer { assert (== (answer) 42) (println "leaf-shadow") }\n')
            link = directory / "link.nano"
            link.symlink_to(leaf)
            for alternate in (f"{directory}/./leaf.nano", str(link)):
                source = (f'module "{leaf}" as first\nmodule "{alternate}" as second\n'
                          'fn main() -> int { return (+ (first.answer) (second.answer)) }\n')
                result, output = self.compile(source, directory, "--test-imports")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr.splitlines().count(b"leaf-shadow"), 1)
                self.assertEqual(self.execute(output).returncode, 84)

    def test_imported_foreign_shadow_failure(self):
        with tempfile.TemporaryDirectory(prefix="nano-import-shadows-") as tmp:
            directory = Path(tmp)
            module_dir, _, env = self.foreign_build_fixture(directory)
            wrapper = directory / "wrapper.nano"
            for expected in (42, 41):
                wrapper.write_text(f'module "{module_dir / "api.nano"}" as lib\n'
                                   'pub fn answer() -> int { unsafe { return (lib.nano_build_answer) } }\n'
                                   f'shadow answer {{ assert (== (answer) {expected}) }}\n')
                source = f'module "{wrapper}" as lib\nfn main() -> int {{ return (lib.answer) }}\n'
                output = directory / "program.nvm"
                output.write_bytes(b"prior output")
                result, output = self.compile(source, directory, "--test-imports", env=env)
                self.assertEqual(result.returncode == 0, expected == 42, result.stderr)
                if expected == 42:
                    self.assertEqual(self.execute(output, env=env).returncode, 42)
                else:
                    self.assertEqual(output.read_bytes(), b"prior output")

    def test_imported_nonterminating_shadow_is_bounded(self):
        with tempfile.TemporaryDirectory(prefix="nano-import-shadows-") as tmp:
            directory = Path(tmp)
            leaf = directory / "leaf.nano"
            leaf.write_text("pub fn answer() -> int { return 42 }\nshadow answer { while true {} }\n")
            source = f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.answer) }}\n'
            result, output = self.compile(source, directory, "--test-imports")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"after 10 seconds", result.stderr)
            self.assertFalse(output.exists())

    def test_dependency_shadow_does_not_inherit_root_unsafe_context(self):
        with tempfile.TemporaryDirectory(prefix="nano-import-shadows-") as tmp:
            directory = Path(tmp)
            leaf = directory / "leaf.nano"
            for explicit in (False, True):
                call = "unsafe { (erf 0.0) }" if explicit else "(erf 0.0)"
                leaf.write_text('extern fn erf(value: float) -> float\npub fn answer() -> int { return 42 }\n'
                                f'shadow answer {{ {call} assert (== (answer) 42) }}\n')
                source = f'unsafe module "{leaf}" as lib\nfn main() -> int {{ return (lib.answer) }}\n'
                result, output = self.compile(source, directory, "--test-imports")
                self.assertEqual(result.returncode == 0, explicit, result.stderr)
                if not explicit:
                    self.assertIn(b"requires unsafe", result.stderr)
                    self.assertFalse(output.exists())

    def compile(self, source, directory, *options, env=None, cwd=ROOT):
        path = directory / "program.nano"
        path.write_text(source)
        output = directory / "program.nvm"
        args = [str(ROOT / "bin/nano_virt"), str(path), "--emit-nvm", "-o", str(output), *options]
        process = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True, env=env)
        try:
            stdout, stderr = process.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        return result, output

    def execute(self, output, env=None):
        return subprocess.run([str(ROOT / "bin/nano_vm"), str(output)], cwd=ROOT,
                              capture_output=True, timeout=10, env=env)

    def test_failed_shadow_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            directory = Path(tmp)
            output = directory / "program.nvm"
            output.write_bytes(b"preserve existing artifact")
            result, output = self.compile("fn f() -> int { return 42 }\nshadow f { assert false }\n"
                                          "fn main() -> int { return 0 }\nshadow main { assert true }\n", directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"shadow", result.stderr.lower())
            self.assertEqual(output.read_bytes(), b"preserve existing artifact")

    def test_main_shadow_does_not_recurse_or_leak_into_product(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile('fn main() -> int { (println "product") return 7 }\n'
                                          'shadow main { assert (== (main) 7) }\n', Path(tmp), "--run")
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(result.stdout, b"product\n")
            self.assertNotIn(b"$shadow_", output.read_bytes())
            execution = self.execute(output)
            self.assertEqual(execution.returncode, 7, execution.stderr)
            self.assertEqual(execution.stdout, b"product\n")

    def test_production_main_is_not_automatically_a_test(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile("fn f() -> int { return 42 }\nshadow f { assert (== (f) 42) }\n"
                                          "fn main() -> int { assert false return 0 }\nshadow main { assert true }\n", Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotEqual(self.execute(output).returncode, 0)

    def test_shadow_only_source(self):
        for assertion, success in (("(== (f) 42)", True), ("false", False)):
            with self.subTest(assertion=assertion), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                result, output = self.compile(f"fn f() -> int {{ return 42 }}\nshadow f {{ assert {assertion} }}\n", Path(tmp))
                self.assertEqual(result.returncode == 0, success, result.stderr)
                self.assertEqual(output.exists(), success)

    def test_shadow_locals_and_globals_are_not_product_state(self):
        source = '''let mut count: int = 0
fn f() -> int { return count }
shadow f { let x: int = 7 set count x assert (== (f) 7) }
fn g() -> int { return 5 }
shadow g { let x: string = "hello" assert (== (str_length x) (g)) }
fn main() -> int { return count }
shadow main { assert true }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_runtime_trap_blocks_publication(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile("fn f() -> int { return 0 }\n"
                                          "shadow f { let a: array<int> = [1] assert (== (at a 2) 0) }\n", Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"shadow", result.stderr.lower())
            self.assertFalse(output.exists())

    def test_nonterminating_shadow_is_bounded(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile("fn f() -> int { return 0 }\nshadow f { while true {} }\n", Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"after 10 seconds", result.stderr)
            self.assertFalse(output.exists())

    def test_root_shadow_calls_imported_helper(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            directory = Path(tmp)
            module = directory / "helper.nano"
            module.write_text("pub fn twice(x: int) -> int { return (* x 2) }\nshadow twice { assert true }\n")
            for expected in (6, 7):
                with self.subTest(expected=expected):
                    source = f'module "{module}" as helper\nfn main() -> int {{ return 0 }}\nshadow main {{ assert (== (helper.twice 3) {expected}) }}\n'
                    result, output = self.compile(source, directory)
                    self.assertEqual(result.returncode == 0, expected == 6, result.stderr)

    def test_foreign_code_cannot_cancel_parent_deadline(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            source = "extern fn alarm(seconds: int) -> int\nfn f() -> int { return 0 }\nshadow f { unsafe { (alarm 0) } while true {} }\n"
            result, output = self.compile(source, Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"after 10 seconds", result.stderr)
            self.assertFalse(output.exists())

    def test_foreign_module_source_directory(self):
        for cached in (False, True):
            for absolute in (False, True):
                with self.subTest(cached=cached, absolute=absolute), tempfile.TemporaryDirectory(prefix="nano-ffi-path-") as tmp:
                    directory = Path(tmp)
                    module_dir = directory / "nested"
                    module_dir.mkdir()
                    module = module_dir / "foreign.nano"
                    module.write_text("pub extern fn nano_shadow_answer() -> int\n")
                    module_path = str(module) if absolute else os.path.relpath(module, ROOT)
                    env = os.environ.copy()
                    env.pop("NANO_BUILD_CACHE", None)
                    if cached:
                        cache = directory / "cache"
                        env["NANO_BUILD_CACHE"] = str(cache)
                        key = hashlib.sha256(os.fsencode(module_dir.resolve())).hexdigest()
                        build_dir = cache / ("v2-" + key)
                    else:
                        build_dir = module_dir / ".build"
                    build_dir.mkdir(parents=True)
                    extension = "dylib" if sys.platform == "darwin" else "so"
                    library = build_dir / f"libforeign.{extension}"
                    built = subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(library)],
                                           input=b"#include <stdint.h>\nint64_t nano_shadow_answer(void) { return 42; }\n",
                                           capture_output=True, timeout=30)
                    self.assertEqual(built.returncode, 0, built.stderr)
                    for expected in (42, 41):
                        source = f'''module "{module_path}" as foreign
fn main() -> int {{ unsafe {{ return (foreign.nano_shadow_answer) }} }}
shadow main {{ assert (== (main) {expected}) }}
'''
                        output = directory / "program.nvm"
                        output.write_bytes(b"preserve")
                        result, output = self.compile(source, directory, "--run", env=env)
                        if expected == 42:
                            self.assertEqual(result.returncode, 42, result.stderr)
                            run = subprocess.run([str(ROOT / "bin/nano_vm"), str(output)], cwd=ROOT,
                                                 env=env, capture_output=True, timeout=10)
                            self.assertEqual(run.returncode, 42, run.stderr)
                        else:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn(b"shadow", result.stderr.lower())
                            self.assertNotIn(b"not found", result.stderr)
                            self.assertEqual(output.read_bytes(), b"preserve")

    def foreign_build_fixture(self, directory):
        module_dir = directory / "foreign"
        module_dir.mkdir()
        module = module_dir / "api.nano"
        module.write_text("pub extern fn nano_build_answer() -> int\n")
        (module_dir / "answer.c").write_text("#include <stdint.h>\nint64_t nano_build_answer(void) { return 42; }\n")
        (module_dir / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"]}))
        source = f'''module "{module}" as foreign
fn main() -> int {{ unsafe {{ return (foreign.nano_build_answer) }} }}
shadow main {{ assert (== (main) 42) }}
'''
        env = os.environ.copy()
        for name in ("NANO_BUILD_CACHE", "NANO_ALLOW_PACKAGE_INSTALL", "NANO_CC", "CC", "PKG_CONFIG"):
            env.pop(name, None)
        return module_dir, source, env

    def test_foreign_library_cold_and_warm_build(self):
        for cached in (False, True):
            for transitive in (False, True):
                with self.subTest(cached=cached, transitive=transitive), tempfile.TemporaryDirectory(prefix="nano-ffi-build-") as tmp:
                    directory = Path(tmp)
                    module_dir, source, env = self.foreign_build_fixture(directory)
                    env["NANO_VERBOSE_BUILD"] = "1"
                    if cached:
                        env["NANO_BUILD_CACHE"] = str(directory / "cache")
                    if transitive:
                        helper = directory / "helper.nano"
                        helper.write_text(f'''module "{module_dir / 'api.nano'}" as foreign
pub fn answer() -> int {{ unsafe {{ return (foreign.nano_build_answer) }} }}
shadow answer {{ assert (== (answer) 42) }}
''')
                        source = f'''module "{helper}" as helper
fn main() -> int {{ return (helper.answer) }}
shadow main {{ assert (== (main) 42) }}
'''
                    result, output = self.compile(source, directory, "--run", env=env)
                    self.assertEqual(result.returncode, 42, result.stderr)
                    execution = subprocess.run([str(ROOT / "bin/nano_vm"), str(output)], cwd=ROOT,
                                               env=env, capture_output=True, timeout=10)
                    self.assertEqual(execution.returncode, 42, execution.stderr)
                    result, output = self.compile(source, directory, "--run", env=env)
                    self.assertEqual(result.returncode, 42, result.stderr)
                    self.assertNotIn(b"[Module] Building answer_native", result.stdout)

    def test_foreign_library_build_failure_and_recovery(self):
        for failure in ("source", "shared_source", "link", "missing_library", "empty_library"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="nano-ffi-build-") as tmp:
                directory = Path(tmp)
                module_dir, source, env = self.foreign_build_fixture(directory)
                original_c = (module_dir / "answer.c").read_text()
                if failure == "source":
                    (module_dir / "answer.c").write_text("#error I reject this fixture\n")
                elif failure == "shared_source":
                    (module_dir / "private.c").write_text("#error I reject this private fixture\n")
                    (module_dir / "module.json").write_text(json.dumps({"name": "answer_native", "c_sources": ["answer.c"], "shared_c_sources": ["private.c"]}))
                else:
                    compiler = directory / "cc-fixture"
                    compiler.write_text(f'''#!{sys.executable}
import os, sys
if "-dynamiclib" in sys.argv or "-shared" in sys.argv:
    if {failure == 'empty_library'!r}:
        open(sys.argv[sys.argv.index("-o") + 1], "wb").close()
    sys.exit({24 if failure == 'link' else 0})
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
                    compiler.chmod(0o700)
                    env["NANO_CC"] = str(compiler)
                # I must reject the build even when no shadow calls the FFI.
                source = source.replace("shadow main { assert (== (main) 42) }", "shadow main { assert true }")
                output = directory / "program.nvm"
                output.write_bytes(b"preserve")
                result, output = self.compile(source, directory, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"could not build foreign support", result.stderr)
                self.assertEqual(output.read_bytes(), b"preserve")
                (module_dir / "answer.c").write_text(original_c)
                (module_dir / "private.c").write_text("int nano_private_fixture(void) { return 1; }\n")
                env.pop("NANO_CC", None)
                result, output = self.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)

    def test_foreign_invalid_manifest_preserves_output(self):
        for manifest in ("{", '{"c_sources": ["answer.c"]}'):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory(prefix="nano-ffi-build-") as tmp:
                directory = Path(tmp)
                module_dir, source, env = self.foreign_build_fixture(directory)
                (module_dir / "module.json").write_text(manifest)
                output = directory / "program.nvm"
                output.write_bytes(b"preserve")
                result, output = self.compile(source, directory, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_bytes(), b"preserve")

    def test_foreign_cache_same_timestamp_content_changes(self):
        for change in ("source", "header", "shared_source", "shared_header", "manifest"):
            with self.subTest(change=change), tempfile.TemporaryDirectory(prefix="nano-cache-") as tmp:
                directory = Path(tmp)
                module_dir, source, env = self.foreign_build_fixture(directory)
                changed = module_dir / "answer.c"
                manifest = {"name": "answer_native", "c_sources": ["answer.c"]}
                if change == "header":
                    changed = module_dir / "answer.h"
                    changed.write_text("#define ANSWER 42\n")
                    (module_dir / "answer.c").write_text('#include <stdint.h>\n#include "answer.h"\nint64_t nano_build_answer(void) { return ANSWER; }\n')
                elif change in ("shared_source", "shared_header"):
                    manifest["shared_c_sources"] = ["private.c"]
                    (module_dir / "answer.c").write_text('#include <stdint.h>\nint64_t private_answer(void);\nint64_t nano_build_answer(void) { return private_answer(); }\n')
                    changed = module_dir / "private.c"
                    changed.write_text('#include <stdint.h>\nint64_t private_answer(void) { return 42; }\n')
                    if change == "shared_header":
                        changed.write_text('#include <stdint.h>\n#include "private.h"\nint64_t private_answer(void) { return ANSWER; }\n')
                        changed = module_dir / "private.h"
                        changed.write_text("#define ANSWER 42\n")
                elif change == "manifest":
                    manifest["cflags"] = ["-DANSWER=42"]
                    changed.write_text('#include <stdint.h>\nint64_t nano_build_answer(void) { return ANSWER; }\n')
                    changed = module_dir / "module.json"
                (module_dir / "module.json").write_text(json.dumps(manifest))
                stamp = changed.stat()
                result, output = self.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                changed.write_text(changed.read_text().replace("42", "43"))
                os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                self.assertEqual(changed.stat().st_mtime_ns, stamp.st_mtime_ns)
                result, output = self.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
                self.assertEqual(result.returncode, 43, result.stderr)
                execution = self.execute(output, env=env)
                self.assertEqual(execution.returncode, 43, execution.stderr)

    def test_foreign_cache_requires_readable_hash_record(self):
        for damage in ("missing", "corrupt", "directory", "legacy_context", "malformed_context"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory(prefix="nano-cache-") as tmp:
                directory = Path(tmp)
                module_dir, source, env = self.foreign_build_fixture(directory)
                compiler = directory / "cc-fixture"
                fail = directory / "fail-compilation"
                compiler.write_text(f'''#!{sys.executable}
import os, pathlib, sys
if pathlib.Path({str(fail)!r}).exists(): sys.exit(24)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
                compiler.chmod(0o700)
                env["NANO_CC"] = str(compiler)
                result, output = self.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                cache = module_dir / ".build" / "current" / "source_hashes.json"
                if damage in ("legacy_context", "malformed_context"):
                    record = json.loads(cache.read_text())
                    if damage == "legacy_context":
                        record.pop("__build_context_v1")
                    else:
                        record["__build_context_v1"] = []
                    cache.write_text(json.dumps(record))
                elif damage == "missing":
                    cache.unlink()
                elif damage == "corrupt":
                    cache.write_text("{")
                elif damage == "directory":
                    cache.unlink()
                    cache.mkdir()
                fail.touch()  # I inject failure without changing the compiler identity.
                output.write_bytes(b"preserve")
                result, output = self.compile(source, directory, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"could not build foreign support", result.stderr)
                self.assertEqual(output.read_bytes(), b"preserve")
                fail.unlink()
                if damage == "directory":
                    cache.rmdir()
                result, output = self.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)

    def test_foreign_cache_failed_link_retry(self):
        with tempfile.TemporaryDirectory(prefix="nano-cache-") as tmp:
            directory = Path(tmp)
            module_dir, source, env = self.foreign_build_fixture(directory)
            changed = module_dir / "answer.c"
            stamp = changed.stat()
            result, output = self.compile(source, directory, "--run", env=env)
            self.assertEqual(result.returncode, 42, result.stderr)
            changed.write_text(changed.read_text().replace("42", "43"))
            os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            compiler = directory / "cc-fixture"
            compiler.write_text(f'''#!{sys.executable}
import os, sys
if "-dynamiclib" in sys.argv or "-shared" in sys.argv:
    sys.exit(24)
os.execv({shutil.which('cc')!r}, [{shutil.which('cc')!r}] + sys.argv[1:])
''')
            compiler.chmod(0o700)
            env["NANO_CC"] = str(compiler)
            output.write_bytes(b"preserve")
            result, output = self.compile(source.replace(" 42)", " 43)"), directory, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"could not build foreign support", result.stderr)
            self.assertEqual(output.read_bytes(), b"preserve")
            env.pop("NANO_CC")
            result, output = self.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
            self.assertEqual(result.returncode, 43, result.stderr)
            execution = self.execute(output, env=env)
            self.assertEqual(execution.returncode, 43, execution.stderr)

    def test_local_opaque_roundtrip(self):
        source = '''opaque type LocalHandle
struct Box { value: int }
fn box() -> Box { return Box { value: 42 } }
shadow box { let item = (box) assert (== item.value 42) }
extern fn malloc(size: int) -> LocalHandle
extern fn free(ptr: LocalHandle) -> void
fn allocate() -> LocalHandle { unsafe { return (malloc 8) } }
shadow allocate { let p = (allocate) unsafe { (free p) } }
fn identity(p: LocalHandle) -> LocalHandle { return p }
shadow identity { let p = (allocate) unsafe { (free (identity p)) } }
fn main() -> int {
    let p = (allocate)
    unsafe { (free (identity p)) }
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-opaque-") as tmp:
            result, output = self.compile(source, Path(tmp), "--run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_imported_opaque_roundtrip_and_wrong_kind(self):
        for transitive in (False, True):
            with self.subTest(transitive=transitive), tempfile.TemporaryDirectory(prefix="nano-opaque-") as tmp:
                directory = Path(tmp)
                module_dir, _, env = self.foreign_build_fixture(directory)
                (module_dir / "answer.c").write_text('''#include <stdint.h>
static int64_t answer = 42;
void *nano_handle_new(void) { return &answer; }
void *nano_handle_echo(void *p) { return p; }
int64_t nano_handle_read(void *p) { return p == &answer ? answer : -1; }
''')
                (module_dir / "api.nano").write_text('''opaque type ForeignHandle
extern fn nano_handle_new() -> ForeignHandle
extern fn nano_handle_echo(p: ForeignHandle) -> ForeignHandle
extern fn nano_handle_read(p: ForeignHandle) -> int
pub fn make() -> ForeignHandle { unsafe { return (nano_handle_new) } }
shadow make { assert (== (read (make)) 42) }
pub fn echo(p: ForeignHandle) -> ForeignHandle { unsafe { return (nano_handle_echo p) } }
shadow echo { assert (== (read (echo (make))) 42) }
pub fn read(p: ForeignHandle) -> int { unsafe { return (nano_handle_read p) } }
shadow read { assert (== (read (make)) 42) }
''')
                imported = module_dir / "api.nano"
                if transitive:
                    helper = directory / "helper.nano"
                    helper.write_text(f'''module "{imported}" as foreign
pub fn make_indirect() -> foreign.ForeignHandle {{ return (foreign.make) }}
shadow make_indirect {{ assert (== (read_indirect (make_indirect)) 42) }}
pub fn echo_indirect(p: foreign.ForeignHandle) -> foreign.ForeignHandle {{ return (foreign.echo p) }}
shadow echo_indirect {{ assert (== (read_indirect (echo_indirect (make_indirect))) 42) }}
pub fn read_indirect(p: foreign.ForeignHandle) -> int {{ return (foreign.read p) }}
shadow read_indirect {{ assert (== (read_indirect (make_indirect)) 42) }}
''')
                    imported = helper
                source = f'''module "{imported}" as handles
fn local_echo(p: handles.ForeignHandle) -> handles.ForeignHandle {{ return (handles.echo p) }}
shadow local_echo {{ assert (== (handles.read (local_echo (handles.make))) 42) }}
fn main() -> int {{ return (handles.read (local_echo (handles.make))) }}
shadow main {{ assert (== (main) 42) }}
'''
                if transitive:
                    for name in ("make", "echo", "read"):
                        source = source.replace(f"handles.{name}", f"handles.{name}_indirect")
                result, output = self.compile(source, directory, "--run", env=env)
                self.assertEqual(result.returncode, 42, result.stderr)
                run = subprocess.run([str(ROOT / "bin/nano_vm"), str(output)], cwd=ROOT,
                                     env=env, capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 42, run.stderr)
                for wrong in ('(handles.read "wrong")', '(local_echo "wrong")'):
                    if transitive:
                        wrong = wrong.replace("handles.read", "handles.read_indirect")
                    output.write_bytes(b"preserve")
                    result, output = self.compile(source.replace('(== (main) 42)', f'(== {wrong} 42)'), directory, env=env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn(b"UNDEFINED FUNCTION", result.stderr)
                    self.assertEqual(output.read_bytes(), b"preserve")

    def test_entry_exit_values(self):
        for value, expected in ((0, 0), (7, 7), (-1, 255), (256, 0)):
            with self.subTest(value=value), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                result, output = self.compile(f"fn main() -> int {{ return {value} }}\nshadow main {{ assert true }}\n", Path(tmp), "--run")
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(self.execute(output).returncode, expected)

    def test_shadow_array_inference(self):
        source = '''fn first(words: array<string>) -> string { return (at words 0) }
shadow first {
    let words = ["nano", "lang"]
    assert (== (first words) "nano")
    let empty: array<string> = []
    assert (== (array_length empty) 0)
    let values = [-1, 2, 3]
    assert (== (+ (at values 0) (at values 1)) 1)
}
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_invalid_shadow_types_preserve_output(self):
        for body in ('assert 7', 'let x: int = "wrong" assert true',
                     'unsafe { assert 7 }', 'return 7', 'break',
                     'let a: array<int> = ["wrong"] assert true'):
            with self.subTest(body=body), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                directory = Path(tmp)
                (directory / "program.nvm").write_bytes(b"preserve")
                result, output = self.compile('fn main() -> int { return 0 }\n'
                                              f'shadow main {{ {body} }}\n', directory)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn(b"type check failed", result.stderr)
                self.assertEqual(output.read_bytes(), b"preserve")

    def test_absolute_value_preserves_numeric_kind(self):
        source = '''fn magnitude(value: float) -> float { return (abs value) }
shadow magnitude {
    assert (== (magnitude -3.5) 3.5)
    assert (== (magnitude 3.5) 3.5)
    assert (== (magnitude 0.0) 0.0)
}
fn integer(value: int) -> int { return (abs value) }
shadow integer {
    assert (== (integer -7) 7)
    assert (== (integer 7) 7)
    assert (== (integer 0) 0)
    let minimum: int = (- (- 0 9223372036854775807) 1)
    assert (== (integer minimum) minimum)
}
fn main() -> int {
    assert (== (abs -3.14) 3.14)
    assert (== (magnitude -9.5) 9.5)
    assert (== (integer -9) 9)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_array_literal_annotations_in_functions(self):
        bodies = ('let a: array<int> = ["wrong"]',
                  'let mut a: array<string> = [] set a [1]',
                  'let a: array<float> = [true]',
                  'let a: array<string> = [1, 2]')
        for compiler in ("nanoc_c", "nano_virt"):
            for body in bodies:
                with self.subTest(compiler=compiler, body=body), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                    directory = Path(tmp)
                    source = directory / "input.nano"
                    source.write_text(f"fn main() -> int {{ {body} return 0 }}\nshadow main {{ assert true }}\n")
                    output = directory / "output"
                    output.write_bytes(b"preserve")
                    args = [str(ROOT / "bin" / compiler), str(source), "-o", str(output)]
                    if compiler == "nano_virt":
                        args.append("--emit-nvm")
                    result = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=25)
                    self.assertNotEqual(result.returncode, 0, result.stderr)
                    self.assertIn(b"array elements", result.stderr)
                    self.assertEqual(output.read_bytes(), b"preserve")

    def test_matching_and_empty_array_literals(self):
        source = '''fn main() -> int {
    let mut words: array<string> = []
    set words ["nano"]
    assert (== (at words 0) "nano")
    let values: array<float> = [1.5, 2.5]
    assert (== (at values 1) 2.5)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            directory = Path(tmp)
            result, output = self.compile(source, directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)
            native = directory / "program.native"
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(directory / "program.nano"),
                                     "-o", str(native)], cwd=ROOT, capture_output=True, timeout=25)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(subprocess.run([str(native)], capture_output=True, timeout=10).returncode, 0)

    def test_min_max_values_and_evaluation_order(self):
        checks = []
        for left, right in ((2, 7), (7, 2), (2, 2), (-7, -2), (-2, -7),
                            (2.5, 7.8), (7.8, 2.5), (2.5, 2.5), (-7.8, -2.5), (-2.5, -7.8)):
            for operation in (min, max):
                checks.append(f"assert (== ({operation.__name__} {left} {right}) {operation(left, right)})")
        source = '''let mut calls: int = 0
fn next(value: int) -> int { set calls (+ (* calls 10) value) return value }
shadow next { set calls 0 assert (== (next 2) 2) assert (== calls 2) set calls 0 }
fn main() -> int {
''' + '\n'.join(checks) + '''
    set calls 0
    assert (== (min (next 2) (next 1)) 1)
    assert (== calls 21)
    set calls 0
    assert (== (max (next 1) (next 2)) 2)
    assert (== calls 12)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_shadow_local_type_overrides_target_parameter_metadata(self):
        source = '''fn f(value: int) -> int { return value }
shadow f {
    let value: float = -3.5
    assert (== (abs value) 3.5)
}
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            result, output = self.compile(source, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.execute(output).returncode, 0)

    def test_nested_function_factory_signatures(self):
        source = '''fn add(a: int, b: int) -> int { return (+ a b) }
shadow add { assert (== (add 2 3) 5) }
fn factory() -> fn(int, int) -> int { return add }
shadow factory { let op: fn(int, int) -> int = (factory) assert (== (op 2 3) 5) }
fn apply(build: fn() -> fn(int, int) -> int) -> int {
    let op: fn(int, int) -> int = (build)
    return (op 2 3)
}
shadow apply { assert (== (apply factory) 5) }
fn main() -> int { return (apply factory) }
shadow main { assert (== (main) 5) }
'''
        with tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
            for attempt in range(5):
                with self.subTest(attempt=attempt):
                    result, output = self.compile(source, Path(tmp))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(self.execute(output).returncode, 5)

    def test_bad_function_signatures_reject_without_crashing(self):
        mismatch = '''fn add(a: int, b: int) -> int { return (+ a b) }
fn factory() -> fn(int, int) -> int { return add }
fn accept(build: fn() -> fn(int, int) -> float) -> int { return 0 }
fn main() -> int { return (accept factory) }
shadow main { assert true }
'''
        for source in (mismatch, "fn f(value: fn(int) ->", "fn f(value: fn(int"):
            with self.subTest(source=source), tempfile.TemporaryDirectory(prefix="nano-shadows-") as tmp:
                directory = Path(tmp)
                (directory / "program.nvm").write_bytes(b"preserve")
                result, output = self.compile(source, directory)
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertEqual(output.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
