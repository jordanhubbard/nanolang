"""I check dependency shadows by default and retain one graph-wide report."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CseedImportShadows(unittest.TestCase):
    def test_private_shadow_calls_fail_before_publication(self):
        for dependency in (False, True):
            for binding in (False, True):
                with self.subTest(dependency=dependency, binding=binding), tempfile.TemporaryDirectory(prefix="nano-shadow-private-") as tmp:
                    directory = Path(tmp)
                    leaf = directory / "secret.nano"
                    leaf.write_text('module Secret\nfn hidden() -> int { return 42 }\nshadow hidden { assert (== (hidden) 42) }\n')
                    body = 'let ignored: int = (hidden)' if binding else '(hidden)'
                    owner = (f'module Reader\nimport "{leaf}"\n'
                             'pub fn answer() -> int { return 0 }\n'
                             f'shadow answer {{ {body} }}\n')
                    if dependency:
                        reader = directory / "reader.nano"
                        reader.write_text(owner)
                        source = f'module "{reader}" as reader\nfn main() -> int {{ return (reader.answer) }}\nshadow main {{ assert (== (main) 0) }}\n'
                    else:
                        source = owner + 'fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'
                    output = directory / "program"
                    output.write_bytes(b"prior artifact")
                    result, output, report = self.compile(directory, source, "--json-errors")
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(b"private", result.stderr)
                    diagnostics = json.loads(result.stdout)["diagnostics"]
                    self.assertTrue(any(item["code"] == "E009" and "private" in item["message"]
                                        for item in diagnostics), diagnostics)
                    self.assertEqual(output.read_bytes(), b"prior artifact")
                    self.assertFalse(report.exists(), "I must reject before executing shadows")
                    leaf.write_text(leaf.read_text().replace('fn hidden()', 'pub fn hidden()'))
                    result, output, report = self.compile(directory, source)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    ran = subprocess.run([str(output)], capture_output=True, timeout=10)
                    self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def compile(self, directory, source, *options):
        path = directory / "main.nano"
        path.write_text(source)
        output = directory / "program"
        report = directory / "shadows.json"
        result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(path), "-o", str(output),
                                 "--llm-shadow-json", str(report), *options], cwd=ROOT,
                                capture_output=True, timeout=60)
        return result, output, report

    def test_default_failure_and_explicit_root_only(self):
        for transitive in (False, True):
            with self.subTest(transitive=transitive), tempfile.TemporaryDirectory(prefix="nano-cseed-shadows-") as tmp:
                directory = Path(tmp)
                leaf = directory / "leaf.nano"
                leaf.write_text("pub fn answer() -> int { return 42 }\nshadow answer { assert false }\n")
                imported = leaf
                if transitive:
                    imported = directory / "middle.nano"
                    imported.write_text(f'module "{leaf}" as lib\npub fn answer() -> int {{ return (lib.answer) }}\n'
                                        'shadow answer { assert (== (answer) 42) }\n')
                source = f'module "{imported}" as lib\nfn main() -> int {{ return (lib.answer) }}\nshadow main {{ assert (== (main) 42) }}\n'
                result, output, report = self.compile(directory, source, "--root-shadows-only")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                prior = output.read_bytes()
                result, output, report = self.compile(directory, source)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_bytes(), prior)
                data = json.loads(report.read_text())
                self.assertFalse(data["success"])
                self.assertEqual(data["test_count"], 3 if transitive else 2)
                self.assertEqual(len(data["failures"]), 1)
                self.assertEqual(Path(data["failures"][0]["source_file"]).name, "leaf.nano")

    def test_diamond_order_owner_and_aggregate_report(self):
        for declared in (False, True):
            with self.subTest(declared=declared), tempfile.TemporaryDirectory(prefix="nano-cseed-shadows-") as tmp:
                directory = Path(tmp)
                leaf = directory / "leaf.nano"
                leaf.write_text(('module Leaf\n' if declared else '') +
                                'pub fn answer() -> int { return 11 }\nshadow answer { assert (== (answer) 11) assert false }\n')
                link = directory / "link.nano"
                link.symlink_to(leaf)
                for name, value, dependency in (("left", 12, leaf), ("right", 13, link)):
                    (directory / f"{name}.nano").write_text((f'module {name.title()}\n' if declared else '') +
                        f'module "{dependency}" as lib\npub fn answer() -> int {{ return (+ (lib.answer) {value - 11}) }}\n'
                        f'shadow answer {{ assert (== (answer) {value}) assert (== (lib.answer) 11) assert false }}\n')
                source = (f'module "{directory / "left.nano"}" as left\nmodule "{directory / "right.nano"}" as right\n'
                          'fn main() -> int { return (+ (left.answer) (right.answer)) }\n'
                          'shadow main { assert (== (main) 25) assert false }\n')
                result, output, report = self.compile(directory, source, "--test-imports")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
                data = json.loads(report.read_text())
                self.assertEqual(data["test_count"], 4)
                self.assertEqual([Path(f["source_file"]).name for f in data["failures"]],
                                 ["leaf.nano", "left.nano", "right.nano", "main.nano"])
                self.assertEqual([f["fail_count"] for f in data["failures"]], [1, 1, 1, 1])

    def test_imported_types_checked_before_execution(self):
        with tempfile.TemporaryDirectory(prefix="nano-cseed-shadows-") as tmp:
            directory = Path(tmp)
            leaf = directory / "leaf.nano"
            leaf.write_text('pub fn answer() -> int { return 42 }\nshadow answer { let wrong: int = "no" }\n')
            source = f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.answer) }}\nshadow main {{ assert (== (main) 42) }}\n'
            result, output, report = self.compile(directory, source)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())

    def test_indirect_shadow_calls_use_declared_signatures(self):
        for returned in (False, True):
            for arguments in ("0.0", "", '"wrong"', "0.0 1.0"):
                with self.subTest(returned=returned, arguments=arguments), tempfile.TemporaryDirectory(prefix="nano-indirect-shadow-") as tmp:
                    directory = Path(tmp)
                    setup = "let op: fn(float) -> float = identity "
                    call = f"(op {arguments})"
                    if returned:
                        setup = ""
                        call = f"((choose) {arguments})"
                    source = ('fn identity(x: float) -> float { return x }\n'
                              'shadow identity { assert (== (identity 0.0) 0.0) }\n'
                              'fn choose() -> fn(float) -> float { return identity }\n'
                              'shadow choose { assert (== ((choose) 0.0) 0.0) }\n'
                              'fn main() -> int { return 0 }\n'
                              f'shadow main {{ {setup}assert (== {call} 0.0) }}\n')
                    result, output, report = self.compile(directory, source)
                    if arguments == "0.0":
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertTrue(json.loads(report.read_text())["success"])
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertFalse(output.exists())
                        self.assertFalse(report.exists())

    def test_zero_argument_function_value_is_called(self):
        with tempfile.TemporaryDirectory(prefix="nano-zero-indirect-") as tmp:
            source = ('fn value() -> float { return 1.5 }\nshadow value { assert (== (value) 1.5) }\n'
                      'fn main() -> int { return 0 }\n'
                      'shadow main { let op: fn() -> float = value assert (== (op) 1.5) }\n')
            result, output, report = self.compile(Path(tmp), source)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(report.read_text())["success"])

    def test_source_only_does_not_execute_dependency_shadows(self):
        with tempfile.TemporaryDirectory(prefix="nano-cseed-shadows-") as tmp:
            directory = Path(tmp)
            leaf = directory / "leaf.nano"
            leaf.write_text('pub fn answer() -> int { return 42 }\nshadow answer { assert false }\n')
            source = f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.answer) }}\nshadow main {{ assert (== (main) 42) }}\n'
            result, output, report = self.compile(directory, source, "--target", "c")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists())
            self.assertFalse(report.exists())

    def test_shadow_locals_do_not_change_production_lowering(self):
        with tempfile.TemporaryDirectory(prefix="nano-shadow-state-") as tmp:
            directory = Path(tmp)
            leaf = directory / "leaf.nano"
            leaf.write_text('pub fn answer() -> int { return 42 }\n'
                            'shadow answer { let s: int = 9 assert (== s 9) }\n')
            source = (f'module "{leaf}" as lib\n'
                      'fn message() -> string { let mut s: string = "value" set s (+ s "!") return s }\n'
                      'shadow message { let s: int = 9 assert (== s 9) assert (== (message) "value!") }\n'
                      'fn main() -> int { (println (message)) return 0 }\n'
                      'shadow main { assert (== (lib.answer) 42) }\n')
            result, output, report = self.compile(directory, source)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(report.read_text())["success"])
            run = subprocess.run([str(output)], capture_output=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout, b"value!\n")

    def test_hanging_or_crashing_dependency_preserves_output(self):
        for body in ("while true { }", "let values: array<int> = [1] (at values 2)",
                     "unsafe { (abort) }", "unsafe { (nano_shadow_exit_zero) }"):
            with self.subTest(body=body), tempfile.TemporaryDirectory(prefix="nano-shadow-supervisor-") as tmp:
                directory = Path(tmp)
                module_dir = directory / "foreign"
                module_dir.mkdir()
                (module_dir / "exit.c").write_text('#include <stdlib.h>\nvoid nano_shadow_exit_zero(void) { exit(0); }\n')
                (module_dir / "module.json").write_text(json.dumps({"name": "shadow_exit", "c_sources": ["exit.c"]}))
                leaf = module_dir / "leaf.nano"
                leaf.write_text(f'extern fn abort() -> void\nextern fn nano_shadow_exit_zero() -> void\n'
                                f'pub fn answer() -> int {{ return 42 }}\nshadow answer {{ {body} }}\n')
                source = f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.answer) }}\n'
                output = directory / "program"
                output.write_bytes(b"prior artifact")
                (directory / "shadows.json").write_text('{"success":true,"test_count":1}')
                result, output, report = self.compile(directory, source)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b"prior artifact")
                self.assertIn(b"shadow", result.stderr.lower())
                data = json.loads(report.read_text())
                self.assertFalse(data["success"], result.stdout + result.stderr)
                self.assertFalse(data["completed"])
                self.assertIsNone(data["test_count"])
                if "while" in body:
                    self.assertIn(b"10 seconds", result.stderr)

    def test_filesystem_shadows_use_private_fixtures_outside_repository(self):
        with tempfile.TemporaryDirectory(prefix="nano-cseed-fs-") as tmp:
            directory = Path(tmp)
            scratch = directory / "scratch"
            scratch.mkdir()
            source = directory / "main.nano"
            source.write_text(f'module "{ROOT / "modules/std/fs.nano"}" as fs\n'
                              'fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n')

            def compile_one(index):
                return subprocess.run([str(ROOT / "bin/nanoc_c"), str(source), "-o", str(directory / f"program-{index}"),
                                       "--llm-shadow-json", str(directory / f"report-{index}.json")],
                                      cwd=directory, env=dict(os.environ, TMPDIR=str(scratch)),
                                      capture_output=True, timeout=120)

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(compile_one, range(2)))
            for index, result in enumerate(results):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((directory / f"report-{index}.json").read_text())
                self.assertTrue(report["success"])
                self.assertGreater(report["test_count"], 10)
            self.assertEqual(list(scratch.glob("nano_fs_*")), [])


if __name__ == "__main__":
    unittest.main()
