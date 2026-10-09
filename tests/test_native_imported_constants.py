"""I preserve qualified literal identity in C-seed shadows and native output."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeImportedConstants(unittest.TestCase):
    def check(self, source, files=None):
        with tempfile.TemporaryDirectory(prefix="nano-native-imported-constants-") as tmp:
            work = Path(tmp)
            (work / "a.nano").write_text('''pub let answer: int = 41
pub let label: string = "first"
pub let enabled: bool = true
pub let fraction: float = 1.25
pub let minimum: int = -9223372036854775808
''')
            (work / "b.nano").write_text('''pub let answer: int = 17
pub let label: string = "second"
pub let enabled: bool = false
pub let fraction: float = 2.5
''')
            source = 'module "a.nano" as first\nmodule "b.nano" as second\n' + source
            for name, content in (files or {}).items():
                (work / name).write_text(content)
            (work / "main.nano").write_text(source)
            environment = {**os.environ, "NANO_BUILD_CACHE": str(work / "cache")}
            for command in ([ROOT / "bin/nanoc_c", work / "main.nano", "-o", work / "main"],
                            [work / "main"]):
                result = subprocess.run(list(map(str, command)), cwd=ROOT, env=environment,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_distinct_module_literals(self):
        self.check('''fn main() -> int {
    assert (== first.answer 41)
    assert (== second.answer 17)
    assert (== first.label "first")
    assert (== second.label "second")
    assert first.enabled
    assert (not second.enabled)
    assert (== first.fraction 1.25)
    assert (== second.fraction 2.5)
    assert (== first.minimum -9223372036854775808)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_local_record_receiver(self):
        self.check('''struct Record { answer: int, label: string }
fn main() -> int {
    let first: Record = Record { answer: 9, label: "local" }
    assert (== first.answer 9)
    assert (== first.label "local")
    assert (== second.answer 17)
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_transitive_import_retains_callee_source(self):
        self.check('''module "b.nano" as source
module "bridge.nano" as bridge
from "bridge.nano" import read as imported_read
fn main() -> int {
    assert (== source.answer 17)
    assert (== (bridge.read) 41)
    let read: fn() -> int = imported_read
    assert (== (read) 41)
    assert (== source.answer 17)
    return 0
}
shadow main { assert (== (main) 0) }
''', files={"bridge.nano": '''module "a.nano" as source
pub fn read() -> int { return source.answer }
shadow read { assert (== (read) 41) }
'''})

    def test_call_argument_order(self):
        self.check('''let mut visits: int = 0
fn next() -> int { set visits (+ visits 1) return visits }
shadow next { set visits 0 assert (== (next) 1) set visits 0 }
fn combine(a: int, b: int, c: int, d: int) -> int {
    return (+ (+ (* a 1000) (* b 100)) (+ (* c 10) d))
}
shadow combine { assert (== (combine 1 2 3 4) 1234) }
fn main() -> int {
    set visits 0
    assert (== (combine (next) first.answer (next) second.answer) 5137)
    assert (== visits 2)
    set visits 0
    return 0
}
shadow main { assert (== (main) 0) }
''')


if __name__ == "__main__":
    unittest.main()
