"""I retain canonical guard effects and check explicit profile refusals."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("NANOLANG_GUARD_BIN", ROOT / "bin"))
PREFIX = "union Choice { Some { number: int }, None {} }\n"
TRACE = PREFIX + r'''let mut trace: int = 0
fn mark(n: int) -> int { set trace (+ (* trace 10) n) return n }
shadow mark { set trace 0 assert (== (mark 1) 1) assert (== trace 1) set trace 0 }
fn gate(n: int, answer: bool) -> bool { let ignored: int = (mark n) return answer }
shadow gate { set trace 0 assert (not (gate 2 false)) assert (== trace 2) set trace 0 }
fn fresh() -> Choice { let ignored: int = (mark 1) return Choice.Some { number: 7 } }
shadow fresh { set trace 0 let c: Choice = (fresh) assert (== trace 1) set trace 0 }
fn selected() -> int {
 return match (fresh) {
  None(e) if (gate 9 true) => (mark 9)
  Some(payload) if (gate 2 false) => (mark 8)
  Some(payload) if (and (== payload.number 7) (gate 3 true)) => (mark 4)
  Some(other) => (mark 8)
  None(e) => (mark 9)
 }
}
shadow selected { set trace 0 assert (== (selected) 4) assert (== trace 1234) set trace 0 }
fn main() -> int {
 set trace 0 assert (== (selected) 4) assert (== trace 1234)
 let payload: int = 23
 let c: Choice = Choice.Some { number: 7 }
 let n: int = match c { Some(payload) if (< payload.number 0) => 99 Some(payload) => payload.number None(empty) => 0 }
 assert (== n 7) assert (== payload 23)
 match c { Some(payload) if false => { assert false } Some(payload) => { assert (== payload.number 7) } None(empty) => { assert false } }
 assert (== payload 23)
 return 0
}
shadow main { assert (== (main) 0) }
'''
WILDCARDS = TRACE.split("fn selected()", 1)[0] + r'''fn select() -> int {
 return match (mark 1) {
  9 if (gate 9 true) => (mark 9)
  _ if (gate 2 false) => (mark 8)
  1 if (gate 3 false) => (mark 8)
  _ if (gate 4 true) => (mark 5)
  _ => (mark 9)
 }
}
shadow select { set trace 0 assert (== (select) 5) assert (== trace 12345) set trace 0 }
fn statement() -> int {
 let mut out: int = 0
 match (mark 1) { _ if (gate 2 false) => { set out (mark 8) } 1 if (gate 3 true) => { set out (mark 6) } _ => { set out (mark 9) } }
 return out
}
shadow statement { set trace 0 assert (== (statement) 6) assert (== trace 1236) set trace 0 }
fn early(n: int) -> int { let result: int = match n { 1 if true => { return 7 } _ => { 9 } } return (+ result 1) }
shadow early { assert (== (early 1) 7) assert (== (early 2) 10) }
fn looped() -> int {
 let mut n: int = 0
 while (< n 3) { set n (+ n 1) match n { 1 if true => { continue } 2 if true => { break } _ => { assert false } } assert false }
 return n
}
shadow looped { assert (== (looped) 2) }
fn main() -> int {
 set trace 0 assert (== (select) 5) assert (== trace 12345)
 set trace 0 assert (== (statement) 6) assert (== trace 1236)
 assert (== (early 1) 7) assert (== (early 2) 10) assert (== (looped) 2)
 return 0
}
shadow main { assert (== (main) 0) }
'''


class CanonicalMatchGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nano-canonical-match-guards-"))
        cls.serial = 0
        print(f"I retain guard artifacts at {cls.work}", flush=True)
        cls.env = dict(os.environ, LSAN_OPTIONS="", ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
                       UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        cls.cc = shlex.split(os.environ.get("CC", "cc"))

    def command(self, args, expected=0, timeout=180):
        type(self).serial += 1
        stem = self.work / f"command-{self.serial:04d}"
        args = list(map(str, args))
        stem.with_suffix(".json").write_text(json.dumps(args))
        try:
            result = subprocess.run(args, cwd=ROOT, env=self.env, capture_output=True,
                                    text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            for suffix, text in ((".stdout", error.stdout), (".stderr", error.stderr)):
                stem.with_suffix(suffix).write_bytes(text if isinstance(text, bytes) else (text or "").encode())
            stem.with_suffix(".status").write_text("124 timeout\n")
            raise
        stem.with_suffix(".stdout").write_text(result.stdout)
        stem.with_suffix(".stderr").write_text(result.stderr)
        stem.with_suffix(".status").write_text(str(result.returncode) + "\n")
        if expected is not None:
            self.assertEqual(result.returncode, expected, f"{args}\n{result.stdout}{result.stderr}")
        return result

    def source(self, name, text):
        path = self.work / (name + ".nano")
        path.write_text(text)
        return path

    def native(self, name, text):
        path = self.source(name, text)
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            output = self.work / f"{name}-{compiler}"
            self.command([BIN / compiler, path, "-o", output])
            self.command([output], timeout=15)

    def refuse(self, name, text, pattern, compilers=("nanoc_c", "nanoc_stage1", "nanoc_stage2"), bytecode=False):
        path = self.source(name, text)
        for compiler in compilers:
            output = self.work / f"{name}-{compiler}-prior"
            output.write_bytes(b"prior guard artifact")
            result = self.command([BIN / compiler, path, "-o", output] + (["--emit-nvm"] if bytecode else []), expected=None)
            self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
            self.assertRegex(result.stdout + result.stderr, pattern)
            self.assertNotRegex(result.stdout + result.stderr, r"(?i)(parse error|parsing failed|unexpected token|error: incompatible)")
            self.assertEqual(output.read_bytes(), b"prior guard artifact")

    def test_native_order_scope_and_control(self):
        self.native("named-order", TRACE)
        self.native("wildcard-order", WILDCARDS)

    def test_named_union_vm_and_sanitized_native(self):
        path = self.source("named-module", TRACE)
        for compiler in ("nano_virt", "nanoc_stage1", "nanoc_stage2"):
            module = self.work / f"named-{compiler}.nvm"
            self.command([BIN / compiler, path, "--emit-nvm", "-o", module])
            self.command([BIN / "nano_vm", "--verify-only", module])
            vm = self.command([BIN / "nano_vm", module])
            source = self.work / f"named-{compiler}.c"
            output = self.work / f"named-{compiler}-native"
            self.command([BIN / "nvm2c", module, "-o", source])
            self.command([*self.cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                          "-fno-omit-frame-pointer", source, "-lm", "-o", output])
            native = self.command([output])
            self.assertEqual(vm.stdout, "")
            self.assertEqual(native.stdout, vm.stdout)

    def test_checked_negatives_preserve_output(self):
        cases = {
            "bool": ("match 1 { 1 if 7 => 2 _ => 0 }", r"(?is)(bool.*guard|guard.*bool)"),
            "coverage": ("match 1 { 1 if true => 2 }", r"(?is)(E035|coverage|exhaustive)"),
            "unreachable": ("match 1 { _ if true => 2 1 => 0 }", r"(?is)(E036|reach.*arm)"),
            "domain": ('match "text" { _ => 1 }', r"(?is)(match.*(int|union)|(int|union).*match)"),
            "union-coverage": ("match Choice.Some { number: 1 } { Some(p) if (> p.number 0) => p.number None(n) => 0 }", r"(?is)(E035|coverage|exhaustive)"),
        }
        for name, (value, pattern) in cases.items():
            self.refuse(name, PREFIX + "fn main() -> int { return " + value + " } shadow main { assert true }", pattern)
        impure = PREFIX + 'let mut state: bool = true\npure fn choose(c: Choice) -> int { return match c { Some(p) if state => p.number Some(p) => 0 None(n) => 0 } } fn main() -> int { return 0 } shadow main { assert true }'
        self.refuse("guard-purity", impure, r"I cannot prove a closed empty effect summary for this pure fn")
        changing = 'resource struct Handle { fd: int } fn consume(h: Handle) -> bool { let Handle { fd } = h return (> fd 0) } shadow consume { assert (consume Handle { fd: 1 }) } fn choose(h: Handle) -> int { return match 1 { _ if (consume h) => 1 _ => 0 } } fn main() -> int { return 0 } shadow main { assert true }'
        self.refuse("guard-owner", changing, r"(?is)(ownership|resource)", ("nanoc_stage1", "nanoc_stage2"))

    def test_shadow_and_profile_refusals(self):
        bad = TRACE.replace('shadow gate { set trace 0 assert (not (gate 2 false)) assert (== trace 2) set trace 0 }', 'shadow gate { assert false }')
        self.refuse("shadow-native", bad, r"(?is)(shadow|assert)")
        self.refuse("shadow-bytecode", bad, r"(?is)(shadow|assert)", ("nanoc_stage1", "nanoc_stage2"), True)
        self.refuse("wildcard-bytecode", WILDCARDS, r"(?is)(match|union)", ("nanoc_stage1", "nanoc_stage2"), True)

    def test_unchecked_generated_terminal_backstops(self):
        source = self.source("backstop-driver", (ROOT / "tests/nanoisa/fixtures/match_guard_backstop_driver.nano.txt").read_text())
        driver = self.work / "backstop-driver"
        self.command([BIN / "nanoc_c", source, "-o", driver], timeout=600)
        for mode in ("expression", "statement"):
            text = self.command([driver, mode]).stdout
            generated = self.work / f"backstop-{mode}.c"
            generated.write_text(text)
            output = self.work / f"backstop-{mode}"
            self.command([*self.cc, "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                          "-fno-omit-frame-pointer", generated, "-o", output])
            self.command([output])
            missed = self.command([output, "miss"], expected=-signal.SIGABRT)
            self.assertIn("I reached no successful checked match arm", missed.stderr)
            self.assertNotIn("runtime error:", missed.stderr)
            self.assertNotIn("ERROR: AddressSanitizer", missed.stderr)

    def test_parser877_unchanged(self):
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            output = self.work / f"parser877-{compiler}"
            self.command([BIN / compiler, ROOT / "tests/parser_parenthesized.nano", "-o", output], timeout=600)
            self.command([output])


if __name__ == "__main__":
    unittest.main()
