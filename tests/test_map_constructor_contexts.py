"""I retain checked map tags at every constructor boundary."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/unit/test_map_constructor_contexts.nano"

class MapConstructorContexts(unittest.TestCase):
    def run_checked(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_scalar_tag_pairs_and_contexts(self):
        baseline = SOURCE.read_text()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for key in ("int", "string"):
                for value in ("int", "string"):
                    with self.subTest(key=key, value=value):
                        source = baseline.replace("HashMap<string,string>", f"HashMap<{key},{value}>")
                        for name in ("before", "grown", "keys"):
                            source = source.replace(f"{name}: array<string>", f"{name}: array<{key}>")
                        for name in ("items", "before_values", "grown_values"):
                            source = source.replace(f"{name}: array<string>", f"{name}: array<{value}>")
                        if key == "int":
                            source = source.replace('"new"', "11").replace('"answer"', "12")
                        if value == "int":
                            source = source.replace('"kept"', "42")
                        path = work / "case.nano"
                        path.write_text(source)
                        self.run_checked([ROOT / "bin/nano_virt", path, "--emit-nvm", "-o", work / "case.nvm"])
                        self.run_checked([ROOT / "bin/nano_vm", work / "case.nvm"])
                        self.run_checked([ROOT / "bin/nanoc_c", path, "-o", work / "native"])
                        self.run_checked([work / "native"])

    def test_scoped_aliases_and_return_evaluation(self):
        source = """union Choice { One {}, Two {} }
fn returned(flag: bool) -> HashMap<string,string> {
 let owned: HashMap<string,string> = (map_new)
 (map_put owned "answer" "kept")
 if flag { return owned }
 assert (== (map_size owned) 1)
 return owned
}
shadow returned {
 let first: HashMap<string,string> = (returned true)
 let second: HashMap<string,string> = (returned false)
 assert (== (map_get first "answer") "kept")
 assert (== (map_get second "answer") "kept")
}
fn scalar_return(flag: bool) -> int {
 let owned: HashMap<string,string> = (map_new)
 (map_put owned "answer" "kept")
 if flag { return (map_size owned) }
 return (+ 1 (map_size owned))
}
shadow scalar_return { assert (== (scalar_return true) 1) assert (== (scalar_return false) 2) }
fn borrowed_return(flag: bool) -> string {
 let owned: HashMap<string,string> = (map_new)
 (map_put owned "answer" "kept")
 if flag { return (map_get owned "answer") }
 return (map_get owned "answer")
}
shadow borrowed_return { assert (== (borrowed_return true) "kept") assert (== (borrowed_return false) "kept") }
fn scoped(choice: Choice, owner: HashMap<string,string>) -> int {
 match choice {
  One(first) => { let alias: HashMap<string,string> = owner (map_put alias "answer" "kept") }
  Two(second) => { let alias: HashMap<string,string> = owner (map_put alias "answer" "kept") }
 }
 return (map_size owner)
}
shadow scoped { let owner: HashMap<string,string> = (map_new) assert (== (scoped Choice.One {} owner) 1) assert (== (scoped Choice.Two {} owner) 1) }
fn main() -> int {
 let owner: HashMap<string,string> = (map_new)
 assert (== (scoped Choice.One {} owner) 1)
 assert (== (map_get owner "answer") "kept")
 assert (== (scoped Choice.Two {} owner) 1)
 assert (== (map_get owner "answer") "kept")
 assert (== (scalar_return true) 1)
 assert (== (scalar_return false) 2)
 assert (== (borrowed_return true) "kept")
 assert (== (borrowed_return false) "kept")
 let first: HashMap<string,string> = (returned true)
 let second: HashMap<string,string> = (returned false)
 assert (== (map_get first "answer") "kept")
 assert (== (map_get second "answer") "kept")
 return 0
}
shadow main { assert (== (main) 0) }
"""
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            path = work / "scopes.nano"
            path.write_text(source)
            self.run_checked([ROOT / "bin/nanoc_c", path, "-o", work / "native"])
            self.run_checked([work / "native"])
            self.run_checked([ROOT / "bin/nano_virt", path, "--emit-nvm", "-o", work / "case.nvm"])
            self.run_checked([ROOT / "bin/nano_vm", work / "case.nvm"])

    def test_match_pattern_scopes(self):
        cases = (
            ("int", "0", "1", "0", "_"),
            ("int", "0", "1", "0 if true", "_"),
            ("Choice", "Choice.One {}", "Choice.Two {}", "One | Two", "_"),
        )
        for typ, first, second, pattern, fallback in cases:
            with self.subTest(type=typ, pattern=pattern), tempfile.TemporaryDirectory() as tmp:
                source = 'union Choice { One {}, Two {} }\n'
                source += f'fn probe(value: {typ}) -> int {{ let mut answer: int = 0 match value {{\n'
                source += pattern + ' => { let selected_map: HashMap<string,string> = (map_new) (map_put selected_map "answer" "kept") set answer (map_size selected_map) }\n'
                source += fallback + ' => { let fallback_map: HashMap<string,string> = (map_new) (map_put fallback_map "answer" "kept") set answer (map_size fallback_map) }\n'
                source += '} return answer }\n'
                source += f'shadow probe {{ assert (== (probe {first}) 1) assert (== (probe {second}) 1) }}\n'
                source += f'fn main() -> int {{ assert (== (probe {first}) 1) assert (== (probe {second}) 1) return 0 }}\n'
                source += 'shadow main { assert (== (main) 0) }\n'
                work = Path(tmp)
                path = work / 'patterns.nano'
                path.write_text(source)
                self.run_checked([ROOT / 'bin/nanoc_c', path, '-o', work / 'native'])
                self.run_checked([work / 'native'])

    def test_owned_opaque_arm_cleanup_occurs_at_scope_exit(self):
        from tests.test_native_effect_execution import NativeEffectExecution
        NativeEffectExecution().run_program("""
union Choice { One {}, Two {} }
opaque type Owned
extern fn nl_owned_create() -> Owned
extern fn nl_owned_count() -> int
fn visit(choice: Choice) -> void {
 unsafe {
  let before: int = (nl_owned_count)
  match choice {
   One(first) => { let item: Owned = (nl_owned_create) assert (== (nl_owned_count) before) }
   Two(second) => { let item: Owned = (nl_owned_create) assert (== (nl_owned_count) before) }
  }
  assert (== (nl_owned_count) (+ before 1))
 }
}
shadow visit { assert true }
fn main() -> int {
 let mut i: int = 0
 while (< i 10) { (visit Choice.One {}) (visit Choice.Two {}) set i (+ i 1) }
 unsafe { assert (== (nl_owned_count) 20) }
 return 0
}
shadow main { assert true }
""", foreign="""
#include "runtime/gc.h"
static int64_t finalized;
static void finish_owned(void *value) { (void)value; ++finalized; }
static inline void *nl_owned_create(void) { return gc_alloc_opaque(8, finish_owned); }
static inline int64_t nl_owned_count(void) { return finalized; }
""")

    def test_missing_or_unsupported_context_preserves_artifact(self):
        bodies = (
            "let values = (map_new)",
            "let values: HashMap<bool,string> = (map_new)",
            "let values: HashMap<string,bool> = (map_new)",
            "let values: HashMap<string,string> = (map_new 1)",
        )
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source, output = work / "bad.nano", work / "prior"
            for body in bodies:
                source.write_text("fn main() -> int { " + body + " return 0 } shadow main { assert true }")
                for compiler in ("nanoc_c", "nano_virt"):
                    with self.subTest(body=body, compiler=compiler):
                        output.write_text("prior artifact")
                        command = [ROOT / "bin" / compiler, source, "-o", output]
                        if compiler == "nano_virt": command.append("--emit-nvm")
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(output.read_text(), "prior artifact")

if __name__ == "__main__": unittest.main()
