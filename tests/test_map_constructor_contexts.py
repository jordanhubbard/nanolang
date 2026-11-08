"""I retain checked map tags at every constructor boundary."""
from pathlib import Path
import contextlib
import hashlib
import json
import subprocess
import os
import shlex
from tests.native_process import native_compile_timeout, native_link_flags, run_captured
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/unit/test_map_constructor_contexts.nano"
PRODUCERS = tuple(os.environ.get("NANO_MAP_CONTEXT_PROVIDERS",
    "nanoc_c,nano_virt,nanoc_stage1,nanoc_stage2").split(","))

class MapConstructorContexts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        allowed = {'nanoc_c', 'nano_virt', 'nanoc_stage1', 'nanoc_stage2'}
        if not PRODUCERS or len(set(PRODUCERS)) != len(PRODUCERS) or set(PRODUCERS)-allowed:
            raise AssertionError('I require explicit unique map source providers.')
        missing = [p for p in PRODUCERS if not (ROOT/'bin'/p).is_file()]
        if missing:
            raise AssertionError('I require actual map source providers: '+', '.join(missing))
        print('I exercise map source providers: '+', '.join(PRODUCERS), flush=True)

    def directory(self, label):
        output = ROOT/'.test_output/map-constructor-contexts'
        output.mkdir(parents=True, exist_ok=True)
        self.active_work = Path(tempfile.mkdtemp(prefix=label+'-', dir=output))
        return contextlib.nullcontext(self.active_work)

    def capture(self, command, *, accepted=True):
        argv = list(map(str, command))
        self.command_number = getattr(self, 'command_number', 0)+1
        label = 'command-'+str(self.command_number).zfill(4)
        build = Path(argv[0]).name in set(PRODUCERS)|{'nvm2c'} or '-std=c11' in argv
        budget = native_compile_timeout() if build else 120
        inputs = {}
        for value in argv:
            path = Path(value)
            if path.is_file() and (path.suffix == '.nano' or path.parent == ROOT/'bin'):
                inputs[value] = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {'argv': argv, 'build': build, 'timeout': budget, 'inputs': inputs}
        try:
            result = run_captured(argv, cwd=ROOT, text=True,
                                  timeout=120, build=build)
        except subprocess.TimeoutExpired as error:
            for stream, value in [('stdout', error.stdout), ('stderr', error.stderr)]:
                (self.active_work/(label+'.'+stream)).write_bytes(
                    value.encode() if isinstance(value,str) else value or b'')
            row.update(timed_out=True, returncode=None)
            with (self.active_work/'commands.jsonl').open('a') as log:
                log.write(json.dumps(row)+'\n')
            raise
        for stream, value in [('stdout', result.stdout), ('stderr', result.stderr)]:
            (self.active_work/(label+'.'+stream)).write_text(value)
        row.update(timed_out=False, returncode=result.returncode)
        with (self.active_work/'commands.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n')
        for path, before in inputs.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), before)
        if accepted:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result

    def run_checked(self, command):
        return self.capture(command)

    def vm_and_native(self, path, work, identity, producer):
        module = work / f"{identity}-{producer}.nvm"
        c_source = module.with_suffix('.c')
        native = work / f"{identity}-{producer}-native"
        self.run_checked([ROOT / "bin" / producer, path, "--emit-nvm", "-o", module])
        self.run_checked([ROOT / "bin/nano_vm", "--verify-only", module])
        self.run_checked([ROOT / "bin/nano_vm", module])
        self.run_checked([ROOT / "bin/nanoisa", "dump", module])
        self.run_checked([ROOT / "bin/nvm2c", module, "-o", c_source])
        self.run_checked([*shlex.split(os.environ.get("NANO_NATIVE_TEST_CC", "cc")),
            "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            c_source, ROOT / "bin/nano_aot_runtime.o", "-lm", *native_link_flags(),
            *(["-Wl,--export-dynamic", "-ldl"] if os.sys.platform.startswith("linux") else []),
            "-o", native])
        self.run_checked([native])

    def test_scalar_tag_pairs_and_contexts(self):
        baseline = SOURCE.read_text()
        with self.directory("contexts") as tmp:
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
                        path = work / f"case-{key}-{value}.nano"
                        path.write_text(source)
                        for producer in (p for p in PRODUCERS if p != "nanoc_c"):
                            with self.subTest(producer=producer):
                                self.vm_and_native(path, work, f"{key}-{value}", producer)
                        if 'nanoc_c' in PRODUCERS:
                            native = work/f"legacy-{key}-{value}"
                            self.run_checked([ROOT / "bin/nanoc_c", path, "-o", native])
                            self.run_checked([native])

    def test_detached_snapshots_and_clear_receiver_runs_once(self):
        for key in ("int", "string"):
            for value in ("int", "string"):
                with self.subTest(key=key, value=value), self.directory("contexts") as tmp:
                    work = Path(tmp)
                    first_key, second_key = ('-9223372036854775808', '9223372036854775807') if key == 'int' else ('"first"', '"second"')
                    first_value, second_value = ('-7', '19') if value == 'int' else ('"original"', '"changed"')
                    pair = f"HashMap<{key},{value}>"
                    path = work / 'snapshots.nano'
                    path.write_text(f'''let mut calls: int = 0
let shared: {pair} = (map_new)
fn receiver() -> {pair} {{ set calls (+ calls 1) return shared }}
shadow receiver {{
 set calls 0
 let alias: {pair} = (receiver)
 (map_put alias {first_key} {first_value})
 assert (map_has shared {first_key})
 assert (== calls 1)
 (map_clear shared)
}}
fn main() -> int {{
 (map_clear shared)
 (map_put shared {first_key} {first_value})
 let keys: array<{key}> = (map_keys shared)
 let items: array<{value}> = (map_values shared)
 (map_put shared {first_key} {second_value})
 (map_put shared {second_key} {second_value})
 set calls 0
 (map_clear (receiver))
 assert (== calls 1)
 assert (== (map_size shared) 0)
 assert (== (array_length keys) 1)
 assert (== (array_length items) 1)
 assert (== (at keys 0) {first_key})
 assert (== (at items 0) {first_value})
 let more_keys: array<{key}> = (array_push keys {second_key})
 let more_items: array<{value}> = (array_push items {second_value})
 assert (== (at more_keys 1) {second_key})
 assert (== (at more_items 1) {second_value})
 (map_put shared {first_key} {second_value})
 assert (== (map_get shared {first_key}) {second_value})
 set calls 0
 (map_clear shared)
 (map_clear (receiver))
 assert (== calls 1)
 assert (== (map_size shared) 0)
 return 0
}}
shadow main {{ assert (== (main) 0) }}
''')
                    for producer in (p for p in PRODUCERS if p != 'nanoc_c'):
                        with self.subTest(producer=producer):
                            self.vm_and_native(path, work, f"snapshots-{key}-{value}", producer)
                    if 'nanoc_c' in PRODUCERS:
                        self.run_checked([ROOT / 'bin/nanoc_c', path, '-o', work / 'seed-native'])
                        self.run_checked([work / 'seed-native'])

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
        with self.directory("contexts") as tmp:
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
            with self.subTest(type=typ, pattern=pattern), self.directory("contexts") as tmp:
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
        with self.directory("contexts") as tmp:
            work = Path(tmp)
            source, output = work / "bad.nano", work / "prior"
            for index, body in enumerate(bodies):
                source = work/f"bad-{index}.nano"
                source.write_text("fn main() -> int { " + body + " return 0 } shadow main { assert true }")
                for compiler in PRODUCERS:
                    with self.subTest(body=body, compiler=compiler):
                        output.write_text("prior artifact")
                        command = [ROOT / "bin" / compiler, source, "-o", output]
                        if compiler != "nanoc_c": command.append("--emit-nvm")
                        result = self.capture(command, accepted=False)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(output.read_text(), "prior artifact")

if __name__ == "__main__": unittest.main()
