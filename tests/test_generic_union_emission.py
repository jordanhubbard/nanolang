"""I keep concrete generic source identities across declaration-level union IDs."""
import os
from pathlib import Path
import subprocess
import sys
from tests import test_scalar_match_values as matches

ROOT = Path(__file__).resolve().parents[1]


class GenericUnionEmission(matches.ScalarMatchValues):
    def test_concrete_instances_context_and_formal_shadowing(self):
        self.paired('''resource struct T { fd: int }
union Box<T> { Some { value: T }, None {} }
union Choice<T,E> { Left { value: T }, Right { value: E } }
fn fresh() -> Box<int> { return Box.Some { value: 7 } }
shadow fresh { assert (== (integer (fresh)) 7) }
fn integer(value: Box<int>) -> int { match value { Some(v) => { return v.value } None(n) => { return 0 } } }
shadow integer { assert (== (integer Box.Some { value: 7 }) 7) }
fn text(value: Box<string>) -> string { match value { Some(v) => { return v.value } None(n) => { return "" } } }
shadow text { assert (== (text Box.Some { value: "kept" }) "kept") }
fn flag(value: Box<bool>) -> bool { match value { Some(v) => { return v.value } None(n) => { return false } } }
shadow flag { assert (flag Box.Some { value: true }) }
fn real(value: Box<float>) -> float { match value { Some(v) => { return v.value } None(n) => { return 0.0 } } }
shadow real { assert (== (real Box.Some { value: 1.5 }) 1.5) }
fn second(value: Choice<int,string>) -> string { match value { Left(v) => { return (int_to_string v.value) } Right(v) => { return v.value } } }
shadow second { assert (== (second Choice.Right { value: "kept" }) "kept") }
fn main() -> int {
 let left: Box<int> = (fresh)
 let right: Box<string> = Box.Some { value: "kept" }
 let copy: Box<int> = left
 assert (== (integer copy) 7) assert (== (integer left) 7)
 assert (== (text right) "kept") assert (== (text Box.None {}) "")
 assert (flag Box.Some { value: true }) assert (not (flag Box.None {}))
 assert (== (real Box.Some { value: 1.5 }) 1.5)
 assert (== (second Choice.Left { value: 7 }) "7")
 assert (== (second Choice.Right { value: "kept" }) "kept")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_array_payload_and_phantom_arguments(self):
        self.paired('''union Box<T> { Some { value: T }, None {} }
union Phantom<T> { Some { value: int }, None {} }
fn array_value(boxed: Box<array<int>>) -> array<int> { match boxed { Some(v) => { return v.value } None(n) => { return [0] } } }
shadow array_value { let boxed: Box<array<int>> = Box.Some { value: [7,8] } let values: array<int> = (array_value boxed) assert (== (at values 1) 8) }
fn phantom(boxed: Phantom<array<int>>) -> int { match boxed { Some(v) => { return v.value } None(n) => { return 0 } } }
shadow phantom { assert (== (phantom Phantom.Some { value: 7 }) 7) }
fn main() -> int {
 let boxed: Box<array<int>> = Box.Some { value: [7,8] }
 let copy: Box<array<int>> = boxed
 let first: array<int> = (array_value copy)
 let second: array<int> = (array_value boxed)
 assert (== (+ (at first 0) (at second 1)) 15)
 assert (== (phantom Phantom.Some { value: 7 }) 7)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_instance_mismatch_refusals_preserve_output(self):
        prefix = 'union Box<T> { Some { value: T }, None {} } union Other<T> { Some { value: T }, None {} } '
        cases = {
            'wrong_local': 'let value: Box<int> = Box.Some { value: "bad" }',
            'wrong_call': 'let wrong: Box<string> = Box.None {} let result: int = (accept wrong)',
            'wrong_return_call': 'let value: Box<int> = (wrong_route)',
            'wrong_return_value': 'let value: Box<int> = (wrong_route)',
            'wrong_declaration': 'let value: Box<int> = Other.Some { value: 1 }',
            'missing_arguments': 'let value: Box = Box.Some { value: 1 }',
            'extra_arguments': 'let value: Box<int,string> = Box.Some { value: 1 }',
            'unsupported_arguments': 'let value: Box<array<string>> = Box.Some { value: ["bad"] }',
            'missing_field': 'let value: Box<int> = Box.Some {}',
            'duplicate_field': 'let value: Box<int> = Box.Some { value: 1, value: 2 }',
        }
        for name, body in cases.items():
            source = self.work / ('bad-generic-' + name + '.nano')
            extra = ''
            if name == 'wrong_return_call':
                extra = 'fn wrong_result() -> Box<string> { return Box.None {} } shadow wrong_result { assert true } fn wrong_route() -> Box<int> { return (wrong_result) } shadow wrong_route { assert true } '
            if name == 'wrong_return_value':
                extra = 'fn wrong_route() -> Box<int> { let wrong: Box<string> = Box.None {} return wrong } shadow wrong_route { assert true } '
            source.write_text(prefix + extra + 'fn accept(value: Box<int>) -> int { return 0 } shadow accept { assert true } fn main() -> int { ' + body + ' return 0 } shadow main { assert true }')
            for tool in [*self.raw, ROOT/'bin/nanoc_stage1', ROOT/'bin/nanoc_stage2']:
                output = self.work/'prior-generic-output'
                output.write_text('retained')
                args = [tool, source]
                if tool not in self.raw:
                    args.append('--emit-nvm')
                result = subprocess.run([*args, '-o', output], cwd=ROOT, text=True, capture_output=True, timeout=180)
                self.assertGreater(result.returncode, 0, (name, tool, result.stdout, result.stderr))
                self.assertEqual(output.read_text(), 'retained')
                self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse error')
                self.assertRegex(result.stdout + result.stderr, r'(?i)(union|generic|type|field|argument|constructor declaration)')

    def test_unchanged_affine_suites_through_explicit_nanoisa(self):
        # I retain every existing source and assertion. This test-only adapter
        # requests canonical bytecode, checks VM execution, then publishes the
        # same module as native output for the suites' ordinary executable API.
        wrappers = self.work/'canonical-tools'
        wrappers.mkdir(exist_ok=True)
        script = wrappers/'driver'
        script.write_text('''#!/usr/bin/env python3
import os, pathlib, subprocess, sys, tempfile
root = pathlib.Path(os.environ['NANOLANG_GENERIC_TEST_ROOT'])
compiler = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
position = args.index('-o') + 1
output = pathlib.Path(args[position])
with tempfile.TemporaryDirectory(prefix='generic-canonical-', dir=output.parent) as directory:
    work = pathlib.Path(directory)
    module, c_source, native = work/'program.nvm', work/'program.c', work/'program'
    args[position] = str(module)
    commands = [[str(root/'bin'/compiler), *args, '--emit-nvm'],
                [str(root/'bin/nano_vm'), '--verify-only', str(module)],
                [str(root/'bin/nano_vm'), str(module)],
                [str(root/'bin/nvm2c'), str(module), '-o', str(c_source)],
                [os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(c_source), '-lm', '-o', str(native)]]
    for command in commands:
        result = subprocess.run(command)
        if result.returncode:
            sys.exit(result.returncode if result.returncode > 0 else 1)
    os.replace(native, output)
''')
        script.chmod(0o755)
        for compiler in ('nanoc_stage1', 'nanoc_stage2'):
            (wrappers/compiler).symlink_to('driver')
        result = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'tests.test_affine_module_identity', 'tests.test_affine_generic_identity'],
                                cwd=ROOT, capture_output=True, text=True, timeout=600,
                                env={**os.environ, 'NANOLANG_GENERIC_TEST_ROOT': str(ROOT),
                                     'NANOLANG_AFFINE_COMPILER_ROOT': str(wrappers),
                                     'NANOLANG_AFFINE_COMPILERS': 'nanoc_stage1,nanoc_stage2'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
