"""I retain distinct concrete scalar-union instances in my affine source path."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/nanoisa/fixtures/affine_scalar_union_instances.nano'

IMPORTED_PROVIDER = '''module ImportedChoice

union Choice<T,U> {
    Left { value: T },
    Right { value: U },
    Empty {}
}

pub fn make_int(right: bool) -> Choice<int,string> {
    if right { return Choice.Right { value: "imported-right" } }
    return Choice.Left { value: 17 }
}
shadow make_int { let value: Choice<int,string> = (make_int false) assert (== (inspect_int value) 27) }

pub fn inspect_int(value: Choice<int,string>) -> int {
    match value {
        Left(payload) => { return (+ payload.value 10) }
        Right(payload) => { assert (== payload.value "imported-right") return 55 }
        Empty(payload) => { return (- 0 1) }
    }
}
shadow inspect_int { assert (== (inspect_int (make_int false)) 27) assert (== (inspect_int (make_int true)) 55) }

pub fn make_float(right: bool) -> Choice<float,bool> {
    if right { return Choice.Right { value: true } }
    return Choice.Left { value: 2.5 }
}
shadow make_float { assert (inspect_float (make_float false)) }

pub fn inspect_float(value: Choice<float,bool>) -> bool {
    return match value {
        Left(payload) => (== payload.value 2.5)
        Right(payload) => payload.value
        Empty(payload) => false
    }
}
shadow inspect_float { assert (inspect_float (make_float false)) assert (inspect_float (make_float true)) }
'''

IMPORTED_ROOT = '''module "provider.nano" as provider
module "provider.nano" as same

resource struct Owner { value: int }

fn alias_route(value: provider.Choice<int,string>) -> same.Choice<int,string> { return value }
shadow alias_route { let value: provider.Choice<int,string> = (provider.make_int false) assert (== (provider.inspect_int (alias_route value)) 27) }

fn main() -> int {
    let owner: Owner = Owner { value: 9 }
    let Owner { value } = owner
    let first: provider.Choice<int,string> = (alias_route (provider.make_int false))
    let second: provider.Choice<float,bool> = (provider.make_float true)
    assert (== value 9)
    assert (== (provider.inspect_int first) 27)
    assert (provider.inspect_float second)
    (println "imported-generic-parity-pass")
    return 0
}
shadow main { assert (== (main) 0) }
'''

OTHER_PROVIDER = IMPORTED_PROVIDER.replace('ImportedChoice', 'OtherChoice').replace('value: 17', 'value: 91').replace('value) 27', 'value) 101').replace('(make_int false)) 27', '(make_int false)) 101')

REFUSAL_PREFIX = '''resource struct Owner { value: int }
union Choice<T,U> { Left { value: T }, Right { value: U }, Empty {} }
fn main() -> int {
    let owner: Owner = Owner { value: 9 }
    let Owner { value } = owner
    assert (== value 9)
'''

REFUSALS = {
    'wrong_payload': (REFUSAL_PREFIX + '''
    let wrong: Choice<int,string> = Choice.Left { value: "wrong" }
    return 0
}
shadow main { assert (== (main) 0) }
''', ('TYPE MISMATCH', 'concrete union field type', 'payload')),
    'cross_instance': (REFUSAL_PREFIX + '''
    let first: Choice<int,string> = Choice.Left { value: 17 }
    let wrong: Choice<float,bool> = first
    return 0
}
shadow main { assert (== (main) 0) }
''', ('TYPE MISMATCH', 'concrete union', 'identity', 'expected a value of type')),
    'result_mismatch': (REFUSAL_PREFIX + '''
    let choice: Choice<int,string> = Choice.Left { value: 17 }
    let wrong: int = match choice {
        Left(payload) => payload.value
        Right(payload) => (== payload.value "right")
        Empty(payload) => 0
    }
    return wrong
}
shadow main { assert (== (main) 17) }
''', ('TYPE MISMATCH', 'matching scalar value-match result type', 'match arm')),
    'incomplete_match': (REFUSAL_PREFIX + '''
    let choice: Choice<int,string> = Choice.Left { value: 17 }
    return match choice {
        Left(payload) => payload.value
        Right(payload) => 0
    }
}
shadow main { assert (== (main) 17) }
''', ('NON-EXHAUSTIVE MATCH', 'complete named scalar union match coverage', 'not total',
      'unconditional coverage of every match value')),
    'escaped_payload': (REFUSAL_PREFIX + '''
    let choice: Choice<int,string> = Choice.Left { value: 17 }
    match choice {
        Left(payload) => { assert (== payload.value 17) }
        Right(payload) => { assert (== payload.value "right") }
        Empty(payload) => { assert true }
    }
    return payload.value
}
shadow main { assert (== (main) 17) }
''', ('UNDEFINED', 'payload', 'scope')),
}


class AffineScalarUnionSource(unittest.TestCase):
    def command(self, *args, env=None, timeout=180):
        result = subprocess.run([str(arg) for arg in args], cwd=ROOT,
                                capture_output=True, text=True, env=env, timeout=timeout)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def rejected(self, args, output, expected):
        prior = b'prior-affine-union-output\n'
        output.write_bytes(prior)
        result = subprocess.run([str(arg) for arg in args], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(output.read_bytes(), prior)
        diagnostic = result.stdout + result.stderr
        self.assertTrue(any(fragment.lower() in diagnostic.lower() for fragment in expected),
                        diagnostic)

    def assert_module(self, work, module, stem, expected):
        self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
        vm = self.command(ROOT / 'bin/nano_vm', module)
        self.assertEqual(vm.stdout, expected)
        dumped = self.command(ROOT / 'bin/nanoisa', 'dump', module).stdout
        self.assertIn('Choice<int,string>', dumped)
        self.assertIn('Choice<float,bool>', dumped)
        self.assertIn('MATCH_TAG', dumped)
        self.assertIn('AGG_GET 0', dumped)
        native_source = work / f'{stem}.c'
        native = work / f'{stem}.native'
        self.command(ROOT / 'bin/nvm2c', module, '-o', native_source)
        compiler = os.environ.get('NANO_NATIVE_TEST_CC', '/opt/homebrew/opt/llvm/bin/clang')
        self.command(compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                     '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                     native_source, '-o', native)
        runtime = {**os.environ, 'ASAN_OPTIONS': asan_options("halt_on_error=1")}
        executed = self.command(native, env=runtime)
        self.assertEqual(executed.stdout, expected)

    def assert_shared_ownership_envelope(self, assembly):
        line = next(row for row in assembly.splitlines()
                    if row.startswith('.ownership "'))
        data = bytes.fromhex(line.split('"')[1])
        self.assertEqual(int.from_bytes(data[0:4], 'little'), 3)
        layouts = int.from_bytes(data[4:8], 'little')
        offset = (8 + layouts + 3) & ~3
        functions = int.from_bytes(data[offset:offset + 4], 'little')
        offset += 4
        for _ in range(functions):
            locals_count = int.from_bytes(data[offset:offset + 2], 'little')
            offset += 4 + (locals_count + 1) * 8
        path_bytes = int.from_bytes(data[offset:offset + 4], 'little')
        self.assertGreaterEqual(path_bytes, 4)
        self.assertEqual(path_bytes % 4, 0)
        offset += 4
        paths = data[offset:offset + path_bytes]
        self.assertEqual(len(paths), path_bytes)
        offset += path_bytes
        self.assertEqual(int.from_bytes(data[offset:offset + 4], 'little'), 1)
        offset += 4
        self.assertEqual(int.from_bytes(data[offset:offset + 2], 'little'), 1)
        self.assertEqual(int.from_bytes(data[offset + 2:offset + 4], 'little'), 1)
        payload_bytes = int.from_bytes(data[offset + 4:offset + 8], 'little')
        offset += 8
        payload = data[offset:offset + payload_bytes]
        self.assertEqual(len(payload), payload_bytes)
        self.assertEqual(int.from_bytes(payload[0:4], 'little'), 2)
        offset += payload_bytes
        offset = (offset + 3) & ~3
        self.assertEqual(offset, len(data))

    def write_imported(self, work, root=IMPORTED_ROOT, provider=IMPORTED_PROVIDER,
                       other=None):
        (work / 'provider.nano').write_text(provider)
        (work / 'main.nano').write_text(root)
        if other is not None:
            (work / 'other.nano').write_text(other)
        return work / 'main.nano'

    def test_distinct_instances_verify_and_execute_in_vm_and_native(self):
        with tempfile.TemporaryDirectory(prefix='nano-affine-union-source-') as raw:
            work = Path(raw)
            expected = 'right\nsemantic-source-parity-pass\n'
            selfhost = work / 'stage2-nanoisa-emit'
            self.command(ROOT / 'bin/nanoc_stage2', ROOT / 'src_nano/nanoisa_emit.nano',
                         '-o', selfhost, timeout=900)

            emitters = (ROOT / 'bin/nanoisa_emit', selfhost)
            for index, emitter in enumerate(emitters):
                with self.subTest(route=f'raw-{index}'):
                    nasm = work / f'raw-{index}.nasm'
                    if index == 0:
                        assembly = self.command(emitter, FIXTURE).stdout
                        nasm.write_text(assembly)
                    else:
                        self.command(emitter, FIXTURE, '-o', nasm)
                        assembly = nasm.read_text()
                    self.assert_shared_ownership_envelope(assembly)
                    self.assertIn('.types 1 0 2', assembly)
                    self.assertIn('AGG_PACK 1 0 0 1', assembly)
                    self.assertIn('AGG_PACK 1 1 1 1', assembly)
                    self.assertIn('MATCH_TAG 0', assembly)
                    self.assertIn('AGG_GET 0', assembly)
                    self.assertIn('.parameters 2 union', assembly)
                    self.assertIn('.parameters 3 union', assembly)
                    module = work / f'raw-{index}.nvm'
                    self.command(ROOT / 'bin/nanoisa', 'asm', nasm, '-o', module)
                    self.assert_module(work, module, f'raw-{index}', expected)

            for compiler_name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(route=compiler_name):
                    module = work / f'{compiler_name}.nvm'
                    result = self.command(ROOT / 'bin' / compiler_name, FIXTURE,
                                          '--emit-nvm', '-o', module)
                    self.assertNotIn('E001 TYPE MISMATCH', result.stderr)
                    self.assert_module(work, module, compiler_name, expected)

    def test_imported_aliases_and_concrete_instances_execute_in_vm_and_native(self):
        with tempfile.TemporaryDirectory(prefix='nano-affine-imported-union-') as raw:
            work = Path(raw)
            source = self.write_imported(work)
            expected = 'imported-generic-parity-pass\n'
            for compiler_name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(route=compiler_name):
                    module = work / f'{compiler_name}.nvm'
                    result = self.command(ROOT / 'bin' / compiler_name, source,
                                          '--emit-nvm', '-o', module)
                    self.assertNotIn('TYPE MISMATCH', result.stdout + result.stderr)
                    self.assert_module(work, module, compiler_name, expected)

    def test_imported_identity_and_shadow_refusals_preserve_prior_output(self):
        cross_instance = IMPORTED_ROOT.replace(
            'assert (provider.inspect_float second)',
            'assert (provider.inspect_float first)')
        wrong_payload = IMPORTED_ROOT.replace(
            'let first: provider.Choice<int,string> = (alias_route (provider.make_int false))',
            'let first: provider.Choice<int,string> = Choice.Left { value: true }')
        cross_declaration = IMPORTED_ROOT.replace(
            'module "provider.nano" as same',
            'module "other.nano" as other').replace(
            'same.Choice<int,string>', 'provider.Choice<int,string>').replace(
            '(alias_route (provider.make_int false))', '(other.make_int false)')
        bad_shadow = IMPORTED_PROVIDER.replace(
            'shadow make_int { let value: Choice<int,string> = (make_int false) assert (== (inspect_int value) 27) }',
            'shadow make_int { assert false }')
        cases = {
            'cross_instance': (cross_instance, IMPORTED_PROVIDER, None,
                               ('concrete union value', 'declared qualified argument type',
                                'expected a value of type')),
            'wrong_payload': (wrong_payload, IMPORTED_PROVIDER, None,
                              ('concrete union field type', 'TYPE MISMATCH')),
            'cross_declaration': (cross_declaration, IMPORTED_PROVIDER,
                                  OTHER_PROVIDER,
                                  ('already defined', 'declared qualified argument type',
                                   'declaration', 'expected a value of type')),
            'failed_dependency_shadow': (IMPORTED_ROOT, bad_shadow, None,
                                         ('failed shadow', 'Assertion failed',
                                          'failed shadows')),
        }
        with tempfile.TemporaryDirectory(prefix='nano-affine-imported-refusal-') as raw:
            work = Path(raw)
            for case, (root, provider, other, expected) in cases.items():
                source = self.write_imported(work, root, provider, other)
                for compiler_name in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                    with self.subTest(case=case, route=compiler_name):
                        output = work / f'{case}-{compiler_name}.nvm'
                        self.rejected([ROOT / 'bin' / compiler_name, source,
                                       '--emit-nvm', '-o', output], output, expected)

    def test_refusals_preserve_prior_output_on_every_frontend(self):
        with tempfile.TemporaryDirectory(prefix='nano-affine-union-refusal-') as raw:
            work = Path(raw)
            selfhost = work / 'stage2-nanoisa-emit'
            self.command(ROOT / 'bin/nanoc_stage2', ROOT / 'src_nano/nanoisa_emit.nano',
                         '-o', selfhost, timeout=900)
            routes = (
                ('raw-cseed', ROOT / 'bin/nanoisa_emit'),
                ('raw-selfhost', selfhost),
                ('nano-virt', ROOT / 'bin/nano_virt'),
                ('stage1', ROOT / 'bin/nanoc_stage1'),
                ('stage2', ROOT / 'bin/nanoc_stage2'),
            )
            for case, (source, expected) in REFUSALS.items():
                source_path = work / f'{case}.nano'
                source_path.write_text(source)
                for route, frontend in routes:
                    with self.subTest(case=case, route=route):
                        output = work / f'{case}-{route}.out'
                        args = [frontend, source_path, '-o', output]
                        if route in ('nano-virt', 'stage1', 'stage2'):
                            args.insert(2, '--emit-nvm')
                        self.rejected(args, output, expected)


if __name__ == '__main__':
    unittest.main()
