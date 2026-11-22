"""I retain source-level module introspection in VM and native products."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_INTROSPECTION_COMPILER", ROOT / "bin/nano_virt")).resolve()


CALLABLE_FIXTURE = ROOT / "tests/fixtures/module_introspection_callables.nano.txt"


class NanoisaIntrospection(unittest.TestCase):
    def test_original_root_module_fixture(self):
        with tempfile.TemporaryDirectory(prefix='nano-root-module-original-') as directory:
            self.exercise_product(ROOT / 'tests/test_module_introspection.nano', Path(directory))

    def test_explicit_root_inventory_and_all_operations(self):
        for unsafe, empty in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(unsafe=unsafe, empty=empty), tempfile.TemporaryDirectory(prefix='nano-root-module-facts-') as directory:
                work = Path(directory).resolve()
                source = work / 'different_filename.nano'
                declarations = []
                for name, result in [('is_unsafe', 'bool'), ('has_ffi', 'bool'), ('name', 'string'),
                                     ('path', 'string'), ('function_count', 'int'), ('struct_count', 'int'),
                                     ('function_name', 'string'), ('struct_name', 'string')]:
                    parameter = 'index:int' if name.endswith('_name') else ''
                    declarations.append(f'extern fn ___module_{name}_root_probe({parameter})->{result}\n')
                source.write_text(('unsafe ' if unsafe else '') + 'module root_probe\n' + ''.join(declarations) +
                    ('' if empty else 'pub ') + 'struct Visible { n:int }\nstruct Hidden { n:int }\n' +
                    ('' if empty else 'pub ') + 'fn answer()->int{return 42}\nshadow answer{assert (== (answer) 42)}\n'
                    'fn private_value()->int{return 9}\nshadow private_value{assert (== (private_value) 9)}\n'
                    'let mut evaluations:int = 0\nfn index()->int{set evaluations (+ evaluations 1) return 0}\nshadow index{let saved:int=evaluations set evaluations 0 assert (== (index) 0) assert (== evaluations 1) set evaluations saved}\n'
                    'fn main()->int{unsafe{\n'
                    '(println (___module_is_unsafe_root_probe)) (println (___module_has_ffi_root_probe))\n'
                    '(println (___module_name_root_probe)) (println (___module_path_root_probe))\n'
                    '(println (___module_function_count_root_probe)) (println (___module_struct_count_root_probe))\n'
                    '(println (___module_function_name_root_probe 0)) (println (___module_struct_name_root_probe 0))\n' +
                    'assert (== (___module_is_unsafe_root_probe) ' + ('true' if unsafe else 'false') + ')\n'
                    'assert (___module_has_ffi_root_probe)\nassert (== (___module_name_root_probe) "root_probe")\n' +
                    f'assert (== (___module_path_root_probe) {json.dumps(str(source))})\n' +
                    f'assert (== (___module_function_count_root_probe) {0 if empty else 1}) assert (== (___module_struct_count_root_probe) {0 if empty else 1})\n' +
                    f'assert (== (___module_function_name_root_probe (index)) {json.dumps("" if empty else "answer")})\n' +
                    f'assert (== (___module_struct_name_root_probe 0) {json.dumps("" if empty else "Visible")})\n' +
                    'assert (== (___module_function_name_root_probe -1) "") assert (== (___module_function_name_root_probe 1) "")\n'
                    'assert (== (___module_struct_name_root_probe -1) "") assert (== (___module_struct_name_root_probe 1) "")\n'
                    '} assert (== evaluations 1) return 0}\nshadow main{assert true}\n')
                self.exercise_product(source, work)

    def test_explicit_root_dependency_collision_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-root-module-collision-') as directory:
            work = Path(directory).resolve()
            (work / 'dependency.nano').write_text('module root_probe\npub fn answer()->int{return 1}\nshadow answer{assert (== (answer) 1)}\n')
            source = work / 'main.nano'
            source.write_text('module "dependency.nano" as dependency\nmodule root_probe\nfn main()->int{return 0}\nshadow main{assert true}\n')
            module = work / 'program.nvm'
            module.write_bytes(b'prior-output')
            result = subprocess.run([COMPILER, source, '--emit-nvm', '-o', module], cwd=ROOT, capture_output=True, timeout=120)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(b'I reject ambiguous module introspection identity: root_probe', result.stdout + result.stderr)
            self.assertEqual(module.read_bytes(), b'prior-output')

    def test_declared_root_wins_over_dependency_filename_alias(self):
        with tempfile.TemporaryDirectory(prefix='nano-root-module-alias-') as directory:
            work = Path(directory).resolve()
            dependency = work / 'root_probe.nano'
            dependency.write_text('module dependency_probe\npub fn foreign_answer()->int{return 7}\nshadow foreign_answer{assert (== (foreign_answer) 7)}\n')
            source = work / 'main.nano'
            source.write_text('module root_probe\nmodule "root_probe.nano" as dependency\n'
                'extern fn ___module_path_root_probe()->string\nextern fn ___module_path_dependency_probe()->string\n'
                'extern fn ___module_function_count_root_probe()->int\nextern fn ___module_function_count_dependency_probe()->int\n'
                'pub fn answer()->int{return 42}\nshadow answer{assert (== (answer) 42)}\n'
                'fn main()->int{unsafe{\n' +
                f'assert (== (___module_path_root_probe) {json.dumps(str(source))})\n' +
                f'assert (== (___module_path_dependency_probe) {json.dumps(str(dependency))})\n' +
                'assert (== (___module_function_count_root_probe) 1) assert (== (___module_function_count_dependency_probe) 1)\n'
                '} assert (== (dependency.foreign_answer) 7) return 0}\nshadow main{assert true}\n')
            self.exercise_product(source, work)

    def test_undeclared_root_does_not_add_a_fallback_identity(self):
        with tempfile.TemporaryDirectory(prefix='nano-root-module-undeclared-') as directory:
            work = Path(directory).resolve()
            (work / 'dependency.nano').write_text('module main\npub fn answer()->int{return 7}\nshadow answer{assert (== (answer) 7)}\n')
            source = work / 'main.nano'
            source.write_text('module "dependency.nano" as dependency\n'
                'extern fn ___module_function_count_main()->int\n'
                'fn main()->int{assert (== (___module_function_count_main) 1) assert (== (dependency.answer) 7) return 0}\n'
                'shadow main{assert true}\n')
            self.exercise_product(source, work)

    def checked(self, args):
        result = subprocess.run([str(x) for x in args], cwd=ROOT,
                                capture_output=True, timeout=120,
                                env={**os.environ})
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-6000:])
        return result

    def exercise(self, empty):
        with tempfile.TemporaryDirectory(prefix='nano-module-facts-') as directory:
            work = Path(directory).resolve()
            dependency = work / 'reflection_probe.nano'
            dependency.write_text('module reflection_probe\n' + ('' if empty else
                'pub struct Visible { value: int }\nstruct Hidden { value: int }\n'
                'pub fn answer() -> int { return 42 }\nshadow answer { assert true }\n'
                'fn private_value() -> int { return 9 }\nshadow private_value { assert true }\n'))
            declarations = []
            for name, result in [('is_unsafe', 'bool'), ('has_ffi', 'bool'),
                                 ('name', 'string'), ('path', 'string'),
                                 ('function_count', 'int'), ('struct_count', 'int'),
                                 ('function_name', 'string'), ('struct_name', 'string')]:
                parameter = 'index: int' if name in ('function_name', 'struct_name') else ''
                declarations.append(f'extern fn ___module_{name}_reflection_probe({parameter}) -> {result}\n')
            expected_function, expected_struct = ('', '') if empty else ('answer', 'Visible')
            source = work / 'main.nano'
            source.write_text(f'module {json.dumps(str(dependency))} as probe\n' + ''.join(declarations) +
                'fn count() -> int { return (___module_function_count_reflection_probe) }\n'
                'shadow count { assert true }\n'
                'let mut evaluations: int = 0\n'
                'fn index() -> int { set evaluations (+ evaluations 1) return 0 }\n'
                'shadow index { assert true }\nfn main() -> int { unsafe {\n'
                'assert (not (___module_is_unsafe_reflection_probe))\n'
                'assert (not (___module_has_ffi_reflection_probe))\n'
                'assert (== (___module_name_reflection_probe) "reflection_probe")\n'
                f'assert (== (___module_path_reflection_probe) {json.dumps(str(dependency))})\n'
                f'assert (== (___module_function_count_reflection_probe) {0 if empty else 1})\n'
                f'assert (== (count) {0 if empty else 1})\n'
                f'assert (== (___module_struct_count_reflection_probe) {0 if empty else 1})\n'
                f'assert (== (___module_function_name_reflection_probe (index)) {json.dumps(expected_function)})\n'
                f'assert (== (___module_struct_name_reflection_probe 0) {json.dumps(expected_struct)})\n'
                'assert (== (___module_function_name_reflection_probe -1) "")\n'
                'assert (== (___module_function_name_reflection_probe 1) "")\n'
                'assert (== (___module_struct_name_reflection_probe -1) "")\n'
                'assert (== (___module_struct_name_reflection_probe 99) "")\n'
                '} assert (== evaluations 1) return 0 }\nshadow main { assert true }\n')
            self.exercise_product(source, work)

    def exercise_product(self, source, work):
        module, c_file, binary = (work / name for name in ('program.nvm', 'program.c', 'program'))
        self.checked([COMPILER, source, '--emit-nvm', '-o', module])
        self.checked([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.checked([ROOT / 'bin/nano_vm', module])
        self.checked([ROOT / 'bin/nvm2c', module, '-o', c_file])
        flags = ['-rdynamic', '-ldl'] if sys.platform.startswith('linux') else []
        self.checked([os.environ.get('NANO_NATIVE_TEST_CC', 'cc'), '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                      '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                      c_file, ROOT / 'bin/nano_aot_runtime.o', '-lm', *flags, '-o', binary])
        self.checked([binary])

    def test_owned_operations_and_refusals(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-facts-') as directory:
            work = Path(directory).resolve()
            dependency = work / 'different_filename.nano'
            dependency.write_text('module callable_probe\npub struct Visible { value: int }\n'
                                  'pub fn answer() -> int { return 7 }\n'
                                  'shadow answer { assert (== (answer) 7) }\n')
            fixture = (ROOT / 'tests/fixtures/module_introspection_owned.nano.txt').read_text()
            fixture = fixture.replace('@MODULE_PATH@', json.dumps(str(dependency)))
            source = work / 'main.nano'
            source.write_text(fixture)
            self.exercise_product(source, work)
            controls = {
                'leak': fixture.replace('assert (== (consume h) 7)', ''),
                'moved': fixture.replace('assert (== (consume h) 7)',
                                         'assert (== (consume h) 7) assert (== (consume h) 7)'),
                'signature': fixture + '\nextern fn ___module_name_unused() -> int\n',
            }
            for name, text in controls.items():
                with self.subTest(refusal=name):
                    source.write_text(text)
                    module = work / 'previous.nvm'
                    module.write_bytes(b'previous module')
                    result = subprocess.run([COMPILER, source, '--emit-nvm', '-o', module],
                                            cwd=ROOT, capture_output=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(module.read_bytes(), b'previous module')
                    diagnostic = result.stdout + result.stderr
                    if name == 'signature':
                        self.assertIn(b'module introspection signature', diagnostic)
                    else:
                        self.assertIn(b'resource remains live' if name == 'leak' else b'moved value', diagnostic)

    def test_all_operations_and_single_index_evaluation(self):
        self.exercise(False)

    def test_empty_export_sets(self):
        self.exercise(True)

    def test_function_values_and_declared_module_path(self):
        with tempfile.TemporaryDirectory(prefix='nano-module-callables-') as directory:
            work = Path(directory).resolve()
            dependency = work / 'different_filename.nano'
            dependency.write_text('module callable_probe\n'
                'pub struct Visible { value: int }\n'
                'pub fn answer() -> int { return 7 }\n'
                'shadow answer { assert (== (answer) 7) }\n')
            source = work / 'main.nano'
            source.write_text(CALLABLE_FIXTURE.read_text().replace('@MODULE_PATH@', json.dumps(str(dependency))))
            self.exercise_product(source, work)

    def test_repository_metadata_programs(self):
        with tempfile.TemporaryDirectory(prefix='nano-module-regressions-') as directory:
            module = Path(directory).resolve() / 'program.nvm'
            for name in ('nl_functions_module_metadata', 'nl_functions_module_introspection_flags',
                         'test_module_introspection_exports', 'test_module_introspection_import'):
                with self.subTest(name=name):
                    self.checked([COMPILER, ROOT / 'tests' / (name + '.nano'), '--emit-nvm', '-o', module])
                    self.checked([ROOT / 'bin/nano_vm', '--verify-only', module])
                    self.checked([ROOT / 'bin/nano_vm', module])

    def test_ambiguous_module_identity_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-module-collision-') as directory:
            module = Path(directory).resolve() / 'previous.nvm'
            module.write_bytes(b'previous module')
            result = subprocess.run([str(COMPILER), str(ROOT / 'tests/negative/neg_module_introspection_identity.nano'),
                                     '--emit-nvm', '-o', str(module)], cwd=ROOT, capture_output=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'I reject ambiguous module introspection identity: duplicate_identity',
                          result.stdout + result.stderr)
            self.assertEqual(module.read_bytes(), b'previous module')

    def test_ordinary_function_with_similar_name_keeps_its_body(self):
        with tempfile.TemporaryDirectory(prefix='nano-module-function-') as directory:
            work = Path(directory).resolve()
            source, module = work / 'main.nano', work / 'main.nvm'
            source.write_text('fn ___module_name_local() -> string { return "ordinary" }\n'
                              'shadow ___module_name_local { assert true }\n'
                              'fn main() -> int { assert (== (___module_name_local) "ordinary") return 0 }\n'
                              'shadow main { assert true }\n')
            self.checked([COMPILER, source, '--emit-nvm', '-o', module])
            self.checked([ROOT / 'bin/nano_vm', module])

    def test_intrinsic_signature_is_required_before_publication(self):
        with tempfile.TemporaryDirectory(prefix='nano-module-signature-') as directory:
            work = Path(directory).resolve()
            source, module = work / 'main.nano', work / 'previous.nvm'
            for declaration in (
                'extern fn ___module_function_name_probe(index: string) -> string',
                'extern fn ___module_struct_name_probe() -> string',
                'extern fn ___module_name_probe() -> int',
                'extern fn ___module_path_probe() -> bool',
                'extern fn ___module_is_unsafe_probe() -> int',
                'extern fn ___module_has_ffi_probe(index: int) -> bool',
                'extern fn ___module_function_count_probe() -> string',
                'extern fn ___module_struct_count_probe(index: int) -> int',
                'extern fn ___module_struct_name_probe(index: bool) -> string',
            ):
                with self.subTest(declaration=declaration):
                    source.write_text(declaration + '\nfn main() -> int { return 0 }\nshadow main { assert true }\n')
                    module.write_bytes(b'previous module')
                    result = subprocess.run([COMPILER, source, '--emit-nvm', '-o', module],
                                            cwd=ROOT, capture_output=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'I require the declared module introspection signature', result.stdout + result.stderr)
                    self.assertEqual(module.read_bytes(), b'previous module')


if __name__ == '__main__':
    unittest.main()
