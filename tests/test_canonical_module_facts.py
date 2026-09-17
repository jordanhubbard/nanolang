"""I retain source-level module introspection in VM and native products."""
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


DRIVER = Path(os.environ.get("NANOC", str(ROOT / "bin/nanoc_stage1")))


class CanonicalModuleFacts(unittest.TestCase):
    def checked(self, args):
        result = subprocess.run([str(x) for x in args], cwd=ROOT,
                                capture_output=True, timeout=120,
                                env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1'})
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-6000:])
        return result

    def exercise(self, empty):
        with tempfile.TemporaryDirectory(prefix='nano-module-facts-') as directory:
            work = Path(directory)
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
                'let mut evaluations: int = 0\n'
                'fn index() -> int { set evaluations (+ evaluations 1) return 0 }\n'
                'shadow index { assert true }\nfn main() -> int { unsafe {\n'
                'assert (not (___module_is_unsafe_reflection_probe))\n'
                'assert (not (___module_has_ffi_reflection_probe))\n'
                'assert (== (___module_name_reflection_probe) "reflection_probe")\n'
                f'assert (== (___module_path_reflection_probe) {json.dumps(str(dependency))})\n'
                f'assert (== (___module_function_count_reflection_probe) {0 if empty else 1})\n'
                f'assert (== (___module_struct_count_reflection_probe) {0 if empty else 1})\n'
                f'assert (== (___module_function_name_reflection_probe (index)) {json.dumps(expected_function)})\n'
                f'assert (== (___module_struct_name_reflection_probe 0) {json.dumps(expected_struct)})\n'
                'assert (== (___module_function_name_reflection_probe -1) "")\n'
                'assert (== (___module_function_name_reflection_probe 1) "")\n'
                'assert (== (___module_struct_name_reflection_probe -1) "")\n'
                'assert (== (___module_struct_name_reflection_probe 99) "")\n'
                '} assert (== evaluations 1) return 0 }\nshadow main { assert true }\n')
            module, c_file, binary = (work / name for name in ('program.nvm', 'program.c', 'program'))
            self.checked([DRIVER, source, '--emit-nvm', '-o', module])
            self.checked([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.checked([ROOT / 'bin/nano_vm', module])
            self.checked([ROOT / 'bin/nvm2c', module, '-o', c_file])
            flags = ['-rdynamic', '-ldl'] if sys.platform.startswith('linux') else []
            self.checked(['cc', '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                          c_file, ROOT / 'bin/nano_aot_runtime.o', '-lm', *flags, '-o', binary])
            self.checked([binary])

    def test_all_operations_and_single_index_evaluation(self):
        self.exercise(False)

    def test_empty_export_sets(self):
        self.exercise(True)

    def test_ordinary_function_with_similar_name_keeps_its_body(self):
        with tempfile.TemporaryDirectory(prefix='nano-module-function-') as directory:
            work = Path(directory)
            source, module = work / 'main.nano', work / 'main.nvm'
            source.write_text('fn ___module_name_local() -> string { return "ordinary" }\n'
                              'shadow ___module_name_local { assert true }\n'
                              'fn main() -> int { assert (== (___module_name_local) "ordinary") return 0 }\n'
                              'shadow main { assert true }\n')
            self.checked([DRIVER, source, '--emit-nvm', '-o', module])
            self.checked([ROOT / 'bin/nano_vm', module])

    def test_intrinsic_signature_is_required_before_publication(self):
        with tempfile.TemporaryDirectory(prefix='nano-module-signature-') as directory:
            work = Path(directory)
            source, module = work / 'main.nano', work / 'previous.nvm'
            dependency = work / 'probe.nano'
            dependency.write_text('module probe\n')
            for declaration in (
                'extern fn ___module_function_name_probe(index: string) -> string',
                'extern fn ___module_struct_name_probe() -> string',
                'extern fn ___module_name_probe() -> int',
            ):
                with self.subTest(declaration=declaration):
                    source.write_text(f'module {json.dumps(str(dependency))} as probe\n' + declaration + '\nfn main() -> int { return 0 }\nshadow main { assert true }\n')
                    module.write_bytes(b'previous module')
                    result = subprocess.run([DRIVER, source, '--emit-nvm', '-o', module],
                                            cwd=ROOT, capture_output=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'introspection', result.stdout + result.stderr)
                    self.assertEqual(module.read_bytes(), b'previous module')

    def test_public_pure_modifier_exports(self):
        with tempfile.TemporaryDirectory(prefix='nano-pure-facts-') as directory:
            work = Path(directory)
            dependency = work / 'pure_exports.nano'
            dependency.write_text(
                'module pure_exports\n'
                'pub pure fn visible() -> int { return 7 }\n'
                'shadow visible { assert (== (visible) 7) }\n'
                'pure fn hidden() -> int { return 3 }\n'
                'shadow hidden { assert (== (hidden) 3) }\n'
                'pub fn ordinary() -> int { return 9 }\n'
                'shadow ordinary { assert (== (ordinary) 9) }\n')
            source = work / 'main.nano'
            source.write_text(f'module {json.dumps(str(dependency))} as dependency\n'
                'extern fn ___module_function_count_pure_exports() -> int\n'
                'extern fn ___module_function_name_pure_exports(index: int) -> string\n'
                'fn main() -> int { unsafe {\n'
                'assert (== (___module_function_count_pure_exports) 2)\n'
                'assert (== (___module_function_name_pure_exports 0) "visible")\n'
                'assert (== (___module_function_name_pure_exports 1) "ordinary")\n'
                'assert (== (___module_function_name_pure_exports 2) "")\n'
                '} assert (== (dependency.visible) 7) return 0 }\n'
                'shadow main { assert true }\n')
            module, binary = work / 'program.nvm', work / 'program'
            self.checked([DRIVER, source, '--emit-nvm', '-o', module])
            self.checked([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.checked([ROOT / 'bin/nano_vm', module])
            self.checked([DRIVER, source, '-o', binary])
            self.checked([binary])

    def test_existing_flags_and_export_shadows(self):
        with tempfile.TemporaryDirectory(prefix='nano-facts-existing-') as directory:
            work = Path(directory)
            for name in ('nl_functions_module_introspection_flags.nano',
                         'test_module_introspection_exports.nano'):
                with self.subTest(source=name):
                    source = ROOT / 'tests' / name
                    module = work / 'program.nvm'
                    self.checked([DRIVER, source, '--emit-nvm', '-o', module])
                    self.checked([ROOT / 'bin/nano_vm', module])
                    binary = work / 'program'
                    self.checked([DRIVER, source, '-o', binary])
                    self.checked([binary])
                    c_file = work / 'program.c'
                    self.checked([DRIVER, source, '--target', 'c', '-o', c_file])
                    self.assertNotIn('___module_', c_file.read_text())

    def test_canonical_source_closure_excludes_legacy_emission(self):
        seen = set()
        pending = [ROOT / 'src_nano/nanoc_v06.nano']
        while pending:
            source = pending.pop().resolve()
            if source in seen:
                continue
            seen.add(source)
            self.assertNotIn(source.name, ('transpiler.nano', 'module_introspection.nano'))
            for name in re.findall(r'^\s*(?:unsafe\s+)?(?:import|from|module)\s+"([^"]+)"',
                                   source.read_text(), re.MULTILINE):
                if not name.endswith('.nano'):
                    name += '.nano'
                candidates = (source.parent / name, ROOT / name, ROOT / 'modules' / name)
                resolved = next((path for path in candidates if path.is_file()), None)
                self.assertIsNotNone(resolved, (source, name))
                pending.append(resolved)

    def test_duplicate_module_identity(self):
        self.checked(['bash', ROOT / 'tests/test_module_introspection_identity.sh', DRIVER])


if __name__ == '__main__':
    unittest.main()
