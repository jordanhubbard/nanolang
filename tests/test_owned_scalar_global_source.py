"""I preserve source global effects through owned VM/native production and shadows."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from tests import test_selected_variant_ownership as original
from tests import test_affine_scalar_union_runtime as runtime
from tests import test_nanoisa_shadow_emitter as shadows

ROOT = Path(__file__).resolve().parents[1]
PREFIX = """resource struct Handle { fd: int }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
"""


def cases():
    retained = []
    suite = original.SelectedVariantOwnership()
    suite.program = lambda source, accepted: retained.append(('original_exactly_once', source, accepted))
    suite.test_scrutinee_call_evaluates_once()
    def program(name, declarations, statements, accepted=True, shadow='assert (== (main) 0)'):
        retained.append((name, PREFIX + declarations + '\nfn main() -> int {\n' + statements +
                         '\nassert (== (close_handle Handle { fd: 7 }) 7) return 0 }\nshadow main { ' + shadow + ' }\n', accepted))
    program('ordered_initializers', 'let first: int = 3\nlet second: int = (+ first 4)', 'assert (== second 7)')
    program('mutable_string', 'let mut text: string = "start"\nlet old: string = text',
            'set text "after" assert (== old "start") assert (== text "after")')
    program('scalar_tags', 'let mut flag: bool = false\nlet mut number: float = 1.5',
            'set flag true set number 2.5 assert flag assert (== number 2.5)')
    program('local_shadows_global', 'let counter: int = 9',
            'let mut counter: int = 0 set counter 7 assert (== counter 7)')
    program('parameter_shadows_global', 'let counter: int = 9\nfn read(counter: int) -> int { return counter }\nshadow read { assert (== (read 7) 7) }',
            'assert (== (read 7) 7) assert (== counter 9)')
    program('loop_counter', 'let mut counter: int = 0',
            'set counter 0 while (< counter 3) { set counter (+ counter 1) } assert (== counter 3)')
    program('ordinary_union_with_global', 'union Value { Some { number: int }, None {} }\nlet counter: int = 7',
            'let value: Value = Value.Some { number: counter } match value { Some(payload) => { assert (== payload.number 7) } None(payload) => { assert false } }')
    program('untyped_global', 'let counter = 7', 'assert (== counter 7)', False)
    program('immutable_assignment', 'let counter: int = 7', 'set counter 8', False)
    program('wrong_store_tag', 'let mut counter: int = 7', 'set counter true', False)
    program('owner_global', 'let owner: Handle = Handle { fd: 7 }', 'let local: Handle = owner let value: int = (close_handle local)', False)
    program('aggregate_global', 'let values: array<int> = [7]', 'assert true', False)
    program('duplicate_global', 'let counter: int = 7\nlet counter: int = 8', 'assert true', False)
    program('read_before_initialization', 'let first: int = second\nlet second: int = 7', 'assert true', False)
    program('helper_before_initialization', 'fn number() -> int { return 7 }\nshadow number { assert (== (number) 7) }\nlet counter: int = (number)', 'assert (== counter 7)', False)
    program('failing_effect_shadow', 'let mut counter: int = 0', 'set counter (+ counter 1)', False,
            'set counter 0 assert (== (main) 0) assert (== counter 0)')
    return retained


class OwnedScalarGlobalSource(unittest.TestCase):
    compiler = runtime.AffineScalarUnionRuntime.compiler

    def test_source_effects_and_refusals(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-global-source-') as directory:
            work = Path(directory)
            for producer in ('nano_virt', 'nanoisa_emit'):
                for name, source, accepted in cases():
                    with self.subTest(producer=producer, case=name):
                        program, module, generated, binary = [work / n for n in ('source.nano', 'source.nvm', 'source.c', 'source.native')]
                        program.write_text(source)
                        module.write_bytes(b'prior module')
                        result = subprocess.run([ROOT / 'bin' / producer, program, '--emit-nvm', '-o', module],
                                                cwd=ROOT, capture_output=True, text=True, timeout=120)
                        # My raw self-hosted producer publishes production only; the
                        # separate shadow emitter below checks this failing assertion.
                        production_only = producer == 'nanoisa_emit' and name == 'failing_effect_shadow'
                        if not accepted and not production_only:
                            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertEqual(module.read_bytes(), b'prior module')
                            continue
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        for command in ([ROOT / 'bin/nano_vm', '--verify-only', module],
                                        [ROOT / 'bin/nano_vm', module],
                                        [ROOT / 'bin/nvm2c', module, '-o', generated],
                                        [self.compiler(), '-std=c11', '-Wall', '-Wextra', '-Werror',
                                         '-fsanitize=address,undefined', '-fno-omit-frame-pointer', generated, '-o', binary],
                                        [binary]):
                            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120,
                                                    env={**os.environ, 'ASAN_OPTIONS': asan_options("halt_on_error=1"),
                                                         'UBSAN_OPTIONS': 'halt_on_error=1'})
                            self.assertEqual(result.returncode, 0, str(command) + '\n' + result.stdout + result.stderr)


    def test_native_stages_effects_and_refusals(self):
        for compiler in ('nanoc_stage1', 'nanoc_stage2'):
            for name, source, accepted in cases():
                with self.subTest(compiler=compiler, case=name), tempfile.TemporaryDirectory(prefix='nano-owned-global-native-') as directory:
                    program, output = Path(directory) / 'source.nano', Path(directory) / 'program'
                    program.write_text(source)
                    output.write_bytes(b'prior artifact')
                    result = subprocess.run([ROOT / 'bin' / compiler, program, '-o', output],
                                            cwd=ROOT, capture_output=True, text=True, timeout=120)
                    if not accepted:
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior artifact')
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        result = subprocess.run([output], capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class OwnedScalarGlobalShadows(unittest.TestCase):
    setUpClass = classmethod(shadows.ShadowEmitter.setUpClass.__func__)
    tearDownClass = classmethod(shadows.ShadowEmitter.tearDownClass.__func__)
    command = staticmethod(shadows.ShadowEmitter.command)
    emit = shadows.ShadowEmitter.emit
    execute = shadows.ShadowEmitter.execute

    def test_selfhosted_shadow_effects(self):
        for name, source, accepted in cases():
            if not accepted and name != 'failing_effect_shadow':
                continue
            with self.subTest(case=name):
                assembly = self.emit(source).stdout
                self.execute(assembly, success=accepted)


if __name__ == '__main__':
    unittest.main()
