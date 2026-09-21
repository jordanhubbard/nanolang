"""I retain checked byte source and reconstruction behavior across actual producers."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import unittest

from tests import test_cast_u8_backends as backend

ROOT = Path(__file__).resolve().parents[1]


class ComputedU8Source(unittest.TestCase):
    setUp = backend.CastU8Backends.setUp
    command = backend.CastU8Backends.command
    run_actual = backend.CastU8Backends.run_actual
    run_trap = backend.CastU8Backends.run_trap
    native_object = backend.CastU8Backends.native_object
    assert_target = backend.CastU8Backends.assert_target
    compare = backend.CastU8Backends.compare

    def producers(self):
        emitters = shlex.split(os.environ['NANO_U8_EMITTERS'])
        self.assertEqual(len(emitters), 3, 'I require separately built C-seed, Stage1 and Stage2 Nano emitters.')
        paths = [ROOT / 'bin/nano_virt', *map(Path, emitters)]
        self.assertEqual(len(set(map(str, paths))), 4)
        (self.artifacts / 'source-providers.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + '\n')
        return paths

    def paired(self, text, backends=True, require_cast=True):
        original = self.artifacts
        source = original / 'source.nano'; source.write_text(text)
        seed_native = original / 'seed-native'
        self.run_actual([ROOT / 'bin/nanoc_c', source, '-o', seed_native])
        seed_output = self.run_actual([seed_native])
        for index, producer in enumerate(self.producers()):
            self.artifacts = original / ('producer-' + str(index)); self.artifacts.mkdir()
            module = self.artifacts / 'source.nvm'
            self.run_actual([producer, source, '--emit-nvm', '-o', module])
            before = module.read_bytes()
            self.run_actual([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.assertEqual(self.run_actual([ROOT / 'bin/nano_vm', module]), seed_output)
            assembly = self.run_actual([ROOT / 'bin/nanoisa', 'dump', module]).decode()
            if require_cast: self.assertIn('CAST_U8', assembly)
            if backends:
                # I reassemble the actual retained producer output, not a substitute program.
                self.compare(assembly)
            self.assertEqual(module.read_bytes(), before)
        self.artifacts = original

    def test_checked_scalar_destinations_and_once_only_calls(self):
        self.paired('''let mut calls: int = 0
let mut saved: u8 = (+ 255 2)
fn tick() -> int { set calls (+ calls 1) return 257 }
shadow tick { set calls 0 assert (== (tick) 257) assert (== calls 1) }
fn narrow(value: int) -> u8 { return value }
shadow narrow { assert (== (cast_int (narrow 257)) 1) }
fn identity(value: u8) -> u8 { return value }
shadow identity { let value: u8 = 255 assert (== (cast_int (identity value)) 255) }
fn tail(value: int) -> u8 { return (tick) }
shadow tail { set calls 0 assert (== (cast_int (tail 0)) 1) assert (== calls 1) }
fn main() -> int {
 set calls 0
 let mut value: u8 = (tick)
 assert (== calls 1)
 assert (== (cast_int value) 1)
 set value (+ 255 3)
 assert (== (cast_int value) 2)
 set saved (- 0 1)
 assert (== (cast_int saved) 255)
 let direct: u8 = (identity (tick))
 assert (== calls 2)
 assert (== (cast_int direct) 1)
 let returned: u8 = (tail 0)
 assert (== calls 3)
 assert (== (cast_int returned) 1)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_checked_enum_destinations_and_once_only_calls(self):
        self.paired('''enum Edge { Below = -1, Above = 256, High = 511 }
let mut calls: int = 0
let mut saved: u8 = Edge.High
fn tick() -> Edge { set calls (+ calls 1) return Edge.Above }
shadow tick { set calls 0 assert (== (tick) Edge.Above) assert (== calls 1) }
fn narrow(value: Edge) -> u8 { return value }
shadow narrow { assert (== (cast_int (narrow Edge.Below)) 255) }
fn identity(value: u8) -> u8 { return value }
shadow identity { assert (== (cast_int (identity Edge.High)) 255) }
fn tail() -> u8 { return (tick) }
shadow tail { set calls 0 assert (== (cast_int (tail)) 0) assert (== calls 1) }
fn main() -> int {
 assert (== (cast_int saved) 255)
 set calls 0
 let mut value: u8 = (tick)
 assert (== calls 1)
 assert (== (cast_int value) 0)
 set value Edge.Below
 assert (== (cast_int value) 255)
 set saved Edge.Above
 assert (== (cast_int saved) 0)
 let direct: u8 = (identity (tick))
 assert (== calls 2)
 assert (== (cast_int direct) 0)
 let returned: u8 = (tail)
 assert (== calls 3)
 assert (== (cast_int returned) 0)
 assert (== (cast_int (narrow Edge.Below)) 255)
 assert (== (cast_int (narrow Edge.Above)) 0)
 assert (== (cast_int (narrow Edge.High)) 255)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_qualified_byte_argument_and_return(self):
        dependency = self.artifacts / 'bytes.nano'
        dependency.write_text('module Bytes\npub fn identity(value: u8) -> u8 { return value }\n'
            'shadow identity { let value: u8 = 255 assert (== (cast_int (identity value)) 255) }\n')
        self.paired(f'import "{dependency}" as bytes\n'
            'fn narrowed(value: int) -> u8 { return (bytes.identity (+ value 1)) }\n'
            'shadow narrowed { assert (== (cast_int (narrowed 256)) 1) }\n'
            'fn main() -> int { assert (== (cast_int (narrowed 256)) 1) return 0 }\n'
            'shadow main { assert (== (main) 0) }\n')

    def test_checked_c_indirect_arguments_and_captured_precedence(self):
        # The independent Nano emitter still refuses indirect calls. I qualify
        # the existing C producer route explicitly without substituting it.
        source = self.artifacts / 'captured.nano'
        source.write_text('''let mut shared: int = 900
fn global_value() -> int { return shared }
shadow global_value { assert (== (global_value) 900) }
fn identity(value: u8) -> u8 { return value }
shadow identity { let value: u8 = 7 assert (== (cast_int (identity value)) 7) }
fn main() -> int {
 let mut shared: u8 = 1
 let update: fn() -> u8 = fn() -> u8 { set shared (+ shared 257) return shared }
 let invoke: fn(u8) -> u8 = identity
 let direct: u8 = (invoke (+ 255 2))
 assert (== (cast_int direct) 1)
 assert (== (cast_int (update)) 2)
 assert (== (cast_int shared) 2)
 assert (== (global_value) 900)
 return 0
}
shadow main { assert (== (main) 0) }
''')
        module = self.artifacts / 'captured.nvm'
        self.run_actual([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', module])
        self.run_actual([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.run_actual([ROOT / 'bin/nano_vm', module])
        text = self.run_actual([ROOT / 'bin/nanoisa', 'dump', module]).decode()
        for opcode in ('CAST_U8', 'CALL_INDIRECT', 'LOAD_UPVALUE', 'STORE_UPVALUE'):
            self.assertIn(opcode, text)

    def test_all_byte_values_and_integer_extrema(self):
        values = list(range(256)) + [256, 257, -1, -256, -9223372036854775808, 9223372036854775807]
        checks = ''.join(f' assert (== (cast_int (narrow {value})) {value % 256})\n' for value in values)
        self.paired('fn narrow(value: int) -> u8 { return value }\n'
            'shadow narrow { assert (== (cast_int (narrow 257)) 1) }\n'
            'fn main() -> int {\n' + checks + ' return 0\n}\nshadow main { assert (== (main) 0) }\n')

    def test_original_byte_program_unchanged(self):
        source = ROOT / 'tests/test_u8_basic.nano'
        before = source.read_bytes()
        self.paired(before.decode(), backends=False, require_cast=False)
        self.assertEqual(source.read_bytes(), before)

    def test_literal_and_wrong_source_refusals_preserve_output(self):
        original = self.artifacts
        for case, expression in enumerate(('256', '-1', 'true', '1.5', '"byte"')):
            self.artifacts = original / ('refusal-' + str(case)); self.artifacts.mkdir()
            path = self.artifacts / 'refused.nano'
            path.write_text('fn narrow() -> u8 { return ' + expression + ' }\n'
                'shadow narrow { assert true }\nfn main() -> int { return 0 }\nshadow main { assert true }\n')
            for producer in self.producers():
                output = self.artifacts / 'previous.nvm'; output.write_bytes(b'previous module\n')
                result = self.run_trap([producer, path, '--emit-nvm', '-o', output])
                self.assertGreater(result['returncode'], 0)
                self.assertRegex(result['stdout'] + result['stderr'], '(?i)(byte|u8|type)')
                self.assertEqual(output.read_bytes(), b'previous module\n')
        self.artifacts = original

    def test_reconstruction_keeps_exact_narrowing(self):
        checks = ''
        for value in (0, 1, 127, 128, 200, 201, 255, 256, 257, -1, -256, -9223372036854775808, 9223372036854775807):
            checks += f'PUSH_I64 {value}\nCAST_U8\nCAST_INT\nPUSH_I64 {value % 256}\nI64_EQ\nJMP_FALSE bad\n'
        text = '.entry main\n.function main 0 0 0 int 1\n' + checks + 'PUSH_I64 0\nRET\nbad:\nPUSH_I64 1\nRET\n.end\n'
        asm = self.artifacts / 'original.nasm'; asm.write_text(text)
        module = self.artifacts / 'original.nvm'
        self.run_actual([ROOT / 'bin/nanoisa', 'asm', asm, '-o', module])
        before = module.read_bytes()
        self.run_actual([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.run_actual([ROOT / 'bin/nano_vm', module])
        c = self.artifacts / 'reconstructed.c'
        self.run_actual([ROOT / 'bin/nvm2hl', '--language', 'c', module, '-o', c])
        self.assertIn('uint8_t', c.read_text())
        for optimization in ('O0', 'O2'):
            executable = self.artifacts / ('reconstructed-' + optimization)
            self.run_actual([*self.cc, *self.flags, '-std=c11', '-' + optimization, '-Wall', '-Wextra', '-Werror', c, '-o', executable])
            self.run_actual([executable])
        nano = self.artifacts / 'reconstructed.nano'
        self.run_actual([ROOT / 'bin/nvm2hl', '--language', 'nano', module, '-o', nano])
        self.assertEqual(nano.read_text().count('fn nlr_int_u8('), 1)
        self.assertIn('shadow nlr_int_u8', nano.read_text())
        source = nano.read_text() + '\nshadow nlr_f0_main { assert (== (nlr_f0_main) 0) }\nshadow main { assert (== (main) 0) }\n'
        self.paired(source)
        self.assertEqual(module.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
