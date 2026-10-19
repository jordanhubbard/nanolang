"""I reconstruct actual local resource producers without losing ownership."""
import json
import os
from pathlib import Path
import re
import shlex
import struct
import sys
import tempfile
import unittest
import zlib

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''resource struct Left { value: int, yes: bool }
resource struct Right { value: int, yes: bool }
fn main() -> int {
    let mut counter: int = 3
    let first: Left = Left { value: counter, yes: true }
    set counter (+ counter 17)
    let moved: Left = first
    let Left { value, yes } = moved
    let left_value: int = value
    let second: Right = Right { yes: false, value: counter }
    set counter (+ counter 31)
    let Right { value, yes } = second
    return (+ (+ left_value value) counter)
}
shadow main { assert true }
'''


class OwnedReconstruction(unittest.TestCase):
    def command(self, args, expected=0, build=False):
        result = run_captured(args, cwd=ROOT, timeout=180, build=build, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout

    def source_module(self, work, compiler=None):
        source, module = work / 'source.nano', work / 'source.nvm'
        source.write_text(SOURCE)
        self.command([compiler or ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module])
        return module

    def facts(self, module):
        return json.loads(self.command([ROOT/'bin/nanoisa_hl_facts', module]))

    def native(self, source, binary):
        cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or 'cc')
        for optimization in ('-O0', '-O2'):
            self.command([*cc, '-std=c11', optimization, '-Wall', '-Wextra', '-Werror',
                          '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source,
                          ROOT/'bin/nano_aot_runtime.o', '-lm', *native_link_flags(), '-o', binary], build=True)
            self.command([binary], expected=74)

    def test_actual_producers_preserve_nominal_moves_fields_and_snapshots(self):
        compilers = [ROOT/'bin/nano_virt']
        compilers += [Path(x) for x in shlex.split(os.environ.get('NANOLANG_OWNED_RECONSTRUCTION_COMPILERS', ''))]
        for compiler in compilers:
            with self.subTest(compiler=str(compiler)), tempfile.TemporaryDirectory(prefix='nano-owned-reconstruct-') as tmp:
                work = Path(tmp)
                module = self.source_module(work, compiler)
                facts = self.facts(module)
                self.assertEqual([x['identity'] for x in facts['layouts']], [0, 1])
                self.assertEqual([x['fields'] for x in facts['layouts']], [[1, 4], [1, 4]])
                self.assertTrue(all(x['resource'] for x in facts['layouts']))
                self.command([ROOT/'bin/nano_vm', module], expected=74)
                for language in ('c', 'nano'):
                    output = work / ('reconstructed.' + language)
                    self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output])
                    text = output.read_text()
                    self.assertIn('NlrOwned0', text)
                    self.assertIn('NlrOwned1', text)
                    self.assertNotRegex(text, r'\bgoto\b|nano_vm|nvm_blob|switch\s*\(')
                    if language == 'c':
                        self.native(output, work/'reconstructed-c')
                    else:
                        rebuilt = work/'rebuilt.nvm'
                        self.command([ROOT/'bin/nano_virt', output, '--emit-nvm', '-o', rebuilt])
                        self.command([ROOT/'bin/nano_vm', rebuilt], expected=74)
                        dump = self.command([ROOT/'bin/nanoisa', 'dump', rebuilt])
                        self.assertIn('.ownership', dump)
                        self.assertIn('OWN_PACK 0', dump)
                        self.assertIn('OWN_PACK 1', dump)
                        self.assertIn('OWN_UNPACK_LOCAL', dump)
                        ownership = bytes.fromhex(''.join(re.findall(r'\.ownership "([0-9a-f]+)"', dump)))
                        self.assertEqual(struct.unpack_from('<II', ownership), (1, 2))
                        self.assertEqual(ownership[8:10], bytes([3, 3]))
                        layouts = bytes.fromhex(''.join(re.findall(r'\.layouts "([0-9a-f]+)"', dump)))
                        self.assertEqual(struct.unpack_from('<I', layouts)[0], 2)
                        at = 4
                        for identity in (0, 1):
                            kind, pad, count, name = struct.unpack_from('<BBHI', layouts, at)
                            self.assertEqual((kind, pad, count), (0, 0, 2))
                            at += 8
                            self.assertEqual([layouts[at + 12*j] for j in range(count)], [1, 4])
                            self.assertEqual([struct.unpack_from('<I', layouts, at+12*j+4)[0]
                                              for j in range(count)], [0xffffffff, 0xffffffff])
                            at += 12*count
                        self.assertEqual(at, len(layouts))
                        native = work/'rebuilt.c'
                        self.command([ROOT/'bin/nvm2c', rebuilt, '-o', native])
                        self.native(native, work/'rebuilt-native')

    def test_initializer_named_entry_refuses_atomically(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-initializer-') as tmp:
            work = Path(tmp)
            module = self.source_module(work)
            assembly = self.command([ROOT/'bin/nanoisa', 'dump', module])
            # I first transport an unused initializer spelling in a valid module.
            name_index = len(re.findall(r'^\.string ', assembly, re.MULTILINE))
            assembly += '\n.string "__init__"\n'
            source, initializer = work/'initializer.nasm', work/'initializer.nvm'
            source.write_text(assembly)
            self.command([ROOT/'bin/nanoisa', 'asm', source, '-o', initializer])
            self.command([ROOT/'bin/nano_vm', initializer], expected=74)
            data = bytearray(initializer.read_bytes())
            header, count = struct.unpack_from('<II', data, 20)
            for i in range(count):
                kind, _, offset, _ = struct.unpack_from('<IIQQ', data, header+24*i)
                if kind == 5:
                    struct.pack_into('<I', data, offset+4, name_index)
                    break
            else:
                self.fail('I require the real function table.')
            struct.pack_into('<I', data, 36, zlib.crc32(data[header:]))
            initializer.write_bytes(data)
            for language in ('c', 'nano'):
                output = work/('previous.'+language)
                output.write_text('previous checked output')
                result = run_captured([ROOT/'bin/nvm2hl', '--language', language,
                                      initializer, '-o', output], cwd=ROOT, timeout=30, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_text(), 'previous checked output')
                self.assertIn('I did not publish', result.stderr)
                self.assertNotIn('checksum', result.stderr.lower())

    def test_verified_scalar_branch_preserves_live_owner(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-branch-') as tmp:
            work = Path(tmp)
            text = SOURCE.replace('set counter (+ counter 17)',
                                  'if (> counter 0) { set counter (+ counter 17) }')
            source, module = work/'branch.nano', work/'branch.nvm'
            source.write_text(text)
            self.command([ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module])
            self.command([ROOT/'bin/nano_vm', module], expected=74)
            for language in ('c', 'nano'):
                output = work/('rebuilt.'+language)
                self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output])
                if language == 'c':
                    self.native(output, work/'native')
                else:
                    rebuilt = work/'rebuilt.nvm'
                    self.command([ROOT/'bin/nano_virt', output, '--emit-nvm', '-o', rebuilt])
                    self.command([ROOT/'bin/nano_vm', rebuilt], expected=74)

    def test_verified_loop_outside_slice_refuses(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-outside-') as tmp:
            work = Path(tmp)
            text = SOURCE.replace('set counter (+ counter 17)',
                                  'while (< counter (+ 17 1)) { set counter (+ counter 17) }')
            source, module = work/'loop.nano', work/'loop.nvm'
            source.write_text(text)
            self.command([ROOT/'bin/nano_virt', source, '--emit-nvm', '-o', module])
            self.command([ROOT/'bin/nano_vm', module], expected=74)
            for language in ('c', 'nano'):
                output = work/('previous.'+language)
                output.write_text('previous checked output')
                result = run_captured([ROOT/'bin/nvm2hl', '--language', language,
                                      module, '-o', output], cwd=ROOT, timeout=30, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_text(), 'previous checked output')
                self.assertIn('pure scalar owned pretest', result.stderr)

    def test_corrupt_ownership_and_regions_preserve_prior_output(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-refuse-') as tmp:
            work = Path(tmp)
            module = self.source_module(work)
            facts = self.facts(module)
            original = module.read_bytes()
            header, count = struct.unpack_from('<II', original, 20)
            sections = {}
            for i in range(count):
                kind, _, offset, size = struct.unpack_from('<IIQQ', original, header + 24*i)
                sections[kind] = (offset, size)
            code = sections[6][0]
            instructions = facts['functions'][0]['code']
            def site(op, n=0):
                return [i for i in instructions if i['op'] == op][n]
            cases = {
                'missing-resource': (sections[13][0]+8, b'\x00'),
                'ordinary-authority': (sections[13][0]+8, b'\x01'),
                'wrong-nominal': (code+site('OWN_PACK')['pc']+1, struct.pack('<I',1)),
                'duplicate-owner': (code+site('OWN_STORE_LOCAL')['pc'], b'\x07\x00\x00'),
                'use-after-move': (code+site('OWN_MOVE_LOCAL',1)['pc']+1,
                                   struct.pack('<H',site('OWN_MOVE_LOCAL')['arg'])),
                'missing-region': (code+site('REGION_BEGIN')['pc'], b'\x00'),
                'unclosed-region': (code+site('REGION_END')['pc'], b'\x00'),
                'wrong-field': (code+site('REF_GET')['pc']+3, struct.pack('<H',99)),
                'wrong-owner': (code+site('BORROW_LOCAL_SHARED')['pc']+3,
                                struct.pack('<H',site('OWN_MOVE_LOCAL')['arg'])),
            }
            for name, (offset, payload) in cases.items():
                data = bytearray(original)
                data[offset:offset+len(payload)] = payload
                struct.pack_into('<I', data, 36, zlib.crc32(data[header:]))
                malformed = work/(name+'.nvm')
                malformed.write_bytes(data)
                for language in ('c','nano'):
                    with self.subTest(case=name, language=language):
                        out = work/('previous.'+language)
                        out.write_text('previous checked output')
                        result = run_captured([ROOT/'bin/nvm2hl', '--language', language,
                                               malformed, '-o', out], cwd=ROOT, timeout=30, text=True)
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertEqual(out.read_text(), 'previous checked output')
                        self.assertIn('I did not publish', result.stderr)
                        self.assertNotIn('checksum', result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
