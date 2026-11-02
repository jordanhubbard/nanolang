"""I preserve exact direct resource calls and their ordered operand snapshots."""
import copy
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

from tests import test_owned_reconstruction as local
from tests.native_process import run_captured

ROOT = local.ROOT
sys.path.insert(0, str(ROOT/'scripts'))
from nanoisa_owned_reconstruction import reconstruct_owned
from nanoisa_reconstruction import Refusal

SOURCE = '''resource struct Left { value: int, yes: bool }
resource struct Right { value: int, yes: bool }
fn make(value: int, yes: bool) -> Left { return Left { value: value, yes: yes } }
shadow make { let Left { value, yes } = (make 7 true) assert (== value 7) }
fn forward(owner: Left) -> Left { return owner }
shadow forward { let Left { value, yes } = (forward Left { value: 7, yes: true }) assert (== value 7) }
fn consume(owner: Left, extra: int) -> int { let Left { value, yes } = owner return (+ value extra) }
shadow consume { assert (== (consume Left { value: 7, yes: true } 3) 10) }
fn combine(left: Left, extra: int, right: Right) -> int {
 let first: int = (consume left 0)
 let Right { value, yes } = right
 return (+ (+ first value) extra)
}
shadow combine { assert (== (combine (make 5 true) 7 Right { value: 6, yes: true }) 18) }
fn flag(owner: Right) -> bool { let Right { value, yes } = owner return yes }
shadow flag { assert (flag Right { value: 1, yes: true }) }
fn main() -> int {
 let mut n: int = 3
 let owner: Left = (forward (make n true))
 set n (+ n 17)
 let result: int = (consume owner (+ n 1))
 let total: int = (combine (make 5 (flag Right { value: 1, yes: true })) (+ 3 4) Right { value: 6, yes: true })
 return (+ (+ result total) (consume (forward (make 30 true)) 2))
}
shadow main { assert (== (main) 74) }
'''


class OwnedCallReconstruction(unittest.TestCase):
    command = local.OwnedReconstruction.command
    native = local.OwnedReconstruction.native
    facts = local.OwnedReconstruction.facts

    def source_module(self, work, compiler=None, source=SOURCE):
        path, module = work/'calls.nano', work/'calls.nvm'
        path.write_text(source)
        self.command([compiler or ROOT/'bin/nano_virt', path, '--emit-nvm', '-o', module])
        return module

    def test_actual_calls_preserve_exact_signatures_order_and_results(self):
        compilers = [ROOT/'bin/nano_virt'] + [Path(x) for x in shlex.split(
            os.environ.get('NANOLANG_OWNED_RECONSTRUCTION_COMPILERS', ''))]
        for compiler in compilers:
            with self.subTest(compiler=str(compiler)), tempfile.TemporaryDirectory(prefix='owned-calls-') as tmp:
                work = Path(tmp)
                module = self.source_module(work, compiler)
                facts = self.facts(module)
                self.assertEqual(len(facts['functions']), 6)
                signatures = [(f['params'], f['result_type']) for f in facts['functions']]
                self.assertIn(([[8, 0]], [8, 0]), signatures)
                self.assertIn(([[8, 1]], [4, 4294967295]), signatures)
                self.assertIn(([[8, 0], [1, 4294967295], [8, 1]], [1, 4294967295]), signatures)
                call_count = sum(i['op'] == 'CALL' for f in facts['functions'] for i in f['code'])
                self.command([ROOT/'bin/nano_vm', module], expected=74)
                for language in ('c', 'nano'):
                    output = work/('rebuilt.'+language)
                    self.command([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output])
                    text = output.read_text()
                    self.assertNotRegex(text, r'\bgoto\b|nano_vm|nvm_blob|\bswitch\b|\bdispatch\b')
                    calls = re.findall(r'= (?:\()?nlr_f\d+_[a-zA-Z0-9_]+', text)
                    self.assertEqual(len(calls), call_count, text)
                    self.assertIn('NlrOwned0', text)
                    self.assertIn('NlrOwned1', text)
                    if language == 'c':
                        self.native(output, work/'native')
                    else:
                        rebuilt = work/'rebuilt.nvm'
                        self.command([ROOT/'bin/nano_virt', output, '--emit-nvm', '-o', rebuilt])
                        self.command([ROOT/'bin/nano_vm', rebuilt], expected=74)
                        regenerated = self.facts(rebuilt)
                        self.assertEqual(regenerated['layouts'], facts['layouts'])
                        for signature in signatures:
                            self.assertIn(signature, [(f['params'], f['result_type']) for f in regenerated['functions']])
                        native = work/'native.c'
                        self.command([ROOT/'bin/nvm2c', rebuilt, '-o', native])
                        self.native(native, work/'native')

    def test_scalar_suffix_cannot_copy_or_consume_staged_owner(self):
        with tempfile.TemporaryDirectory(prefix='owned-call-suffix-') as tmp:
            facts = self.facts(self.source_module(Path(tmp)))
            code = facts['functions'][0]['code']
            at = next(n for n, ins in enumerate(code) if ins['op'] == 'OWN_MOVE_LOCAL') + 1
            for op in ('DUP', 'POP', 'SWAP', 'ROT3', 'PICK', 'ROLL'):
                altered = copy.deepcopy(facts)
                altered['functions'][0]['code'].insert(at, {'op': op, 'arg': 0, 'args': [0], 'pc': 99999})
                for language in ('c', 'nano'):
                    with self.subTest(op=op, language=language), self.assertRaises(Refusal):
                        reconstruct_owned(altered, language)

    def test_terminal_helper_branch_preserves_call_graph(self):
        with tempfile.TemporaryDirectory(prefix='owned-call-branch-') as tmp:
            work = Path(tmp)
            source = SOURCE.replace('return owner', 'if true { return owner } else { return owner }')
            module = self.source_module(work, source=source)
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

    def test_generated_function_budget_refuses_atomically(self):
        cases = {
            'nano-function-budget': SOURCE.replace('fn main()',
                'fn extra(owner: Left) -> Left { return (forward owner) }\n'
                'shadow extra { assert (== (consume (extra (make 1 true)) 0) 1) }\nfn main()').replace(
                    '(+ 3 4)', '(+ n (- 7 n))'),
        }
        for name, source in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix='owned-calls-outside-') as tmp:
                work = Path(tmp)
                module = self.source_module(work, source=source)
                self.command([ROOT/'bin/nano_vm', module], expected=74)
                for language in ('nano',):
                    output = work/('previous.'+language)
                    output.write_text('previous checked output')
                    result = run_captured([ROOT/'bin/nvm2hl', '--language', language, module, '-o', output],
                                          cwd=ROOT, timeout=30, text=True)
                    self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(output.read_text(), 'previous checked output')
                    self.assertIn('I did not publish', result.stderr)
                    self.assertIn('reserve one of eight producer functions', result.stderr)

    def test_corrupt_call_transfers_refuse_atomically(self):
        with tempfile.TemporaryDirectory(prefix='owned-call-refusal-') as tmp:
            work = Path(tmp)
            module = self.source_module(work)
            facts, original = self.facts(module), module.read_bytes()
            header, count = struct.unpack_from('<II', original, 20)
            sections = {}
            for n in range(count):
                kind, _, offset, size = struct.unpack_from('<IIQQ', original, header+24*n)
                sections[kind] = (offset, size)
            code_base = sections[6][0]
            # I use the decoded main-relative offsets; no guessed function entry size.
            main = facts['functions'][0]['code']
            calls = [i for i in main if i['op'] == 'CALL']
            move = next(i for i in main if i['op'] == 'OWN_MOVE_LOCAL')
            ownership = sections[13][0]
            layout_count = struct.unpack_from('<I', original, ownership+4)[0]
            at = ownership + ((8+layout_count+3)//4)*4 + 4
            descriptors = []
            for fn in facts['functions']:
                locals_count, parameter_count = struct.unpack_from('<HH', original, at)
                self.assertEqual((locals_count, parameter_count), (fn['locals'], len(fn['params'])))
                descriptors.append(at+4)
                at += 4 + 8*(1+locals_count)
            helper = 4
            helper_call = next(i for i in facts['functions'][helper]['code'] if i['op'] == 'CALL')
            self.assertEqual(sum(f['size'] for f in facts['functions']), sections[6][1])
            helper_offset = sum(f['size'] for f in facts['functions'][:helper])
            cases = {
                'helper-cycle': (code_base+helper_offset+helper_call['pc']+1, struct.pack('<I', helper)),
                'borrowed-parameter-mode': (descriptors[2]+8+1, b'\x01'),
                'entry-cycle': (code_base+calls[0]['pc']+1, struct.pack('<I', 0)),
                'missing-callee': (code_base+calls[0]['pc']+1, struct.pack('<I', 99)),
                'wrong-result-nominal': (descriptors[1]+4, struct.pack('<I', 1)),
                'wrong-argument-nominal': (descriptors[2]+8+4, struct.pack('<I', 1)),
                'lost-owner': (code_base+move['pc'], b'\x00\x00\x00'),
                'duplicate-owner': (code_base+move['pc'], b'\x07'),
                'missing-resource-authority': (sections[13][0]+8, b'\x00'),
            }
            for name, (offset, payload) in cases.items():
                data = bytearray(original)
                data[offset:offset+len(payload)] = payload
                struct.pack_into('<I', data, 36, zlib.crc32(data[header:]))
                bad = work/(name+'.nvm'); bad.write_bytes(data)
                for language in ('c', 'nano'):
                    with self.subTest(case=name, language=language):
                        output = work/('previous.'+language);output.write_text('previous checked output')
                        result = run_captured([ROOT/'bin/nvm2hl', '--language', language, bad, '-o', output],
                                              cwd=ROOT, timeout=30, text=True)
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertEqual(output.read_text(), 'previous checked output')
                        self.assertIn('I did not publish', result.stderr)
                        self.assertNotIn('checksum', result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
