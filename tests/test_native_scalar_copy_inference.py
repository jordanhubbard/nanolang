"""I require copies to preserve inferred scalar storage and strict depth guards."""
from itertools import product
from pathlib import Path
import hashlib
import json
import struct
import zlib
import tempfile
import unittest

from tests.nanolang_translator_routes import checked, ROOT
from tests.native_process import run_captured


def source(called, sliced, written, guarded, operation):
    text = 'struct Item { variant: int, excluded: array<bool> }\n'
    text += 'fn select_flag(item: Item) -> bool {\n'
    if sliced:
        text += ' let copied: array<bool> = (array_slice item.excluded 0 1)\n'
    text += ' let mut selected: bool = (!= item.variant 0)\n'
    if guarded:
        text += ' if (== item.variant 65535) { set selected (at item.excluded 0) }\n'
    else:
        text += ' set selected (at item.excluded 0)\n'
    if written:
        text += ' (array_set '+('copied' if sliced else 'item.excluded')+' 0 selected)\n'
    result = {'identity': 'selected', 'and': '(and selected true)',
              'or': '(or selected false)', 'not': '(not selected)'}[operation]
    text += ' return '+result+'\n}\nfn main() -> int {\n'
    if called:
        call = '(select_flag Item { variant: 65535, excluded: [false] })'
        text += ' assert '+(call if operation == 'not' else '(not '+call+')')+'\n'
    text += ' return 0\n}\n'
    positive = '(select_flag Item { variant: 65535, excluded: [true] })'
    negative = '(select_flag Item { variant: 65535, excluded: [false] })'
    text += 'shadow select_flag { assert '+('(not '+positive+')' if operation == 'not' else positive)
    text += ' assert '+(negative if operation == 'not' else '(not '+negative+')')+' }\n'
    return text+'shadow main { assert (== (main) 0) }\n'


class ScalarCopyInference(unittest.TestCase):
    def test_complete_source_copy_matrix_and_prior_output(self):
        work = Path(tempfile.mkdtemp(prefix='nano-scalar-copy-regression-'))
        record = dict(scope='I require all valid source/PICK0 translations. I retain every failed assertion; '
                      'PICK1 must refuse before publication. I do not change C admission or shape constraints.',
                      cases=[], commands=[])
        prior = b'I retain my prior C output.\0\xff\n'

        def save():
            (work/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')

        def execute(command, label):
            index = len(record['commands'])
            result = run_captured(command, timeout=60, build=True, cwd=ROOT)
            (work/f'{index:04d}.stdout').write_bytes(result.stdout)
            (work/f'{index:04d}.stderr').write_bytes(result.stderr)
            record['commands'].append(dict(label=label, command=list(map(str, command)), exit=result.returncode,
                stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
            save()
            return result

        selected = None
        for called, sliced, written, guarded, operation in product((False, True), (False, True),
                (False, True), (False, True), ('identity', 'and', 'or', 'not')):
            name = f'called{int(called)}-slice{int(sliced)}-write{int(written)}-guard{int(guarded)}-{operation}'
            path = work/(name+'.nano')
            path.write_text(source(called, sliced, written, guarded, operation))
            row = dict(name=name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            record['cases'].append(row)
            save()
            with self.subTest(case=name):
                made = execute([ROOT/'bin/nano_virt', path, '-o', path.with_suffix('.nvm')], name+'-producer')
                self.assertEqual(made.returncode, 0, made.stderr.decode(errors='replace'))
                output = path.with_suffix('.c'); output.write_bytes(prior)
                translated = execute([ROOT/'bin/nvm2c', path.with_suffix('.nvm'), '-o', output], name+'-aot')
                row['module_sha256'] = hashlib.sha256(path.with_suffix('.nvm').read_bytes()).hexdigest()
                row['aot_exit'] = translated.returncode
                row['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
                row['prior_preserved'] = output.read_bytes() == prior
                save()
                if translated.returncode:
                    self.assertEqual(output.read_bytes(), prior)
                if not called and not sliced and written and not guarded and operation == 'and':
                    selected = path.with_suffix('.nvm')
                self.assertEqual(translated.returncode, 0, translated.stderr.decode(errors='replace'))
                self.assertNotEqual(output.read_bytes(), prior)
        self.assertIsNotNone(selected)
        assembly = checked([ROOT/'bin/nanoisa', 'dump', selected])
        self.assertEqual(assembly.count('  DUP\n'), 1)
        for depth in (0, 1):
            raw = work/f'pick-{depth}.nasm'
            raw.write_text(assembly.replace('  DUP\n', f'  PICK {depth}\n'))
            with self.subTest(pick_depth=depth):
                if depth:
                    raw.with_suffix('.nvm').write_bytes(prior)
                made = execute([ROOT/'bin/nanoisa', 'asm', raw, '-o', raw.with_suffix('.nvm')], f'pick-{depth}-assembler')
                if depth:
                    self.assertNotEqual(made.returncode, 0)
                    self.assertIn(b'stack underflow', made.stderr)
                    self.assertEqual(raw.with_suffix('.nvm').read_bytes(), prior)
                    module = bytearray((work/'pick-0.nvm').read_bytes())
                    self.assertEqual(module[:4], b'NVM\x02')
                    sections = struct.unpack_from('<I', module, 24)[0]
                    code = [struct.unpack_from('<IIQQ', module, 40+index*24)
                            for index in range(sections) if struct.unpack_from('<I', module, 40+index*24)[0] == 6]
                    self.assertEqual(len(code), 1)
                    start = code[0][2]+61
                    self.assertEqual(module[start:start+3], b'\xf4\0\0')
                    struct.pack_into('<H', module, start+1, 1)
                    struct.pack_into('<I', module, 36, zlib.crc32(module[40:]) & 0xffffffff)
                    raw.with_suffix('.nvm').write_bytes(module)
                    record['negative_mutation'] = dict(source='pick-0.nvm', target='pick-1.nvm', byte_offset=start+1,
                        prior_depth=0, depth=1, checksum=struct.unpack_from('<I', module, 36)[0],
                        sha256=hashlib.sha256(module).hexdigest())
                    save()
                else:
                    self.assertEqual(made.returncode, 0, made.stderr.decode(errors='replace'))
                verified = execute([ROOT/'bin/nano_vm', '--verify-only', raw.with_suffix('.nvm')], f'pick-{depth}-verify')
                output = raw.with_suffix('.c'); output.write_bytes(prior)
                translated = execute([ROOT/'bin/nvm2c', raw.with_suffix('.nvm'), '-o', output], f'pick-{depth}-aot')
                if depth:
                    self.assertNotEqual(verified.returncode, 0)
                    self.assertIn(b"stack underflow", verified.stderr)
                    self.assertNotEqual(translated.returncode, 0)
                    self.assertEqual(output.read_bytes(), prior)
                else:
                    self.assertEqual(verified.returncode, 0, verified.stderr.decode(errors='replace'))
                    if translated.returncode:
                        self.assertEqual(output.read_bytes(), prior)
                    self.assertEqual(translated.returncode, 0, translated.stderr.decode(errors='replace'))
                    self.assertNotEqual(output.read_bytes(), prior)
