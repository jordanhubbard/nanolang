"""I check executable scalar guards before claiming immutable passive inputs."""
from pathlib import Path
import resource
import signal
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def fixture(version=2, tags=('int', 'bool', 'string', 'float')):
    numbers = {'int': 1, 'bool': 4, 'string': 5, 'float': 3}
    pushes = {'int': 'PUSH_I64 42', 'bool': 'PUSH_BOOL 1',
              'string': 'PUSH_STR text', 'float': 'PUSH_F64 1.5'}
    n = len(tags)
    # Each guard is six bytes; an ordinary branch follows the guard prefix.
    entry = 6*n + 5
    end = entry + 6*n
    words = [version, 1, 1, 0, entry, end, n]
    for i in range(n):
        words += [entry + 6*i, entry + 6*(i+1), n+i, 0, 1, 0, 0, i]
    claim = struct.pack('<' + 'I'*len(words), *words).hex()
    guards = ''.join(f'LOAD_LOCAL {i}\nTYPE_CHECK {numbers[t]}\nASSERT\n'
                     for i, t in enumerate(tags))
    nodes = ''.join(f'LOAD_LOCAL {i}\nSTORE_LOCAL {n+i}\n' for i in range(n))
    output = ''.join(f'LOAD_LOCAL {n+i}\nPRINTLN\n' for i in range(n))
    args = '\n'.join(pushes[t] for t in tags)
    return (f'.string text "guarded"\n.entry 1\n'
            f'.function helper {n} {2*n} 0 int 1\n{guards}JMP entry\nentry:\n'
            f'{nodes}{output}PUSH_I64 0\nRET\n.end\n'
            f'.parameters 0 {" ".join(tags)}\n'
            f'.function main 0 0 0 int 1\n{args}\nCALL helper\nRET\n.end\n'
            f'.passive "{claim}"\n')


class PassiveInputs(unittest.TestCase):
    def command(self, *args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, timeout=90)

    def assemble(self, directory, source, accepted=True):
        text = directory/'input.nasm'
        module = directory/'input.nvm'
        text.write_text(source)
        result = self.command(ROOT/'bin/nanoisa', 'asm', text, '-o', module)
        if accepted:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'verif', result.stderr.lower())
        return module

    def paired_roundtrip(self, source, output):
        with tempfile.TemporaryDirectory(prefix='nano-passive-inputs-') as tmp:
            directory = Path(tmp)
            module = self.assemble(directory, source)
            vm = self.command(ROOT/'bin/nano_vm', module)
            self.assertEqual(vm.returncode, 0, vm.stderr)
            self.assertEqual(vm.stdout, output)
            original = module.read_bytes()
            dump = self.command(ROOT/'bin/nanoisa', 'dump', module)
            self.assertEqual(dump.returncode, 0, dump.stderr)
            module = self.assemble(directory, dump.stdout.decode())
            self.assertEqual(module.read_bytes(), original)
            generated, native = directory/'input.c', directory/'input'
            translated = self.command(ROOT/'bin/nvm2c', module, '-o', generated)
            self.assertEqual(translated.returncode, 0, translated.stderr)
            compiled = self.command('cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                                    generated, '-lm', '-o', native)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            ran = self.command(native)
            self.assertEqual(ran.returncode, 0, ran.stderr)
            self.assertEqual(ran.stdout, output)

    def test_four_guarded_inputs_execute_and_roundtrip(self):
        self.paired_roundtrip(fixture(), b'42\ntrue\nguarded\n1.5\n')

    def test_guarded_input_feeds_flow_dependency(self):
        words = [2,1,2,0,11,33,2, 11,17,1,0,1,0,0,0,
                 17,33,2,1,0,0,0,0]
        claim = struct.pack('<' + 'I'*len(words), *words).hex()
        source = ('.entry 1\n.function helper 1 3 0 int 1\n'
                  'LOAD_LOCAL 0\nTYPE_CHECK 1\nASSERT\nJMP entry\nentry:\n'
                  'LOAD_LOCAL 0\nSTORE_LOCAL 1\n'
                  'LOAD_LOCAL 1\nPUSH_I64 1\nADD\nSTORE_LOCAL 2\n'
                  'LOAD_LOCAL 2\nPRINTLN\nPUSH_I64 0\nRET\n.end\n'
                  '.parameters 0 int\n.function main 0 0 0 int 1\n'
                  'PUSH_I64 42\nCALL helper\nRET\n.end\n'
                  f'.passive "{claim}"\n')
        self.paired_roundtrip(source, b'43\n')

    def test_actual_argument_is_checked_before_node_output(self):
        # The ordinary VM accepts a call whose runtime tag differs from the
        # declaration. The executable guard must stop it before any node output.
        source = fixture(tags=('int',)).replace('PUSH_I64 42', 'PUSH_BOOL 1')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            module = self.assemble(directory, source)
            result = self.command(ROOT/'bin/nano_vm', module)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
            self.assertIn(b'assert', result.stderr.lower())
            generated, native = directory/'wrong.c', directory/'wrong'
            result = self.command(ROOT/'bin/nvm2c', module, '-o', generated)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.command('cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                                  generated, '-lm', '-o', native)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([native], capture_output=True, timeout=30,
                                    preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            self.assertEqual(result.returncode, -signal.SIGABRT)
            self.assertEqual(result.stdout, b'')

    def test_older_and_unknown_record_versions_refuse_external_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            for version in (1, 3):
                with self.subTest(version=version):
                    self.assemble(Path(tmp), fixture(version), accepted=False)

    def test_incomplete_or_inconsistent_guard_claims_are_refused(self):
        source = fixture(tags=('int',))
        cases = {
            'missing assertion': source.replace('ASSERT\n', 'POP\n'),
            'different guard tag': source.replace('TYPE_CHECK 1', 'TYPE_CHECK 4'),
            'unknown declaration': source.replace('.parameters 0 int', '.parameters 0 void').replace('TYPE_CHECK 1', 'TYPE_CHECK 0'),
            'unsupported declaration': source.replace('.parameters 0 int', '.parameters 0 u8').replace('TYPE_CHECK 1', 'TYPE_CHECK 2'),
            'parameter write': source.replace('LOAD_LOCAL 1\nPRINTLN',
                                             'PUSH_I64 3\nSTORE_LOCAL 0\nLOAD_LOCAL 1\nPRINTLN'),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in cases.items():
                with self.subTest(name=name):
                    self.assemble(Path(tmp), text, accepted=False)


if __name__ == '__main__':
    unittest.main()
