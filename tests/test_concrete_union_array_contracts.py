"""I check concrete record-array payload identity before publishing artifacts."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DECL = '''struct Plain { value: int }
struct Other { value: int }
union Box<T> { Some { values: array<T> } }
union Envelope<T> { Some { item: Box<T> } }
'''
END = '\nshadow main { assert (== (main) 0) }\n'

class ConcreteUnionArrays(unittest.TestCase):
    def test_wrong_concrete_contexts(self):
        programs = [
            'fn main() -> int { let rows: array<array<Other>> = [[Other { value: 7 }]] let value: Box<array<Plain>> = Box.Some { values: rows } return 0 }',
            'fn main() -> int { let value: Box<Plain> = Box.Some { values: [Other { value: 7 }] } return 0 }',
            'let value: Box<Plain> = Box.Some { values: [Other { value: 7 }] } fn main() -> int { return 0 }',
            'fn take(value: Box<Plain>) -> int { return 0 } shadow take { assert true } fn main() -> int { return (take Box.Some { values: [Other { value: 7 }] }) }',
            'fn make() -> Box<Plain> { return Box.Some { values: [Other { value: 7 }] } } shadow make { assert true } fn main() -> int { return 0 }',
            'fn main() -> int { let mut value: Box<Plain> = Box.Some { values: [] } set value Box.Some { values: [Other { value: 7 }] } return 0 }',
            'fn main() -> int { let value: Envelope<Plain> = Envelope.Some { item: Box.Some { values: [Other { value: 7 }] } } return 0 }',
            'fn main() -> int { let value: Box<array<Plain>> = Box.Some { values: [[Other { value: 7 }]] } return 0 }',
        ]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source, output = work/'case.nano', work/'result'
            for index, program in enumerate(programs):
                source.write_text(DECL + program + END)
                for compiler in ('nanoc_c', 'nano_virt'):
                    with self.subTest(case=index, compiler=compiler):
                        output.write_bytes(b'prior artifact')
                        command = [ROOT/'bin'/compiler, source, '-o', output]
                        if compiler == 'nano_virt': command.append('--emit-nvm')
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('declared nominal record type', result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior artifact')

    def test_matching_concrete_contexts_execute(self):
        programs = [
            'fn main() -> int { let rows: array<array<Plain>> = [[Plain { value: 7 }]] let value: Box<array<Plain>> = Box.Some { values: rows } return 0 }',
            'fn main() -> int { let value: Box<Plain> = Box.Some { values: [Plain { value: 7 }] } return 0 }',
            'fn make() -> Box<Plain> { return Box.Some { values: [Plain { value: 7 }] } } shadow make { let value: Box<Plain> = (make) assert true } fn main() -> int { let value: Box<Plain> = (make) return 0 }',
            'fn main() -> int { let value: Box<array<Plain>> = Box.Some { values: [[Plain { value: 7 }]] } return 0 }',
            'struct T { other: string } fn main() -> int { let value: Box<Plain> = Box.Some { values: [Plain { value: 7 }] } return 0 }',
            'fn main() -> int { let value: Box<Plain> = Box.Some { values: [] } return 0 }',
        ]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for index, program in enumerate(programs):
                with self.subTest(case=index):
                    source, output = work/'case.nano', work/'result'
                    source.write_text(DECL + program + END)
                    result = subprocess.run([ROOT/'bin/nanoc_c', source, '-o', output], cwd=ROOT,
                                            capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    result = subprocess.run([output], capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

if __name__ == '__main__': unittest.main()
