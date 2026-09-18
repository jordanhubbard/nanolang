"""I retain explicit constructor instances without replacing them with context."""
import subprocess
from tests import test_scalar_match_values as matches

ROOT = matches.ROOT


class ExplicitGenericConstructors(matches.ScalarMatchValues):
    def test_explicit_instances_in_locals_calls_returns_and_shadows(self):
        self.paired('''union Box<T> { Some { value: T }, None {} }
union Choice<T,E> { Left { value: T }, Right { value: E } }
fn fresh() -> Box<int> { return Box<int>.Some { value: 7 } }
shadow fresh { assert (== (read (fresh)) 7) }
fn read(value: Box<int>) -> int {
 match value { Some(v) => { return v.value } None(n) => { return 0 } }
}
shadow read { assert (== (read Box<int>.None {}) 0) }
fn choose(value: Choice<int,int>) -> int {
 match value { Left(v) => { return v.value } Right(v) => { return v.value } }
}
shadow choose { assert (== (choose Choice<int,int>.Right { value: 8 }) 8) }
fn main() -> int {
 let value: Box<int> = Box<int>.Some { value: 7 }
 let left: int = 3 let right: int = 9
 assert (left < right) assert (< left right)
 assert (== (read value) 7) assert (== (read (fresh)) 7)
 assert (== (choose Choice<int,int>.Left { value: 4 }) 4)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_nested_explicit_type_arguments(self):
        self.paired('''union Box<T> { Some { value: T }, None {} }
fn read(value: Box<array<int>>) -> int {
 match value { Some(v) => { return (at v.value 1) } None(n) => { return 0 } }
}
shadow read { assert (== (read Box<array<int>>.Some { value: [3,8] }) 8) }
fn main() -> int {
 let value: Box<array<int>> = Box<array<int>>.Some { value: [3,8] }
 assert (== (read value) 8) assert (== (read Box<array<int>>.None {}) 0)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_explicit_empty_instance_mismatches_preserve_output(self):
        cases = {
            'local': ('', 'let wrong: Box<int> = Box<string>.None {}'),
            'argument': ('', 'let wrong: int = (accept Box<string>.None {})'),
            'return': ('fn wrong() -> Box<int> { return Box<string>.None {} } shadow wrong { assert true }', 'let value: Box<int> = (wrong)'),
        }
        for name, (extra, body) in cases.items():
            source = self.work / ('explicit-wrong-' + name + '.nano')
            source.write_text('union Box<T> { Some { value: T }, None {} } '
                              'fn accept(value: Box<int>) -> int { return 0 } shadow accept { assert true } '
                              + extra + ' fn main() -> int { ' + body + ' return 0 } shadow main { assert true }')
            for tool in [*self.raw, ROOT/'bin/nano_virt', ROOT/'bin/nanoc_stage1', ROOT/'bin/nanoc_stage2']:
                with self.subTest(case=name, producer=tool.name):
                    output = self.work/'explicit-retained'
                    output.write_text('retained')
                    args = [tool, source]
                    if tool not in self.raw:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, text=True, capture_output=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(output.read_text(), 'retained')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse error')
                    self.assertRegex(result.stdout + result.stderr, r'(?i)(instance|union|type|constructor)')
