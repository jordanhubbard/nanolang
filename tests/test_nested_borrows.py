"""I borrow actual nested resource places and distinguish overlapping paths."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from tests.test_exclusive_borrows import PRELUDE

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = os.environ.get('NANO_BORROW_COMPILERS', 'nanoc_c,nanoc_stage1,nanoc_stage2').split(',')
SUPPORT = PRELUDE + '''struct Pair { left: Handle, right: Handle }
struct Wrapper { pair: Pair, number: int }
fn close_pair(pair: Pair) -> int { let Pair { left, right } = pair return (+ (consume left) (consume right)) }
shadow close_pair { assert (== (close_pair Pair { left: Handle { fd: 1 }, right: Handle { fd: 2 } }) 3) }
fn close_wrapper(wrapper: Wrapper) -> int { let Wrapper { pair, number } = wrapper return (+ (close_pair pair) number) }
shadow close_wrapper { assert (== (close_wrapper Wrapper { pair: Pair { left: Handle { fd: 1 }, right: Handle { fd: 2 } }, number: 3 }) 6) }
fn both(a: &mut Handle, b: &mut Handle) -> int { set a.fd (+ a.fd 1) set b.fd (+ b.fd 2) return (+ a.fd b.fd) }
shadow both { let mut a: Handle = Handle { fd: 1 } let mut b: Handle = Handle { fd: 2 } assert (== (both &mut a &mut b) 6) assert (== (+ (consume a) (consume b)) 6) }
fn observe(a: &mut Handle, n: int) -> int { set a.fd (+ a.fd n) return a.fd }
shadow observe { let mut a: Handle = Handle { fd: 1 } assert (== (observe &mut a 2) 3) assert (== (consume a) 3) }
fn share(a: &Handle, b: &mut Handle) -> int { set b.fd (+ a.fd b.fd) return b.fd }
shadow share { let a: Handle = Handle { fd: 1 } let mut b: Handle = Handle { fd: 2 } assert (== (share &a &mut b) 3) assert (== (+ (consume a) (consume b)) 4) }
'''
PAIR = 'let mut pair: Pair = Pair { left: Handle { fd: 10 }, right: Handle { fd: 20 } }\n'

class NestedBorrows(unittest.TestCase):
    def check(self, body, reject=False, extra='', files=None):
        source = SUPPORT + extra + '\nfn main() -> int {\n' + body + '\nreturn 0 }\nshadow main { assert (== (main) 0) }\n'
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-nested-borrow-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'program'
                for name, contents in (files or {}).items():
                    (Path(directory)/name).write_text(contents)
                path.write_text(source)
                output.write_text('prior artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=180)
                diagnostic = result.stdout + result.stderr
                if reject:
                    self.assertGreater(result.returncode, 0, diagnostic)
                    self.assertEqual(output.read_text(), 'prior artifact')
                    for marker in ('C compilation failed', 'Parse error', 'Parsing error', 'Failed to parse', 'Segmentation fault'):
                        self.assertNotIn(marker, diagnostic)
                else:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    ran = subprocess.run([output], capture_output=True, text=True, timeout=30)
                    self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_nested_addresses_and_disjoint_exclusive_fields(self):
        self.check('''let mut wrapper: Wrapper = Wrapper { pair: Pair { left: Handle { fd: 10 }, right: Handle { fd: 20 } }, number: 3 }
assert (== (bump &mut wrapper.pair.left) 11)
assert (== wrapper.pair.left.fd 11)
assert (== (both &mut wrapper.pair.left &mut wrapper.pair.right) 34)
assert (== (read &wrapper.pair.left) 12)
assert (== (close_wrapper wrapper) 37)''')

    def test_disjoint_read_and_shared_exclusive_arguments(self):
        self.check(PAIR + '''assert (== (observe &mut pair.left pair.right.fd) 30)
assert (== (share &pair.left &mut pair.right) 50)
assert (== (close_pair pair) 80)''')
        self.check(PAIR + '''assert (== (observe &mut pair.left (bump &mut pair.right)) 31)
assert (== (close_pair pair) 52)''')

    def test_same_place_overlap_and_later_conflicting_reads(self):
        for operation in ('(both &mut pair.left &mut pair.left)',
                          '(share &pair.left &mut pair.left)',
                          '(observe &mut pair.left (read &pair.left))',
                          '(observe &mut pair.left pair.left.fd)',
                          '(observe &mut pair.left (bump &mut pair.left))'):
            with self.subTest(operation=operation):
                self.check(PAIR + operation + '\n(close_pair pair)', reject=True)

    def test_whole_owner_moves_partial_moves_and_escape_are_refused(self):
        for operation in ('(observe &mut pair.left (close_pair pair))',
                          '(observe &mut pair.left (consume pair.left))',
                          'let escaped = &pair.left',
                          '(consume pair.left)'):
            with self.subTest(operation=operation):
                self.check(PAIR + operation + '\n(close_pair pair)', reject=True)

    def test_mutability_moved_root_and_nominal_identity(self):
        self.check(PAIR.replace('let mut pair', 'let pair') + '(bump &mut pair.left)\n(close_pair pair)', reject=True)
        self.check(PAIR + '(close_pair pair)\n(read &pair.left)', reject=True)
        self.check(PAIR + '(read &pair.absent)\n(close_pair pair)', reject=True)
        self.check('(bump &mut Pair { left: Handle { fd: 1 }, right: Handle { fd: 2 } }.left)', reject=True)
        extra = '''resource struct Other { fd: int }
struct ForeignPair { member: Other }
fn close_foreign(pair: ForeignPair) -> int { let ForeignPair { member } = pair let Other { fd } = member return fd }
shadow close_foreign { assert (== (close_foreign ForeignPair { member: Other { fd: 1 } }) 1) }
'''
        self.check('let mut pair: ForeignPair = ForeignPair { member: Other { fd: 1 } }\n(bump &mut pair.member)\n(close_foreign pair)', reject=True, extra=extra)

    def test_prefix_names_are_disjoint_and_imported_names_remain_nominal(self):
        extra = """struct PrefixPair { left: Handle, leftover: Handle }
fn close_prefix(pair: PrefixPair) -> int { let PrefixPair { left, leftover } = pair return (+ (consume left) (consume leftover)) }
shadow close_prefix { assert (== (close_prefix PrefixPair { left: Handle { fd: 1 }, leftover: Handle { fd: 2 } }) 3) }
"""
        self.check('let mut pair: PrefixPair = PrefixPair { left: Handle { fd: 10 }, leftover: Handle { fd: 20 } }\nassert (== (both &mut pair.left &mut pair.leftover) 33)\nassert (== (close_prefix pair) 33)', extra=extra)
        module = """pub struct Handle { fd: int }
pub struct Envelope { member: Handle }
pub fn make() -> Envelope { return Envelope { member: Handle { fd: 1 } } }
shadow make { let value: Envelope = (make) assert (== value.member.fd 1) }
"""
        self.check('let mut pair: other.Envelope = (other.make)\n(bump &mut pair.member)',
                   reject=True, extra='module "other.nano" as other\n', files={'other.nano': module})

    def test_argument_branch_holds_restore_before_whole_consumption(self):
        self.check(PAIR + '''let choice: Choice = Choice.Some { value: 2 }
assert (== (observe &mut pair.left (match choice { Some(v) => { let ignored: int = (read &pair.right) v.value } None(empty) => { 0 } })) 12)
assert (== (close_pair pair) 32)''')

if __name__ == '__main__':
    unittest.main()
