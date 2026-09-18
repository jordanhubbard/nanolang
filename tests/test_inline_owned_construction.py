"""I stage exact inline children before later constructor expressions."""
import os
from pathlib import Path
import subprocess
import unittest
from tests import test_source_borrow_emission as support

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/nanoisa/fixtures/source_inline_owned_children.nano'


class InlineOwnedConstruction(support.SourceBorrowEmission):
    @classmethod
    def tearDownClass(cls):
        # I retain first failures and the exact produced artifacts for review.
        cls.temporary._finalizer.detach()
        print(f'I retain inline owner fixtures at {cls.work}')

    def test_ordered_children_and_all_selected_shadows(self):
        normal, shadow = self.graph_positive('inline-children', FIXTURE.read_text(), b'NFL', b'NFL')
        self.assertIn('OWN_PACK 2', normal)
        self.assertIn('OWN_PACK 2', shadow)
        empty = '''resource struct Empty {}
struct Box { child: Empty }
fn main() -> int { let box: Box = Box { child: Empty {} } let Box { child } = box let Empty {} = child return 0 }
shadow main { assert (== (main) 0) }
'''
        self.graph_positive('inline-empty', empty, b'', b'')

    def test_refusals_and_physical_local_budget(self):
        base = FIXTURE.read_text()
        cases = {
            'nominal': base.replace('struct Pair', 'resource struct Other { value: int }\nstruct Pair', 1)
                .replace('left: Leaf { value: (mark 3 "L") }', 'left: Other { value: 3 }'),
            'duplicate': base.replace('right: (factory 7), left:', 'right: (factory 7), right:'),
            'missing': base.replace('enabled: true, pair:', 'pair:', 1),
            'moved': base.replace('left: named, right: other', 'left: named, right: named'),
            'unconsumed': base.replace('left: named, right: other', 'left: Leaf { value: 9 }, right: other'),
        }
        # 129 leaf children need at least258 staging slots before parent packing.
        count = 129
        fields = ', '.join(f'f{i}: Leaf' for i in range(count))
        values = ', '.join(f'f{i}: Leaf {{ value: {i} }}' for i in range(count))
        names = ', '.join(f'f{i}' for i in range(count))
        consume = ' '.join(f'let Leaf {{ value }} = f{i} assert (== value {i})' for i in range(count))
        cases['local_budget'] = (f'resource struct Leaf {{ value: int }}\nstruct Wide {{ {fields} }}\n'
            f'fn main() -> int {{ let owner: Wide = Wide {{ {values} }} let Wide {{ {names} }} = owner {consume} return 0 }}\n'
            'shadow main { assert (== (main) 0) }\n')
        for name, text in cases.items():
            source = self.work / f'inline-refused-{name}.nano'
            source.write_text(text)
            for compiler in [ROOT / 'bin' / x for x in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2')] + self.emitters:
                with self.subTest(case=name, compiler=compiler.name):
                    output = self.work / 'inline-prior.nvm'
                    output.write_bytes(b'previous verified publication')
                    args = [compiler, source]
                    if compiler not in self.emitters:
                        args.append('--emit-nvm')
                    result = subprocess.run([*args, '-o', output], cwd=ROOT, capture_output=True, text=True, timeout=180)
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b'previous verified publication')
                    self.assertNotRegex(result.stdout + result.stderr, r'(?i)parse (?:error|failed)|unexpected token')
                    if name == 'local_budget':
                        self.assertRegex(result.stdout + result.stderr, r'256 local slots')

    def test_emitted_partial_construction_cleanup(self):
        source = self.work / 'inline-cleanup.nano'
        source.write_text(FIXTURE.read_text())
        module, generated = self.work / 'inline-cleanup.nvm', self.work / 'inline-cleanup.c'
        self.command(ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', module)
        self.command(ROOT / 'bin/nvm2c', module, '-o', generated)
        harness = self.work / 'inline-cleanup-harness.c'
        harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
static size_t live, attempts, fail_at;
static void *allocate(size_t count,size_t width) {
 assert(count && width);
 if(++attempts==fail_at)return NULL;
 void *p=calloc(count,width);if(p)live++;return p;
}
static void release(void *p){if(p){assert(live);live--;free(p);}}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
#include "inline-cleanup.c"
int main(void) {
 int64_t value=-91;
 assert(nvm_owned_entry(&value)==0 && value==0 && live==0);
 size_t total=attempts;assert(total>=8);
 for(size_t budget=1;budget<=total;budget++) {
  fail_at=budget;attempts=0;value=-91;
  assert(nvm_owned_entry(&value)==1 && value==-91 && live==0);
  fail_at=0;attempts=0;value=-91;
  assert(nvm_owned_entry(&value)==0 && value==0 && live==0);
 }
 return 0;
}
''')
        binary = self.work / 'inline-cleanup'
        self.command(os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                     '-fsanitize=address,undefined', '-fno-omit-frame-pointer', harness, '-o', binary)
        result = subprocess.run([binary], cwd=ROOT, capture_output=True, text=True, timeout=30,
                                env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('Sanitizer', result.stderr)


def load_tests(loader, standard_tests, pattern):
    # I select only this child, not the inherited full source suite twice.
    return unittest.TestSuite(InlineOwnedConstruction(name) for name in (
        'test_ordered_children_and_all_selected_shadows',
        'test_refusals_and_physical_local_budget',
        'test_emitted_partial_construction_cleanup'))
