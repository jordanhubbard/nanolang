"""I retain mandatory assertion behavior and terminal ownership cleanup."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OwnedAssertions(unittest.TestCase):
    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_paired_assertions_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-assert-') as name:
            tmp = Path(name)
            run = self.checked([os.environ.get('NANO_OWNED_ASSERT_TEST', ROOT/'obj/test_owned_assertions'), tmp])
            print(run.stdout, end='')
            cases = [line.split() for line in run.stdout.splitlines() if line.startswith('case ')]
            self.assertEqual(len(cases), 5)
            for _, index, succeeds in cases:
                with self.subTest(case=index):
                    succeeds = succeeds == '1'
                    artifact, generated = tmp/f'case{index}.nvm', tmp/f'case{index}.c'
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', artifact])
                    for flags in ([], ['--check-shadows']):
                        run = subprocess.run([ROOT/'bin/nano_vm', *flags, artifact], cwd=ROOT,
                                             capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode == 0, succeeds, run.stdout + run.stderr)
                        if not succeeds:
                            self.assertIn('Assertion failed', run.stdout + run.stderr)
                    rebuilt = tmp/f'rebuilt{index}.c'
                    self.checked([ROOT/'bin/nvm2c', artifact, '-o', rebuilt])
                    self.assertEqual(generated.read_bytes(), rebuilt.read_bytes())
                    harness = tmp/f'harness{index}.c'
                    harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
static size_t live;
static void *allocate(size_t n,size_t s){void *p=calloc(n,s);if(p)live++;return p;}
static void release(void *p){assert(live);live--;free(p);}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
''' + f'#include "{generated.name}"\n' + '''int main(void){
 for(unsigned i=0;i<20;i++){
  int64_t result=-91;int status=nvm_owned_entry(&result);
  assert(status==EXPECTED_STATUS);assert(result==EXPECTED_RESULT);assert(live==0);
 }
 return 0;
}
'''.replace('EXPECTED_STATUS', '0' if succeeds else '2')
   .replace('EXPECTED_RESULT', '0' if succeeds else '-91'))
                    binary = tmp/f'check{index}'
                    self.checked([os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-g', harness, '-o', binary])
                    self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'})
                    self.checked([os.environ.get('CC', 'cc'), '-std=c11', '-Wall', '-Wextra', '-Werror', generated, '-o', binary])
                    native = subprocess.run([binary], capture_output=True, timeout=30)
                    self.assertEqual(native.returncode == 0, succeeds)


if __name__ == '__main__':
    unittest.main()
