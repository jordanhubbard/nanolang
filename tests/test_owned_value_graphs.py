"""I keep consuming arguments unique through normal and terminal cleanup."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import os
from pathlib import Path
import subprocess
import shlex
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OwnedValueGraphs(unittest.TestCase):
    binary_environment = 'NANO_OWNED_VALUE_GRAPH_TEST'
    binary_default = ROOT/'obj/test_owned_value_graphs'
    case_count = 12

    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_owned_value_graphs_and_cleanup(self):
        compiler = shlex.split(os.environ.get('NANOLANG_GUARD_SAN_CC', os.environ.get('CC', 'cc')))
        with tempfile.TemporaryDirectory(prefix='nano-owned-graphs-') as name:
            tmp = Path(name)
            run = self.checked([os.environ.get(self.binary_environment, self.binary_default), tmp])
            print(run.stdout, end='')
            cases = [line.split() for line in run.stdout.splitlines() if line.startswith('case ')]
            self.assertEqual(len(cases), self.case_count)
            for _, index, status, value in cases:
                with self.subTest(case=index):
                    succeeds = status == '0'
                    artifact, generated = tmp/f'case{index}.nvm', tmp/f'case{index}.c'
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', artifact])
                    vm = subprocess.run([ROOT/'bin/nano_vm', artifact], capture_output=True, timeout=30)
                    self.assertEqual(vm.returncode, int(value) if succeeds else 1)
                    rebuilt = tmp/f'rebuilt{index}.c'
                    self.checked([ROOT/'bin/nvm2c', artifact, '-o', rebuilt])
                    self.assertEqual(generated.read_bytes(), rebuilt.read_bytes())
                    extra_path = tmp/f'case{index}.guard.c'
                    extra = extra_path.read_text() if extra_path.exists() else ''
                    harness = tmp/f'harness{index}.c'
                    harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
static size_t live,attempts,fail_at;
static void *allocate(size_t n,size_t s){if(++attempts==fail_at)return NULL;void *p=calloc(n,s);if(p)live++;return p;}
static void release(void *p){assert(live);live--;free(p);}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
''' + f'#include "{generated.name}"\n' + '''int main(void){
 for(unsigned i=0;i<20;i++){
  int64_t result=-91;int status=nvm_owned_entry(&result);
  assert(status==EXPECTED_STATUS);assert(result==EXPECTED_RESULT);assert(live==0);
 }
 EXTRA_CHECKS
 for(fail_at=1;fail_at<128;fail_at++){
  attempts=0;int64_t result=-91;int status=nvm_owned_entry(&result);
  assert(live==0);
  if(status!=1){assert(status==EXPECTED_STATUS);break;}
  assert(result==-91);
 }
 assert(fail_at<128);
 return 0;
}
'''.replace('EXTRA_CHECKS', extra).replace('EXPECTED_STATUS', status)
   .replace('EXPECTED_RESULT', value if succeeds else '-91'))
                    binary = tmp/f'check{index}'
                    self.checked([*compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-fno-omit-frame-pointer', '-g', harness, '-o', binary])
                    self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': asan_options("halt_on_error=1")})
                    self.checked([*compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', generated, '-o', binary])
                    native = subprocess.run([binary], capture_output=True, timeout=30)
                    self.assertEqual(native.returncode, int(value) if succeeds else 1)


if __name__ == '__main__':
    unittest.main()
