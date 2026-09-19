"""I qualify admitted mixed programs and preserve roots on native failure."""
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from tests.test_owned_string_fields import leak_detection_for
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MixedSamplesRuntime(unittest.TestCase):
    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_vm_native_lifecycle_and_allocations(self):
        compiler = os.environ.get('CC', 'cc')
        identity = self.checked([compiler, '--version'])
        leaks = leak_detection_for(identity.stdout + identity.stderr, sys.platform)
        retained = os.environ.get('NANO_MIXED_RUNTIME_DIR')
        context = nullcontext(retained) if retained else tempfile.TemporaryDirectory(prefix='nano-mixed-runtime-')
        with context as directory:
            work = Path(directory)
            result = self.checked([ROOT/'obj/test_mixed_samples_runtime', work])
            print(result.stdout, end='')
            cases = [line.split() for line in result.stdout.splitlines() if line.startswith('case ')]
            self.assertEqual(len(cases), 12)
            for _, index, status, value in cases:
                with self.subTest(case=index):
                    source = work/f'case{index}.c'
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', work/f'case{index}.nvm'])
                    rebuilt = work/f'rebuilt{index}.c'
                    self.checked([ROOT/'bin/nvm2c', work/f'case{index}.nvm', '-o', rebuilt])
                    self.assertEqual(source.read_bytes(), rebuilt.read_bytes())
                    harness = work/f'harness{index}.c'
                    harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
static size_t mixed_live,mixed_attempts,mixed_fail;
static void *mixed_malloc(size_t n){if(++mixed_attempts==mixed_fail)return NULL;void *p=malloc(n);if(p)mixed_live++;return p;}
static void *mixed_calloc(size_t n,size_t s){if(++mixed_attempts==mixed_fail)return NULL;void *p=calloc(n,s);if(p)mixed_live++;return p;}
static void mixed_free(void *p){if(p){assert(mixed_live);mixed_live--;free(p);}}
#define malloc mixed_malloc
#define calloc mixed_calloc
#define free mixed_free
#define NVM2C_NO_MAIN
''' + f'#include "{source.name}"\n' + '''
int main(void){
 for(unsigned repeat=0;repeat<4;repeat++){
  int64_t value=-91;int status=nvm_owned_entry(&value);
  assert(status==WANTED);assert(value==VALUE);assert(!mixed_live);
 }
 int reached=0;
 for(size_t fault=1;fault<256;fault++){
  mixed_attempts=0;mixed_fail=fault;int64_t value=-91;int status=nvm_owned_entry(&value);
  size_t attempted=mixed_attempts;mixed_fail=0;
  fprintf(stderr,"mixed native fault=%zu attempts=%zu status=%d live=%zu\\n",fault,attempted,status,mixed_live);
  assert(!mixed_live);
  if(attempted<fault){assert(status==WANTED);assert(value==VALUE);reached=1;break;}
  assert(status==1);assert(value==-91);
  value=-91;assert(nvm_owned_entry(&value)==WANTED);assert(value==VALUE);assert(!mixed_live);
 }
 assert(reached);return 0;
}
'''.replace('WANTED', status).replace('VALUE', value if status=='0' else '-91'))
                    for optimization in ('-O0', '-O2'):
                        binary = work/f'check{index}{optimization}'
                        self.checked([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                                      optimization, '-fsanitize=address,undefined',
                                      '-fno-omit-frame-pointer', '-g', harness, '-o', binary])
                        checked = self.checked([binary], env={**os.environ,
                            'ASAN_OPTIONS': f'detect_leaks={leaks}:halt_on_error=1'})
                        print(f'case={index} optimization={optimization}\n{checked.stderr}', end='')

if __name__ == '__main__': unittest.main()
