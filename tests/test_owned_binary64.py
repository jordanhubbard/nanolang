"""I retain binary64 scalar semantics and owner cleanup in one verified module."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class OwnedBinary64(unittest.TestCase):
    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_owned_scalar_float_matrix_and_cleanup(self):
        cc = os.environ.get('CC', 'cc')
        with tempfile.TemporaryDirectory(prefix='nano-owned-binary64-') as directory:
            tmp = Path(directory)
            generated = self.checked([ROOT/'obj/test_owned_binary64', tmp])
            print(generated.stdout, end='')
            cases = [line.split() for line in generated.stdout.splitlines() if line.startswith('case ')]
            self.assertEqual(len(cases), 5)
            for _, index, status in cases:
                with self.subTest(case=index):
                    artifact, source = tmp/f'case{index}.nvm', tmp/f'case{index}.c'
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', artifact])
                    vm = subprocess.run([ROOT/'bin/nano_vm', artifact], capture_output=True, timeout=90)
                    self.assertEqual(vm.returncode, 1 if status == '2' else 0, vm.stderr)
                    rebuilt = tmp/f'rebuilt{index}.c'
                    self.checked([ROOT/'bin/nvm2c', artifact, '-o', rebuilt])
                    self.assertEqual(source.read_bytes(), rebuilt.read_bytes())
                    harness = tmp/f'harness{index}.c'
                    policy = "" if int(index) < 2 else """
 double nan=nown_float_bits(UINT64_C(0x7ff8000000000042));
 double special[]={nano_rt_f64_add(nan,1.0),nano_rt_f64_sub(nan,1.0),nano_rt_f64_mul(nan,1.0),nano_rt_f64_div(nan,1.0),nano_rt_f64_div(nan,-0.0)};
 for(size_t i=0;i<5;i++){uint64_t observed;memcpy(&observed,&special[i],8);assert(observed==(i==4?0:UINT64_C(0x7ff8000000000000)));}
"""
                    harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
static size_t live,attempts,fail_at;
static void *allocate(size_t n,size_t s){if(++attempts==fail_at)return NULL;void *p=calloc(n,s);if(p)live++;return p;}
static void release(void *p){assert(live);live--;free(p);}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
''' + f'#include "{source.name}"\n' + '''int main(void){
POLICY
 const uint64_t bits[]={0,UINT64_C(0x8000000000000000),1,UINT64_C(0x0010000000000000),UINT64_C(0x7fefffffffffffff),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000042),UINT64_C(0xfff80000000000ff)};
 for(size_t i=0;i<sizeof(bits)/sizeof(bits[0]);i++){double value=nown_float_bits(bits[i]);uint64_t observed;memcpy(&observed,&value,8);assert(observed==bits[i]);}
 for(unsigned i=0;i<8;i++){int64_t result=-91;int status=nvm_owned_entry(&result);assert(status==EXPECTED_STATUS);assert(result==EXPECTED_RESULT);assert(live==0);}
 for(fail_at=1;fail_at<16;fail_at++){attempts=0;int64_t result=-91;int status=nvm_owned_entry(&result);assert(live==0);if(status!=1){assert(status==EXPECTED_STATUS);break;}assert(result==-91);}
 assert(fail_at<16);return 0;
}
'''.replace('POLICY', policy).replace('EXPECTED_STATUS', status).replace('EXPECTED_RESULT', '-91' if status == '2' else '0'))
                    binary = tmp/f'native{index}'
                    self.checked([cc, '-std=c11', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                                  '-g', harness, '-o', binary])
                    self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'})
                    self.checked([cc, '-std=c11', '-Wall', '-Wextra', '-Werror', source, '-o', binary])
                    native = subprocess.run([binary], capture_output=True, timeout=90)
                    self.assertEqual(native.returncode, vm.returncode, native.stderr)

if __name__ == '__main__':
    unittest.main()
