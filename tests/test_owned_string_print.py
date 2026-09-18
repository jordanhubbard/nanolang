"""I keep bounded owned string output byte-exact in VM and generated C."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = b"before\nbefore\n\nresource closed"


class OwnedStringPrint(unittest.TestCase):
    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0,
                         result.stdout.decode(errors='replace') + result.stderr.decode(errors='replace'))
        return result

    def test_vm_native_output_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='nano-owned-string-print-') as name:
            tmp = Path(name)
            fixture = self.checked([ROOT/'obj/test_owned_string_print', tmp])
            cases = [line.split() for line in fixture.stdout.decode().splitlines()
                     if line.startswith('case ')]
            self.assertEqual(cases, [['case', '0', '0', '42'], ['case', '1', '2', '42']])
            compiler = os.environ.get('CC', 'cc')
            identity = self.checked([compiler, '--version']).stdout.decode(errors='replace')
            leaks = '0' if os.uname().sysname == 'Darwin' and 'Apple clang version' in identity else '1'
            print(f'owned-string sanitizer compiler={compiler} detect_leaks={leaks}')
            for _, index, status, value in cases:
                with self.subTest(case=index):
                    succeeds = status == '0'
                    artifact, generated = tmp/f'case{index}.nvm', tmp/f'case{index}.c'
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', artifact])
                    vm = subprocess.run([ROOT/'bin/nano_vm', artifact], cwd=ROOT,
                                        capture_output=True, timeout=30)
                    self.assertEqual(vm.returncode, int(value) if succeeds else 1, vm.stderr.decode())
                    self.assertEqual(vm.stdout, EXPECTED)
                    rebuilt = tmp/f'rebuilt{index}.c'
                    self.checked([ROOT/'bin/nvm2c', artifact, '-o', rebuilt])
                    self.assertEqual(generated.read_bytes(), rebuilt.read_bytes())
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
 for(unsigned i=0;i<4;i++){
  int64_t result=-91;int status=nvm_owned_entry(&result);
  assert(status==EXPECTED_STATUS);assert(result==EXPECTED_RESULT);assert(live==0);
 }
 for(fail_at=1;fail_at<32;fail_at++){
  attempts=0;int64_t result=-91;int status=nvm_owned_entry(&result);
  assert(live==0);
  if(status!=1){assert(status==EXPECTED_STATUS);assert(result==EXPECTED_RESULT);break;}
  assert(result==-91);
 }
 assert(fail_at<32);return 0;
}
'''.replace('EXPECTED_STATUS', status)
   .replace('EXPECTED_RESULT', value if succeeds else '-91'))
                    binary = tmp/f'check{index}'
                    self.checked([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-g',
                                  harness, '-o', binary])
                    native = subprocess.run([binary], cwd=tmp, capture_output=True, timeout=30,
                                            env={**os.environ,
                                                 'ASAN_OPTIONS': f'detect_leaks={leaks}:halt_on_error=1'})
                    self.assertEqual(native.returncode, 0, native.stderr.decode(errors='replace'))
                    self.assertEqual(native.stdout, EXPECTED * 5)
                    self.checked([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                                  generated, '-o', binary])
                    plain = subprocess.run([binary], cwd=tmp, capture_output=True, timeout=30)
                    self.assertEqual(plain.returncode, int(value) if succeeds else 1,
                                     plain.stderr.decode(errors='replace'))
                    self.assertEqual(plain.stdout, EXPECTED)


if __name__ == '__main__':
    unittest.main()
