"""I keep consuming arguments unique through normal and terminal cleanup."""
import os
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def leak_detection_for(compiler_identity, platform):
    """I disable LSan only for the Apple sanitizer runtime that rejects it."""
    if platform == 'darwin' and 'Apple clang version' in compiler_identity:
        return '0'
    return '1'


class OwnedStringFields(unittest.TestCase):
    def checked(self, args, **kwargs):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_leak_detection_policy_is_compiler_specific(self):
        self.assertEqual(leak_detection_for('Apple clang version 21.0.0', 'darwin'), '0')
        self.assertEqual(leak_detection_for('clang version 23.0.0', 'darwin'), '1')
        self.assertEqual(leak_detection_for('Apple clang version 21.0.0', 'linux'), '1')

    def test_retained_string_fields_and_cleanup(self):
        compiler = os.environ.get('CC', 'cc')
        identity = self.checked([compiler, '--version'])
        leak_detection = leak_detection_for(identity.stdout + identity.stderr, sys.platform)
        print(f'owned-result sanitizer compiler={compiler} detect_leaks={leak_detection}')
        retained = os.environ.get('NANO_OWNED_STRING_FIELD_DIR')
        context = nullcontext(retained) if retained else tempfile.TemporaryDirectory(prefix='nano-owned-string-fields-')
        with context as name:
            tmp = Path(name)
            run = self.checked([os.environ.get('NANO_OWNED_STRING_FIELD_TEST', ROOT/'obj/test_owned_string_fields'), tmp])
            print(run.stdout, end='')
            print(run.stderr, end='')
            cases = [line.split() for line in run.stdout.splitlines() if line.startswith('case ')]
            self.assertEqual(len(cases), 4)
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
 assert(nown_string_equal(NULL,NULL));
 assert(!nown_string_new((const unsigned char *)"",SIZE_MAX));
 nown_string *a=nown_string_new((const unsigned char *)"same",4);
 nown_string *b=nown_string_new((const unsigned char *)"same",4);
 nown_string *c=nown_string_new((const unsigned char *)"sam",3);
 assert(a&&b&&c&&a!=b&&nown_string_equal(a,b)&&!nown_string_equal(a,c)&&!nown_string_equal(a,NULL));
 nown_value va={.string=a};a->refs=SIZE_MAX;assert(!nown_retain(va));assert(a->refs==SIZE_MAX);a->refs=1;
 nown_release(va);nown_release((nown_value){.string=b});nown_release((nown_value){.string=c});assert(!live);
 for(unsigned i=0;i<20;i++){
  int64_t result=-91;int status=nvm_owned_entry(&result);
  assert(status==EXPECTED_STATUS);assert(result==EXPECTED_RESULT);assert(live==0);
 }
 for(fail_at=1;fail_at<128;fail_at++){
  attempts=0;int64_t result=-91;int status=nvm_owned_entry(&result);
  assert(live==0);
  fprintf(stderr,"string native fault=%zu status=%d live=%zu\\n",fail_at,status,live);
  if(status!=1){assert(status==EXPECTED_STATUS);break;}
  assert(result==-91);
  size_t saved=fail_at;fail_at=0;result=-91;
  assert(nvm_owned_entry(&result)==EXPECTED_STATUS);assert(result==EXPECTED_RESULT);assert(!live);fail_at=saved;
 }
 assert(fail_at<128);
 return 0;
}
'''.replace('EXPECTED_STATUS', status)
   .replace('EXPECTED_RESULT', value if succeeds else '-91'))
                    binary = tmp/f'check{index}'
                    self.checked([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-g', harness, '-o', binary])
                    native_check = self.checked([binary], env={**os.environ,
                                                'ASAN_OPTIONS': f'detect_leaks={leak_detection}:halt_on_error=1'})
                    print(native_check.stderr, end='')
                    self.checked([compiler, '-std=c11', '-Wall', '-Wextra', '-Werror', generated, '-o', binary])
                    native = subprocess.run([binary], capture_output=True, timeout=30)
                    self.assertEqual(native.returncode, int(value) if succeeds else 1)


if __name__ == '__main__':
    unittest.main()
