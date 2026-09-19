"""I retain STRING initialization across both ordinary branch arms."""
import os
from contextlib import nullcontext
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class OwnedStringJoins(unittest.TestCase):
    def checked(self, args, **kwargs):
        run = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                             text=True, timeout=90, **kwargs)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return run

    def test_initialized_arms_and_checked_missing_arm(self):
        retained = os.environ.get('NANO_OWNED_STRING_JOIN_DIR')
        context = nullcontext(retained) if retained else tempfile.TemporaryDirectory(prefix='nano-string-joins-')
        with context as name:
            tmp = Path(name)
            run = self.checked([ROOT/'obj/test_owned_string_joins', tmp])
            print(run.stdout, end='')
            self.assertEqual([line for line in run.stdout.splitlines() if line.startswith('case ')],
                             ['case 0 0 42', 'case 1 0 42'])
            cc = os.environ.get('CC', 'cc')
            identity = self.checked([cc, '--version']).stdout
            leaks = '0' if os.uname().sysname == 'Darwin' and 'Apple clang version' in identity else '1'
            for arm in range(2):
                with self.subTest(arm=arm):
                    module, generated = tmp/f'case{arm}.nvm', tmp/f'case{arm}.c'
                    self.checked([ROOT/'bin/nano_vm', '--verify-only', module])
                    vm = subprocess.run([ROOT/'bin/nano_vm', module], capture_output=True, timeout=30)
                    self.assertEqual(vm.returncode, 42, vm.stderr.decode(errors='replace'))
                    rebuilt = tmp/f'rebuilt{arm}.c'
                    self.checked([ROOT/'bin/nvm2c', module, '-o', rebuilt])
                    self.assertEqual(generated.read_bytes(), rebuilt.read_bytes())
                    harness = tmp/f'harness{arm}.c'
                    harness.write_text('#include <assert.h>\n#define NVM2C_NO_MAIN\n' +
                                       f'#include "{generated.name}"\n' +
                                       'int main(void){for(int i=0;i<20;i++){int64_t result=-91;'
                                       'assert(nvm_owned_entry(&result)==0);assert(result==42);}return 0;}\n')
                    binary = tmp/f'check{arm}'
                    self.checked([cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-g',
                                  harness, '-o', binary])
                    self.checked([binary], env={**os.environ, 'ASAN_OPTIONS': f'detect_leaks={leaks}:halt_on_error=1'})

if __name__ == '__main__':
    unittest.main()
