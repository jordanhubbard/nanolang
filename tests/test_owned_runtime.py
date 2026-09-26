"""I execute real owned transfers with identical VM/native results and cleanup."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]


class OwnedRuntime(unittest.TestCase):
    executable = "test_owned_runtime"
    case_count = 12
    max_live_records = 3
    case_live_records = {}
    refusal_count = 7
    noninteger_cases = {5, 6}
    assertion_cases = set()
    compiler = staticmethod(lambda: os.environ.get("CC", "cc"))
    def test_paired_execution_lifetimes_and_refusals(self):
        with tempfile.TemporaryDirectory(prefix="nano-owned-runtime-") as name:
            tmp = Path(name)
            run = subprocess.run([os.environ.get("NANO_OWNED_RUNTIME_TEST", str(ROOT / "obj" / self.executable)), tmp],
                                 capture_output=True, text=True, timeout=90, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            print(run.stdout, end="")
            cases = [line.split() for line in run.stdout.splitlines() if line.startswith("case ")]
            self.assertEqual(len(cases), self.case_count)
            for _, number, expected in cases:
                with self.subTest(case=number):
                    artifact = tmp / f"case{number}.nvm"
                    verified = subprocess.run([ROOT / "bin/nano_vm", "--verify-only", artifact], capture_output=True, timeout=30)
                    self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
                    vm_result = subprocess.run([ROOT / "bin/nano_vm", artifact], capture_output=True, timeout=30)
                    # The CLI uses only int results as process exit codes; the C
                    # fixture checks exact bool/u8 tags and values separately.
                    exit_code = 0 if int(number) in self.noninteger_cases else int(expected) & 255
                    if int(number) in self.assertion_cases:
                        exit_code = 1
                    self.assertEqual(vm_result.returncode, exit_code, vm_result.stdout + vm_result.stderr)
                    translated = tmp / f"roundtrip{number}.c"
                    rebuilt = subprocess.run([ROOT / "bin/nvm2c", artifact, "-o", translated], capture_output=True, timeout=30)
                    self.assertEqual(rebuilt.returncode, 0, rebuilt.stdout + rebuilt.stderr)
                    generated = tmp / f"case{number}.c"
                    self.assertEqual(translated.read_bytes(), generated.read_bytes())
                    harness = tmp / f"check{number}.c"
                    harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
static size_t live,peak,attempt,fail_at;
static void *allocate(size_t n,size_t s){if(++attempt==fail_at)return NULL;void *p=calloc(n,s);if(p){live++;if(live>peak)peak=live;}return p;}
static void release(void *p){assert(live);live--;free(p);}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
''' + f'#include "{generated.name}"\n' + '''int main(void){
 for(size_t failure=1;;failure++){
  int64_t result=0;attempt=peak=0;fail_at=failure;
  int status=nvm_owned_entry(&result);
  assert(live==0);assert(peak<=MAX_LIVE_RECORDS);
  if(status==EXPECTED_STATUS){if(!status)assert(result==(int64_t)UINT64_C(EXPECTED));break;}
  assert(status==1);
  assert(failure<1100);
 }
 puts("owned cleanup passed");return 0;
}
'''.replace("EXPECTED_STATUS", "2" if int(number) in self.assertion_cases else "0").replace("EXPECTED", str(int(expected) & ((1 << 64) - 1))).replace("MAX_LIVE_RECORDS", str(self.case_live_records.get(int(number), self.max_live_records))))
                    binary = tmp / f"check{number}"
                    compiled = subprocess.run([self.compiler(), "-std=c11", "-Wall", "-Wextra", "-Werror",
                                               "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g",
                                               harness, "-o", binary], capture_output=True, text=True, timeout=60)
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    executed = subprocess.run([binary], capture_output=True, text=True, timeout=90,
                                              env={**os.environ, "ASAN_OPTIONS": asan_options("halt_on_error=1")})
                    self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
                    self.assertIn("owned cleanup passed", executed.stdout)
            self.assertEqual(len(list(tmp.glob("refused*.nvm"))), self.refusal_count)
            for artifact in sorted(tmp.glob("refused*.nvm")):
                with self.subTest(refusal=artifact.name):
                    previous = tmp / "previous.c"
                    previous.write_text("previous output\n")
                    for command in ([ROOT / "bin/nano_vm", artifact],
                                    [ROOT / "bin/nvm2c", artifact, "-o", previous]):
                        refusal = subprocess.run(command, capture_output=True, timeout=30)
                        self.assertNotEqual(refusal.returncode, 0, refusal.stdout + refusal.stderr)
                        self.assertRegex((refusal.stdout + refusal.stderr).decode(), r"ownership|owned|reference")
                    self.assertEqual(previous.read_text(), "previous output\n")


if __name__ == "__main__":
    unittest.main()
