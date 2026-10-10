"""I require the existing VM/native result, lifetime and allocation assertions."""
from tests.test_owned_runtime import OwnedRuntime as _OwnedRuntime
from tests.test_owned_runtime import ROOT
from pathlib import Path
import os
import subprocess
import tempfile

class OwnedGlobalRuntime(_OwnedRuntime):
    executable = "test_owned_global_runtime"
    case_count = 8
    max_live_records = 2
    refusal_count = 5
    noninteger_cases = set()

    def test_native_trap_releases_globals(self):
        with tempfile.TemporaryDirectory(prefix="nano-global-trap-") as name:
            tmp = Path(name)
            run = subprocess.run([ROOT / "obj" / self.executable, tmp], capture_output=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            harness = tmp / "trap_check.c"
            harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
static size_t live,attempt,fail_at;
static void *allocate(size_t n,size_t s){if(++attempt==fail_at)return NULL;void *p=calloc(n,s);if(p)live++;return p;}
static void release(void *p){assert(live);live--;free(p);}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
#include "trapped.c"
int main(void){
 for(size_t failure=1;;failure++){
  attempt=0;fail_at=failure;int64_t result=-99;
  assert(nvm_owned_entry(&result)!=0);assert(result==-99);assert(!live);
  if(attempt<failure)break;
  assert(failure<32);
 }
 for(unsigned repeat=0;repeat<3;repeat++){
  fail_at=0;int64_t result=-99;assert(nvm_owned_entry(&result)!=0);assert(!live);assert(result==-99);
 }
 return 0;
}
''')
            binary = tmp / "trap_check"
            compiled = subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                                       "-fsanitize=address,undefined", "-fno-omit-frame-pointer", str(harness),
                                       "-lm", "-o", str(binary)], capture_output=True, timeout=90)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1", UBSAN_OPTIONS="halt_on_error=1")
            ran = subprocess.run([binary], capture_output=True, timeout=30, env=env)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

del _OwnedRuntime
