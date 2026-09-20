"""I execute one exact heterogeneous scalar union through VM and native C."""
from pathlib import Path
import os
import platform
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AffineScalarUnionRuntime(unittest.TestCase):
    def checked(self, command, *, env=None, timeout=90):
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=timeout)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def compiler(self):
        selected = os.environ.get("NANO_AFFINE_UNION_CC")
        if selected:
            return selected
        homebrew = Path("/opt/homebrew/opt/llvm/bin/clang")
        if platform.system() == "Darwin" and homebrew.is_file():
            return str(homebrew)
        return os.environ.get("CC", "cc")

    def test_vm_native_and_allocation_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="nano-affine-union-") as name:
            work = Path(name)
            generated = work / "union.c"
            artifact = work / "union.nvm"
            self.checked([ROOT / "obj/test_affine_bytecode", generated, artifact])
            self.checked([ROOT / "bin/nano_vm", "--verify-only", artifact])
            self.checked([ROOT / "bin/nano_vm", artifact])

            translated = work / "translated.c"
            self.checked([ROOT / "bin/nvm2c", artifact, "-o", translated])
            self.assertEqual(generated.read_bytes(), translated.read_bytes())

            harness = work / "harness.c"
            harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
static size_t live,peak,attempt,fail_at;
static void *allocate(size_t count,size_t size){
 if(++attempt==fail_at)return NULL;
 void *value=calloc(count,size);if(value){live++;if(live>peak)peak=live;}return value;
}
static void release(void *value){if(value){assert(live);live--;free(value);}}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
#include "union.c"
int main(void){
 for(size_t failure=1;;failure++){
  int64_t result=0;attempt=peak=0;fail_at=failure;
  int status=nvm_owned_entry(&result);
  assert(live==0);assert(peak<=2);
  if(!status){assert(result==1);break;}
  assert(failure<16);
 }
 puts("exact scalar union cleanup passed");return 0;
}
''')
            compiler = self.compiler()
            binary = work / "harness"
            self.checked([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                          "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g",
                          harness, "-o", binary])
            executed = self.checked([binary], env={**os.environ,
                "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                "UBSAN_OPTIONS": "halt_on_error=1"})
            self.assertIn("exact scalar union cleanup passed", executed.stdout)
            print(f"affine scalar union sanitizer compiler={compiler}")


if __name__ == "__main__":
    unittest.main()
