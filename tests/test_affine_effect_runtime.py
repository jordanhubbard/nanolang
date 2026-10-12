"""I execute affine effect transfers and qualify native failure cleanup."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CASES = {name: 0 for name in ("resume", "return", "nested", "call_resume",
                             "call_return", "contained", "capture", "recursive", "contained_return",
                             "nested_return", "void", "string", "owned_string", "float", "borrow", "owned_union", "recursive_borrow", "global_resume", "global_string",
                             "global_return", "global_recursive")}
CASES.update(assertion=2, unhandled=3, depth=3, global_assertion=2, global_unhandled=3)


class AffineEffectRuntime(unittest.TestCase):
    def run_checked(self, command, expected=0, **kwargs):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                timeout=120, **kwargs)
        self.assertEqual(result.returncode, expected, str(command) + "\n" + result.stdout + result.stderr)
        return result

    def test_execution_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="nano-affine-effects-") as directory:
            work = Path(directory)
            self.run_checked([ROOT / "obj/test_affine_bytecode", "--effects", work])
            compiler = os.environ.get("NANO_AFFINE_EFFECT_CC", str(Path("/opt/homebrew/opt/llvm/bin/clang")
                if Path("/opt/homebrew/opt/llvm/bin/clang").exists() else "cc"))
            for name, expected_status in CASES.items():
                with self.subTest(case=name):
                    artifact = work / f"{name}.nvm"
                    self.run_checked([ROOT / "bin/nano_vm", "--verify-only", artifact])
                    self.run_checked([ROOT / "bin/nano_vm", artifact], 1 if expected_status else 7)
                    vm = self.run_checked([os.environ.get("NANO_AFFINE_EFFECT_VM", str(ROOT / "obj/test_affine_effect_vm")), artifact, str(expected_status)])
                    print(f"{name}: {vm.stdout.strip()}")
                    translated = work / "translated.c"
                    self.run_checked([ROOT / "bin/nvm2c", artifact, "-o", translated])
                    self.assertEqual(translated.read_bytes(), (work / f"{name}.c").read_bytes())
                    harness = work / "harness.c"
                    harness.write_text('''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
static size_t live,attempt,fail_at;
static void *allocate(size_t count,size_t size){
 if(++attempt==fail_at)return NULL;
 void *value=calloc(count,size);if(value)live++;return value;
}
static void release(void *value){if(value){assert(live);live--;free(value);}}
#define NOWN_ALLOC allocate
#define NOWN_FREE release
#define NVM2C_NO_MAIN
#include "translated.c"
static size_t execute(size_t failure,int expected){
 int64_t result=1234;attempt=0;fail_at=failure;
 int status=nvm_owned_entry(&result);
 assert(live==0);
 if(failure && failure<=attempt){assert(status==1);assert(result==1234);}
 else {assert(status==expected);assert(result==(expected?1234:7));}
 return attempt;
}
int main(void){
''' + f" size_t attempts=execute(0,{expected_status});\n" + (
                        f" for(size_t failure=1;failure<=attempts;failure++)execute(failure,{expected_status});\n"
                        if name != "depth" else
                        " size_t failures[]={1,2,10,100,1023,1024,2047};\n"
                        " for(size_t i=0;i<sizeof(failures)/sizeof(*failures);i++)execute(failures[i],3);\n"
                    ) + f' execute(0,{expected_status});printf("{name}: %zu allocations, cleanup passed\\n",attempts);return 0;\n}}\n')
                    binary = work / "harness"
                    self.run_checked([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1",
                                      "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g",
                                      harness, "-o", binary])
                    print(self.run_checked([binary], env={**os.environ,
                        "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                        "UBSAN_OPTIONS": "halt_on_error=1"}).stdout.strip())


if __name__ == "__main__":
    unittest.main()
