"""I check native array declarations against the function's defining image."""
import os
import contextlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArrayAbiLoader(unittest.TestCase):
    def test_declarations(self):
        cc = shlex.split(os.environ.get("NANO_NATIVE_TEST_CC") or os.environ.get("CC", "cc"))
        crypto = shlex.split(subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "libcrypto"], text=True, timeout=30))
        retained = os.environ.get("NANO_ARRAY_ABI_REPORT_DIR")
        context = (contextlib.nullcontext(tempfile.mkdtemp(prefix="nano-array-abi-", dir=retained))
                   if retained else tempfile.TemporaryDirectory(prefix="nano-array-abi-"))
        with context as tmp:
            if retained:
                print("I retain native array ABI artifacts at", tmp, flush=True)
            directory = Path(tmp)

            def build(name, source, flags):
                path = directory / (name + ".c")
                path.write_text(source)
                output = directory / name
                result = subprocess.run([*cc, "-std=c11", "-g", "-Isrc", str(path),
                                         *flags, "-o", str(output)], cwd=ROOT,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                return output

            shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            dependency = build("dependency.dylib", """
                #include "runtime/dyn_array.h"
                const uint32_t borrowed__nano_array_abi = NANO_DYN_ARRAY_ABI_VERSION;
                int dependency(void) { return 7; }
                """, shared)
            library = build("artifact.dylib", """
                #include "runtime/dyn_array.h"
                #include <stdlib.h>
                int matching(void) { return 1; }
                NANO_EXPORT_ARRAY_ABI(matching);
                int mismatch(void) { return 2; }
                const uint32_t mismatch__nano_array_abi = 99;
                int legacy(void) { return 3; }
                int stale(void) { abort(); }
                const uint32_t stale__nano_array_abi = 1;
                extern int dependency(void);
                int borrowed(void) { return dependency(); }
                """, [*shared, str(dependency)])
            probe = build("probe", """
                #include "runtime/ffi_loader.h"
                #include "runtime/native_array_abi.h"
                #include "runtime/dyn_array.h"
                #include <assert.h>
                #include <string.h>
                #include <sys/wait.h>
                #include <unistd.h>
                static int local_provider(void) { return 42; }
                static int other_provider(void) { return 43; }
                NANO_DECLARE_LOCAL_ARRAY_ABI(local_provider);
                static const NanoLocalArrayAbi wrong_version = {(void *)local_provider, 99};
                static const NanoLocalArrayAbi wrong_function = {(void *)other_provider, NANO_DYN_ARRAY_ABI_VERSION};
                int main(int argc, char **argv) {
                    assert(argc == 2);
                    nano_require_local_array_abi((void *)local_provider, local_provider__nano_local_array_abi,
                                                 "local_provider__nano_array_abi", 2, "local_provider");
                    assert(local_provider() == 42);
                    const NanoLocalArrayAbi *invalid[] = {&wrong_version, &wrong_function};
                    for (int i = 0; i < 2; ++i) {
                        pid_t child = fork(); assert(child >= 0);
                        if (!child) {
                            nano_require_local_array_abi((void *)local_provider, invalid[i],
                                "local_provider__nano_array_abi", 2, "local_provider");
                            _exit(0);
                        }
                        int status; assert(waitpid(child, &status, 0) == child);
                        assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
                    }
                    assert(ffi_loader_init(false));
                    assert(ffi_loader_open("fixture", argv[1]));
                    const char *names[] = {"matching", "mismatch", "legacy", "borrowed", "stale"};
                    char error[512];
                    assert(NANO_DYN_ARRAY_ABI_VERSION == 2);
                    for (int i = 0; i < 5; ++i) {
                        void *fn = ffi_loader_resolve_module(names[i], "fixture");
                        assert(fn);
                        bool accepted = i == 0;
                        assert(ffi_loader_check_array_abi("fixture", names[i], fn, NANO_DYN_ARRAY_ABI_VERSION,
                                                         error, sizeof error) == accepted);
                        assert(accepted ? !error[0] : strstr(error, "native array ABI") != NULL);
                        assert(ffi_loader_check_array_abi(NULL, names[i], fn, NANO_DYN_ARRAY_ABI_VERSION,
                                                         error, sizeof error) == accepted);
                        assert(!ffi_loader_check_array_abi("absent", names[i], fn, NANO_DYN_ARRAY_ABI_VERSION,
                                                          error, sizeof error));
                        for (uint32_t expected = 1; expected <= 2; ++expected) {
                            /* I preserve the utility's explicit legacy-host contract too. */
                            bool compatible = expected == 2 ? i == 0 : (i == 2 || i == 4);
                            assert(ffi_loader_check_array_abi("fixture", names[i], fn, expected,
                                                             error, sizeof error) == compatible);
                            pid_t child = fork();
                            assert(child >= 0);
                            if (!child) {
                                char marker[128];
                                snprintf(marker, sizeof marker, "%s__nano_array_abi", names[i]);
                                nano_require_native_array_abi(fn, marker, expected, names[i]);
                                _exit(0);
                            }
                            int status;
                            assert(waitpid(child, &status, 0) == child);
                            if (compatible)
                                assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
                            else
                                assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
                        }
                    }
                    ffi_loader_shutdown();
                    return 0;
                }
                """, ["-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE", "src/runtime/ffi_loader.c", "src/runtime/module_build_dir.c", "-pthread", *crypto,
                       *(["-ldl"] if sys.platform.startswith("linux") else [])])
            result = subprocess.run([str(probe), str(library)], capture_output=True,
                                    text=True, timeout=15)
            if retained:
                (directory/'stdout.txt').write_text(result.stdout)
                (directory/'stderr.txt').write_text(result.stderr)
                (directory/'status.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
            self.assertEqual(result.returncode, 0, result.stderr)


    def test_generated_walkdir_provider(self):
        with tempfile.TemporaryDirectory(prefix="nano-walk-abi-") as tmp:
            directory = Path(tmp)
            data = directory / "data"
            data.mkdir()
            (data / "item").write_text("one")
            source = directory / "walk.nano"
            source.write_text(
                'from "modules/std/fs.nano" import walkdir\n'
                'fn main() -> int {\n'
                f' let files: array<string> = (walkdir {json.dumps(str(data))})\n'
                ' assert (== (array_length files) 1)\n'
                f' assert (== (at files 0) {json.dumps(str(data / "item"))})\n'
                ' return 0\n}\nshadow main { assert (== (main) 0) }\n')
            compiler = ROOT / "bin/nanoc_c"
            output = directory / "walk"
            built = subprocess.run([str(compiler), str(source), "-o", str(output)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ran = subprocess.run([str(output)], cwd=ROOT, capture_output=True,
                                 text=True, timeout=30)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            for result in (built, ran):
                for marker in ("AddressSanitizer", "LeakSanitizer", "runtime error:"):
                    self.assertNotIn(marker, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
