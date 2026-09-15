"""I check native array declarations against the function's defining image."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArrayAbiLoader(unittest.TestCase):
    def test_declarations(self):
        cc = shlex.split(os.environ.get("CC", "cc"))
        with tempfile.TemporaryDirectory(prefix="nano-array-abi-") as tmp:
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
                #include <stdint.h>
                const uint32_t borrowed__nano_array_abi = 1;
                int dependency(void) { return 7; }
                """, shared)
            library = build("artifact.dylib", """
                #include "runtime/dyn_array.h"
                int matching(void) { return 1; }
                NANO_EXPORT_ARRAY_ABI(matching);
                int mismatch(void) { return 2; }
                const uint32_t mismatch__nano_array_abi = 99;
                int legacy(void) { return 3; }
                extern int dependency(void);
                int borrowed(void) { return dependency(); }
                """, [*shared, str(dependency)])
            probe = build("probe", """
                #include "runtime/ffi_loader.h"
                #include "runtime/native_array_abi.h"
                #include <assert.h>
                #include <string.h>
                #include <sys/wait.h>
                #include <unistd.h>
                /* I do not exercise artifact-directory discovery in this probe. */
                bool nano_module_artifact_dir(const char *p, char *d, size_t n) {
                    (void)p; (void)d; (void)n; return false;
                }
                int main(int argc, char **argv) {
                    assert(argc == 2);
                    assert(ffi_loader_init(false));
                    assert(ffi_loader_open("fixture", argv[1]));
                    const char *names[] = {"matching", "mismatch", "legacy", "borrowed"};
                    const bool accepted[] = {true, false, true, false};
                    char error[512];
                    for (int i = 0; i < 4; ++i) {
                        void *fn = ffi_loader_resolve_module(names[i], "fixture");
                        assert(fn);
                        assert(ffi_loader_check_array_abi("fixture", names[i], fn, 1,
                                                         error, sizeof error) == accepted[i]);
                        assert(accepted[i] ? !error[0] : strstr(error, "native array ABI") != NULL);
                        assert(!ffi_loader_check_array_abi("fixture", names[i], fn, 2,
                                                          error, sizeof error));
                        assert(!ffi_loader_check_array_abi("absent", names[i], fn, 1,
                                                          error, sizeof error));
                        for (uint32_t expected = 1; expected <= 2; ++expected) {
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
                            if (expected == 1 && accepted[i])
                                assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
                            else
                                assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
                        }
                    }
                    ffi_loader_shutdown();
                    return 0;
                }
                """, ["-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE", "src/runtime/ffi_loader.c", "-pthread",
                       *(["-ldl"] if sys.platform.startswith("linux") else [])])
            result = subprocess.run([str(probe), str(library)], capture_output=True,
                                    text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
