"""I build AOT sanitizer tests in fresh, privately owned output directories."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FLAGS = "-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"


def run_sanitizers(make, cc):
    with tempfile.TemporaryDirectory(prefix="nano-aot-sanitizers-") as directory:
        work = Path(directory)
        env = os.environ.copy()
        # Existing generated helpers retain allocations until process exit.
        # I test memory access and undefined behavior, not leak freedom.
        env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
        command = [*shlex.split(make), "-j1", f"CC={cc} {FLAGS}",
                   f"OBJ_DIR={work / 'obj'}", f"BIN_DIR={work / 'bin'}",
                   f"FILE_PUBLIC_LIBRARY={work / 'lib/libnano_file_runtime.a'}",
                   f"NVM2C_TEST_BINARY={work / 'bin/test_nvm2c'}", "test-nvm2c"]
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode:
            return result.returncode
        # Verify actual object instrumentation, not just the requested flags.
        for name in ("nvm2c", "nvm2c_shape"):
            artifact = work / f"obj/nanoisa/{name}.o"
            symbols = subprocess.check_output(["nm", "-u", str(artifact)], text=True)
            if "__asan_" not in symbols or "__ubsan_" not in symbols:
                print(f"I did not find ASan and UBSan instrumentation in {name}.o", flush=True)
                return 1
        print("I verified fresh ASan/UBSan translator and shape objects.", flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make", default="make")
    parser.add_argument("--cc", default="cc")
    args = parser.parse_args()
    return run_sanitizers(args.make, args.cc)


if __name__ == "__main__":
    raise SystemExit(main())
