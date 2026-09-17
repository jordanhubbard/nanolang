"""I compare long linker groups with direct driver invocations.

Run python3 -m tests.characterize_link_argument_transport [compiler].
--require-consistent rejects failed builds, changed answers, lost reuse,
unusable returned flags and failed-replacement publication.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from tests import test_bytecode_shadows as shadows


def measure(compiler):
    probe = shadows.ROOT / "obj/test_module_generation_probe"
    if not probe.is_file():
        raise RuntimeError("I need make obj/test_module_generation_probe")
    cases = []
    platform = "ldflags_macos" if sys.platform == "darwin" else "ldflags_linux"
    for origin in ("ldflags", platform, "pkg_config", "system_libs"):
        with tempfile.TemporaryDirectory(prefix="nano-link-arguments-") as tmp:
            directory = Path(tmp)
            module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
            env["CC"] = compiler

            def run(arguments, **kwargs):
                return subprocess.run(arguments, cwd=directory, env=env,
                                      capture_output=True, timeout=30, **kwargs)

            def invoke(mode):
                return run([str(probe), mode, str(module)])

            def answer(path):
                result = run([sys.executable, "-c", "import ctypes,sys; "
                              "lib=ctypes.CDLL(sys.argv[1]); "
                              "lib.nano_build_answer.restype=ctypes.c_int64; "
                              "print(lib.nano_build_answer())", str(path)])
                return int(result.stdout) if result.returncode == 0 else None

            metadata = {"name": "answer_native", "c_sources": ["answer.c"]}
            config = module / "module.json"
            config.write_text(json.dumps(metadata))
            initial = invoke("build")
            if initial.returncode:
                raise RuntimeError(initial.stderr.decode())
            previous = invoke("directory").stdout.strip()
            values = ["-L/nano/missing"] * 1200 + ["-lm", "-lc", "-lm"]
            if origin == "pkg_config":
                metadata[origin] = ["link-fixture"]
                pkg = directory / "pkg-config"
                pkg.write_text(f"#!{sys.executable}\nimport sys\n"
                               f"if '--libs' in sys.argv: print({' '.join(values)!r})\n")
                pkg.chmod(0o700)
                env["PKG_CONFIG"] = str(pkg)
            elif origin == "system_libs":
                metadata[origin] = ["m", "c"] * 1200 + ["m"]
                values = ["-l" + value for value in metadata[origin]]
            else:
                metadata[origin] = values
            config.write_text(json.dumps(metadata))
            original = config.read_bytes()
            direct_library = directory / ("direct.dylib" if sys.platform == "darwin" else "direct.so")
            prefix = [compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC"]
            direct = run(prefix + [str(module / "answer.c"), "-o", str(direct_library)] + values)
            built = invoke("build")
            generation = invoke("directory").stdout.strip()
            cold = answer(invoke("library").stdout.decode().strip()) if not built.returncode else None
            warm = invoke("build") if not built.returncode else None
            warm_answer = answer(invoke("library").stdout.decode().strip()) if warm and not warm.returncode else None
            reused = bool(warm and not warm.returncode and invoke("directory").stdout.strip() == generation)
            unchanged = config.read_bytes() == original
            # A failed replacement must leave the previously published generation intact.
            metadata.setdefault("ldflags", []).append("-Wl,-nano-invalid-option")
            config.write_text(json.dumps(metadata))
            rejected = invoke("build")
            preserved = rejected.returncode != 0 and invoke("directory").stdout.strip() == generation
            metadata["ldflags"].pop()
            metadata["c_sources"] = []
            config.write_text(json.dumps(metadata))
            returned = invoke("build-info")
            flags = " ".join(line[len("link:"):] for line in returned.stdout.decode().splitlines()
                             if line.startswith("link:"))
            decoded = []
            for word in shlex.split(flags):
                decoded.extend(shlex.split(Path(word[1:]).read_text()) if word.startswith("@") else [word])
            later_library = directory / "later.so"
            later = run(shlex.join(prefix + [str(module / "answer.c"), "-o", str(later_library)]) + " " + flags,
                        shell=True) if not returned.returncode else None
            cases.append({"origin": origin, "argument_bytes": len(" ".join(values)),
                          "direct_status": direct.returncode,
                          "direct_answer": answer(direct_library) if not direct.returncode else None,
                          "build_status": built.returncode, "build_error": built.stderr.decode()[-500:],
                          "cold_answer": cold, "warm_answer": warm_answer, "generation_reused": reused,
                          "initial_generation_preserved": not built.returncode or generation == previous,
                          "failed_replacement_preserved": preserved,
                          "metadata_unchanged": unchanged, "returned_arguments_equal": decoded == values,
                          "returned_status": returned.returncode,
                          "later_status": later.returncode if later else None,
                          "later_answer": answer(later_library) if later and not later.returncode else None,
                          "metadata_fixture_bytes": len(original)})
    return {"platform": sys.platform, "compiler": compiler, "cases": cases}


def require_consistent(result):
    if not result["cases"]:
        raise SystemExit("I need measured linker cases before accepting transport.")
    for case in result["cases"]:
        if (any(case[key] != 0 for key in ("direct_status", "build_status", "returned_status", "later_status")) or
                any(case[key] != 42 for key in ("direct_answer", "cold_answer", "warm_answer", "later_answer")) or
                not all(case[key] for key in ("generation_reused", "initial_generation_preserved",
                                               "failed_replacement_preserved", "metadata_unchanged",
                                               "returned_arguments_equal"))):
            raise SystemExit("I did not preserve complete linker arguments and reusable output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", nargs="?", default="cc")
    parser.add_argument("--require-consistent", action="store_true")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if not compiler:
        raise SystemExit("I need a C compiler executable")
    result = measure(compiler)
    print(json.dumps(result, indent=2))
    if args.require_consistent:
        require_consistent(result)
