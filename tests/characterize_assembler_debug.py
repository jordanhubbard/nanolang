"""I compare native and production standalone-assembler debug evidence.

--require-debug fails if production drops native debug sections or source
provenance. Matching these observations is not full DWARF equivalence.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from tests.test_bytecode_shadows import BytecodeShadows, ROOT


def run(argv, cwd, env=None):
    result = subprocess.run([str(arg) for arg in argv], cwd=cwd, env=env,
                            capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"I could not run {argv!r}: {result.stderr.decode()}")
    return result.stdout.decode()


def evidence(obj, source, cwd):
    listing = run(["objdump", "-h", obj], cwd)
    sections = [words[1] for line in listing.splitlines()
                if len(words := line.split()) > 1 and words[0].isdigit()
                and words[1].startswith((".debug_", "__debug_"))]
    command = ["dwarfdump", "--debug-info", "--debug-line"] if sys.platform == "darwin" else [
        "readelf", "--debug-dump=info", "--debug-dump=decodedline"]
    decoded = run([*command, obj], cwd)
    return {"sections": sections, "source_named": source.name in decoded,
            "compile_units": decoded.count("DW_TAG_compile_unit")}


def measure(compiler):
    version = run([compiler, "--version"], ROOT).splitlines()[0]
    flags = ["-g"] + (["-fno-integrated-as"] if "clang" in version else [])
    cases = []
    for suffix in ("s", "S"):
        for shared in (False, True):
            with tempfile.TemporaryDirectory(prefix="nano-assembler-debug-") as tmp:
                directory = Path(tmp)
                module, _, env = BytecodeShadows().foreign_build_fixture(directory)
                env["NANO_CC"] = compiler
                env["NANO_AS_CAPTURE_HELPER"] = str(ROOT / "bin/nano_as_capture.so")
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "cache")
                source = module / ("payload." + suffix)
                symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                assembly = f'.text\nnop\n.data\n.globl {symbol}\n{symbol}:\n.byte 42\n'
                if suffix == "S": assembly = '#define INSTRUCTION nop\n' + assembly.replace('nop', 'INSTRUCTION')
                source.write_text(assembly)
                (module / "answer.c").write_text('extern unsigned char snapshot_payload[];\n'
                    'long long nano_build_answer(void) { return snapshot_payload[0]; }\n')
                (module / "module.json").write_text(json.dumps({"name": "answer_native",
                    "c_sources": ["answer.c", source.name], "cflags": flags}))
                native = directory / "native.o"
                run([compiler, "-c", "-fPIC", *flags, source, "-o", native], directory)
                expected = evidence(native, source, directory)
                if not expected["sections"] or not expected["source_named"]:
                    raise RuntimeError("I need a native object with debug sections and source provenance")
                probe = ROOT / "obj/test_module_generation_probe"
                run([probe, "build", module], directory, env)
                generation = Path(run([probe, "directory", module], directory, env).strip())
                observed = evidence(generation / "answer_native_1.o", source, directory)
                library = run([probe, "library", module], directory, env).strip()
                value = run([sys.executable, "-c", "import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); "
                    "l.nano_build_answer.restype=ctypes.c_int64; print(l.nano_build_answer())", library], directory)
                run([probe, "build", module], directory, env)
                reused = generation == Path(run([probe, "directory", module], directory, env).strip())
                cases.append({"suffix": suffix, "cache": "shared" if shared else "local",
                              "native": expected, "production": observed,
                              "answer": int(value), "generation_reused": reused})
    return {"compiler": compiler, "version": version, "platform": sys.platform, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", nargs="?", default="cc")
    parser.add_argument("--require-debug", action="store_true")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if not compiler: raise SystemExit("I need a C compiler")
    result = measure(compiler)
    print(json.dumps(result, indent=2))
    if args.require_debug and any(case["native"] != case["production"] or case["answer"] != 42
                                  or not case["generation_reused"] for case in result["cases"]):
        raise SystemExit("I did not retain the native debug evidence and reusable result")
