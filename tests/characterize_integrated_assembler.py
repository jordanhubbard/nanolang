"""I compare integrated-Clang assembler text replay with native compilation.

This experiment does not change production admission. --require-identical
requires both runtime payload equality and byte-identical native objects.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


def run(argv, cwd):
    result = subprocess.run([str(arg) for arg in argv], cwd=cwd, capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"I could not run {argv!r}: {result.stderr.decode()}")
    return result


def measure(compiler):
    cases = []
    for suffix in ("s", "S"):
        for debug in (False, True):
            with tempfile.TemporaryDirectory(prefix="nano-integrated-text-") as tmp:
                root = Path(tmp)
                included = root / "included files"
                included.mkdir()
                payload = included / "payload.bin"
                payload.write_bytes(b"x42y")
                macro = included / "macro.s"
                macro.write_text('.macro emit file\n.incbin "\\file", 1, 2\n.endm\n'
                                 f'emit "{payload}"\nemit "{payload}"\n')
                nested = included / "nested.s"
                nested.write_text('.include "macro.s"\n.if 0\n.include "missing.s"\n.endif\n')
                symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                source = root / ("input." + suffix)
                assembly = f'.data\n.globl {symbol}\n{symbol}:\n.include "nested.s"\n.text\n'
                if suffix == "S":
                    assembly = '#define SECTION .data\n' + assembly.replace('.data', 'SECTION', 1)
                source.write_text(assembly)
                flags = ["-g"] if debug else []
                search = "-Wa,-I," + str(included)
                native, replay = root / "native.o", root / "replay.o"
                native_report = run([compiler, "-###", "-c", *flags, search, source, "-o", native], root)
                preprocessed = any('"-cc1"' in line and '"-E"' in line
                                   for line in native_report.stderr.decode().splitlines())
                run([compiler, "-c", *flags, search, source, "-o", native], root)
                retained = root / "retained.s"
                if preprocessed:
                    run([compiler, "-E", "-x", "assembler-with-cpp", source, "-o", retained], root)
                else:
                    retained.write_bytes(source.read_bytes())
                report = run([compiler, "-###", "-c", "-x", "assembler", *flags, search, retained, "-o", replay], root)
                jobs = [shlex.split(line.strip()) for line in report.stderr.decode().splitlines()
                        if line.lstrip().startswith('"')]
                if len(jobs) != 1 or len(jobs[0]) < 2 or jobs[0][1] != "-cc1as":
                    raise RuntimeError(f"I need exactly one selected integrated assembler job: {jobs!r}")
                job = jobs[0]
                if job.count("-filetype") != 1 or job.count("-o") != 1:
                    raise RuntimeError("I need unambiguous assembler output options")
                expanded = root / "expanded.s"
                job[job.index("-filetype") + 1] = "asm"
                job[job.index("-o") + 1] = str(expanded)
                run([*job, "-msave-temp-labels"], root)
                expanded_bytes = expanded.read_bytes()
                for path in (source, retained, macro, nested, payload):
                    path.unlink()
                # Debug emission already happened during text capture. I follow
                # production phase routing and do not request it a second time.
                replayed = subprocess.run([compiler, "-c", "-x", "assembler", str(expanded), "-o", str(replay)],
                                          cwd=root, capture_output=True, timeout=20)
                if replayed.returncode:
                    cases.append({"suffix": suffix, "debug": debug, "native_preprocessing": preprocessed,
                                  "selected_backend": job[0], "inputs_deleted": True,
                                  "replay_error": replayed.stderr.decode(), "object_identical": False,
                                  "expanded_bytes": len(expanded_bytes)})
                    continue
                # Fresh processes avoid the dynamic loader retaining earlier libraries.
                def read_payload(obj):
                    library = obj.with_suffix(".dylib" if sys.platform == "darwin" else ".so")
                    run([compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                         obj, "-o", library], root)
                    result = run([sys.executable, "-c",
                        "import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); "
                        "print(bytes((ctypes.c_ubyte*4).in_dll(l,'snapshot_payload')).hex())", library], root)
                    return result.stdout.decode().strip()
                native_payload, replay_payload = read_payload(native), read_payload(replay)
                cases.append({"suffix": suffix, "debug": debug,
                              "native_preprocessing": preprocessed,
                              "selected_backend": job[0], "inputs_deleted": True,
                              "native_payload": native_payload, "replay_payload": replay_payload,
                              "object_identical": native.read_bytes() == replay.read_bytes(),
                              "expanded_bytes": len(expanded_bytes),
                              "remaining_file_reads": any(word in expanded_bytes for word in (b".incbin", b".include"))})
    return {"compiler": compiler, "version": run([compiler, "--version"], Path.cwd()).stdout.decode().splitlines()[0],
            "platform": sys.platform, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", nargs="?", default="clang")
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if not compiler:
        raise SystemExit("I need a Clang executable")
    result = measure(compiler)
    print(json.dumps(result, indent=2))
    if args.require_identical and any(not case["object_identical"] or case["remaining_file_reads"] or
                                     case["native_payload"] != "34323432" or case["replay_payload"] != "34323432"
                                     for case in result["cases"]):
        raise SystemExit("I did not reproduce the native object and payload from retained text")
