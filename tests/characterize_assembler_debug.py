"""I compare native and production standalone-assembler debug evidence.

--require-debug fails if production drops native debug sections or source
provenance. Matching these observations is not full DWARF equivalence.
--candidate instead tests debug flags, retained original basenames and explicit
assembler directory remapping, requiring byte-identical native objects too.
"""

import argparse
import difflib
import hashlib
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


def evidence(obj, source, cwd, text=None):
    listing = run(["objdump", "-h", obj], cwd)
    sections = [words[1] for line in listing.splitlines()
                if len(words := line.split()) > 1 and words[0].isdigit()
                and words[1].startswith((".debug_", "__debug_"))]
    command = ["dwarfdump", "--debug-info", "--debug-line"] if sys.platform == "darwin" else [
        "readelf", "--debug-dump=info", "--debug-dump=decodedline"]
    decoded = run([*command, obj], cwd)
    normalized = decoded.replace(str(obj), "@object")
    if text is not None: text.extend(normalized.splitlines(keepends=True))
    return {"sections": sections, "source_named": source.name in decoded,
            "decoded_debug_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "private_snapshot_named": "__snapshot_" in decoded or "__assembler_" in decoded,
            "compile_units": decoded.count("DW_TAG_compile_unit")}


def measure(compiler, candidate=False, flat=False, debug_options=("-g",), macro_read=False):
    if flat and not candidate:
        raise ValueError("I require candidate mode for the flat-path experiment")
    version = run([compiler, "--version"], ROOT).splitlines()[0]
    flags = list(debug_options) + (["-fno-integrated-as"] if "clang" in version else [])
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
                if macro_read:
                    payload = directory / "debug payload.bin"
                    payload.write_bytes(b"*")
                    assembly = '.macro read_payload path\n.incbin "\\path"\n.endm\n' + assembly.replace(
                        '.byte 42', 'read_payload "' + str(payload) + '"')
                if suffix == "S": assembly = '#define INSTRUCTION nop\n' + assembly.replace('nop', 'INSTRUCTION')
                source.write_text(assembly)
                (module / "answer.c").write_text('extern unsigned char snapshot_payload[];\n'
                    'long long nano_build_answer(void) { return snapshot_payload[0]; }\n')
                (module / "module.json").write_text(json.dumps({"name": "answer_native",
                    "c_sources": ["answer.c", source.name], "cflags": flags}))
                native = directory / "native.o"
                run([compiler, "-c", "-fPIC", *flags, source, "-o", native], directory)
                native_text = []
                expected = evidence(native, source, directory, native_text)
                if debug_options and debug_options[-1] != "-g0" and (not expected["sections"] or not expected["source_named"]):
                    raise RuntimeError("I need a native object with debug sections and source provenance")
                probe = ROOT / "obj/test_module_generation_probe"
                run([probe, "build", module], directory, env)
                generation = Path(run([probe, "directory", module], directory, env).strip())
                production_object = generation / "answer_native_1.o"
                production_text = []
                observed = evidence(production_object, source, directory, production_text)
                candidate_evidence = None
                candidate_diff = None
                candidate_identical = None
                if candidate:
                    retained = generation / "__snapshot_0_1.s"
                    private = directory / "retained unit"
                    replay_source = private / source.name
                    if not flat:
                        private.mkdir()
                        replay_source.write_bytes(retained.read_bytes())
                    option = "-Wa,-fdebug-prefix-map=" if "clang" in version and sys.platform == "darwin" else "-Wa,--debug-prefix-map="
                    remaps = [option + str(private) + "=" + str(source.parent)]
                    if flat:
                        replay_source = retained
                        remaps = [option + str(retained.parent) + "=" + str(source.parent),
                                  option + retained.name + "=" + source.name]
                    if source.parent.resolve() != source.parent:
                        remaps.append(option + str(source.parent.resolve()) + "=" + str(source.parent))
                    output = directory / "candidate.o"
                    original = source.read_bytes()
                    source.unlink()
                    try:
                        run([compiler, "-c", "-fPIC", *flags, *remaps, "-x", "assembler", replay_source, "-o", output], directory)
                    finally:
                        source.write_bytes(original)
                    candidate_text = []
                    candidate_evidence = evidence(output, source, directory, candidate_text)
                    candidate_identical = native.read_bytes() == output.read_bytes()
                    candidate_diff = ''.join(difflib.unified_diff(native_text, candidate_text, fromfile="native", tofile="candidate"))
                library = run([probe, "library", module], directory, env).strip()
                value = run([sys.executable, "-c", "import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); "
                    "l.nano_build_answer.restype=ctypes.c_int64; print(l.nano_build_answer())", library], directory)
                run([probe, "build", module], directory, env)
                warm_generation = Path(run([probe, "directory", module], directory, env).strip())
                reused = generation == warm_generation
                warm_text = []
                evidence(warm_generation / "answer_native_1.o", source, directory, warm_text)
                cases.append({"suffix": suffix, "cache": "shared" if shared else "local",
                              "native": expected, "production": observed,
                              "production_object_identical": native.read_bytes() == production_object.read_bytes(),
                              "production_debug_diff": ''.join(difflib.unified_diff(native_text, production_text,
                                                                                     fromfile="native", tofile="production")),
                              "warm_debug_diff": ''.join(difflib.unified_diff(production_text, warm_text,
                                                                               fromfile="cold", tofile="warm")),
                              "candidate": candidate_evidence,
                              "candidate_debug_diff": candidate_diff,
                              "candidate_object_identical": candidate_identical,
                              "candidate_source_deleted": candidate,
                              "answer": int(value), "generation_reused": reused})
    return {"compiler": compiler, "version": version, "platform": sys.platform,
            "candidate_mode": ("flat" if flat else "original-basename") if candidate else None,
            "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", nargs="?", default="cc")
    parser.add_argument("--require-debug", action="store_true")
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--flat", action="store_true",
                        help="I test directory and basename remaps without copying the retained unit")
    args = parser.parse_args()
    if args.flat and not args.candidate:
        parser.error("I require --candidate with --flat")
    compiler = shutil.which(args.compiler)
    if not compiler: raise SystemExit("I need a C compiler")
    result = measure(compiler, candidate=args.candidate, flat=args.flat)
    print(json.dumps(result, indent=2))
    if args.require_debug and any(case["native"] != case["candidate" if args.candidate else "production"] or case["answer"] != 42
                                  or not case["generation_reused"] or (args.candidate and not case["candidate_object_identical"])
                                  for case in result["cases"]):
        raise SystemExit("I did not retain the native debug evidence and reusable result")
