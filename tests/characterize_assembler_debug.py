"""I compare native and production standalone-assembler debug evidence.

--require-debug retains the original lexical-source comparison; its source
argument can differ from production's physical module root. Matching debug
observations alone is not full DWARF equivalence.
--candidate instead tests debug flags, retained original basenames and explicit
assembler directory remapping, requiring byte-identical native objects too.
--require-physical-debug compares production with my physical-module-root
source policy, retaining the lexical control as a separate observation.
--capture-object tests a native object from retained pre-expansion text; it
does not change production capture timing or dependency observation.
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


def measure(compiler, candidate=False, flat=False, debug_options=("-g",), macro_read=False, module_alias=False,
            capture_object=False, nested_read=False, integrated=False):
    if nested_read and not macro_read:
        raise ValueError("I require macro reads for the nested include fixture")
    if flat and not candidate:
        raise ValueError("I require candidate mode for the flat-path experiment")
    version = run([compiler, "--version"], ROOT).splitlines()[0]
    if integrated and "clang" not in version:
        raise ValueError("I require Clang for integrated assembler measurement")
    flags = list(debug_options) + (["-fno-integrated-as"] if "clang" in version and not integrated else [])
    cases = []
    for suffix in ("s", "S"):
        for shared in (False, True):
            with tempfile.TemporaryDirectory(prefix="nano-assembler-debug-") as tmp:
                directory = Path(tmp)
                module, _, env = BytecodeShadows().foreign_build_fixture(directory)
                requested_module = module
                if module_alias:
                    requested_module = directory / "import alias"
                    requested_module.symlink_to(module, target_is_directory=True)
                env["NANO_CC"] = compiler
                env["NANO_AS_CAPTURE_HELPER"] = str(ROOT / "bin/nano_as_capture.so")
                if shared: env["NANO_BUILD_CACHE"] = str(directory / "cache")
                source = module / ("payload." + suffix)
                symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                assembly = f'.text\nnop\n.data\n.globl {symbol}\n{symbol}:\n.byte 42\n'
                if macro_read:
                    payload = directory / "debug payload.bin"
                    payload.write_bytes(b"*")
                    macro_text = '.macro read_payload path\n.incbin "\\path"\n.endm\n'
                    if nested_read:
                        macro_source = directory / "debug macro.s"
                        macro_source.write_text(macro_text)
                        macro_text = '.include "' + str(macro_source) + '"\n'
                    assembly = macro_text + assembly.replace(
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
                physical_source = module.resolve() / source.name
                physical_native = directory / "physical-native.o"
                run([compiler, "-c", "-fPIC", *flags, physical_source, "-o", physical_native], directory)
                physical_text = []
                physical_expected = evidence(physical_native, physical_source, directory, physical_text)
                if debug_options and debug_options[-1] != "-g0" and (not expected["sections"] or not expected["source_named"]):
                    raise RuntimeError("I need a native object with debug sections and source provenance")
                probe = ROOT / "obj/test_module_generation_probe"
                run([probe, "build", requested_module], directory, env)
                generation = Path(run([probe, "directory", requested_module], directory, env).strip())
                production_object = generation / "answer_native_1.o"
                production_text = []
                observed = evidence(production_object, source, directory, production_text)
                native_retained = generation / "__native_unit_0_1.o"
                retained_evidence = evidence(native_retained, physical_source, directory) if native_retained.is_file() else None
                retained_identical = physical_native.read_bytes() == native_retained.read_bytes() if native_retained.is_file() else None
                captured_evidence = None
                captured_identical = None
                captured_value = None
                captured_diff = None
                removed_inputs = []
                if capture_object:
                    private = directory / "object capture"
                    private.mkdir()
                    retained_unit = private / source.name
                    retained_unit.write_bytes((generation / "__snapshot_0_1.i").read_bytes())
                    captured_object = directory / "captured.o"
                    map_option = "-fdebug-prefix-map=" if sys.platform == "darwin" or integrated else "--debug-prefix-map="
                    forward = [] if integrated else ["-Xassembler"]
                    remaps = forward + [map_option + str(private) + "=" + str(physical_source.parent)]
                    if private.resolve() != private:
                        remaps += forward + [map_option + str(private.resolve()) + "=" + str(physical_source.parent)]
                    run([compiler, "-c", "-fPIC", *flags, *remaps, "-x", "assembler",
                         retained_unit, "-o", captured_object], directory)
                    captured_text = []
                    captured_evidence = evidence(captured_object, physical_source, directory, captured_text)
                    captured_identical = physical_native.read_bytes() == captured_object.read_bytes()
                    captured_diff = ''.join(difflib.unified_diff(physical_text, captured_text,
                                                               fromfile="physical-native", tofile="captured-object"))
                    unit_inputs = [source, retained_unit]
                    if macro_read: unit_inputs.append(payload)
                    if nested_read: unit_inputs.append(macro_source)
                    saved_inputs = {path: path.read_bytes() for path in unit_inputs}
                    try:
                        for path in unit_inputs:
                            path.unlink()
                            removed_inputs.append(str(path))
                        library = directory / ("captured.dylib" if sys.platform == "darwin" else "captured.so")
                        run([compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC",
                             module / "answer.c", captured_object, "-o", library], directory)
                        captured_value = int(run([sys.executable, "-c", "import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); "
                            "l.nano_build_answer.restype=ctypes.c_int64; print(l.nano_build_answer())", library], directory))
                    finally:
                        for path, contents in saved_inputs.items(): path.write_bytes(contents)
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
                    option = "-fdebug-prefix-map=" if integrated else "-Wa,-fdebug-prefix-map=" if "clang" in version and sys.platform == "darwin" else "-Wa,--debug-prefix-map="
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
                library = run([probe, "library", requested_module], directory, env).strip()
                value = run([sys.executable, "-c", "import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); "
                    "l.nano_build_answer.restype=ctypes.c_int64; print(l.nano_build_answer())", library], directory)
                run([probe, "build", module.resolve()], directory, env)
                warm_generation = Path(run([probe, "directory", module.resolve()], directory, env).strip())
                reused = generation == warm_generation
                warm_text = []
                evidence(warm_generation / "answer_native_1.o", source, directory, warm_text)
                cases.append({"suffix": suffix, "cache": "shared" if shared else "local",
                              "published_unit_aliases": [p.name for p in generation.glob("__unit_*")],
                              "native": expected, "production": observed,
                              "retained_native_object": retained_evidence,
                              "retained_native_object_identical": retained_identical,
                              "captured_object": captured_evidence,
                              "captured_object_identical": captured_identical,
                              "captured_object_value": captured_value,
                              "captured_object_debug_diff": captured_diff,
                              "captured_object_removed_inputs": removed_inputs,
                              "lexical_source": str(source), "physical_source": str(physical_source),
                              "requested_module": str(requested_module),
                              "physical_native": physical_expected,
                              "physical_object_identical": physical_native.read_bytes() == production_object.read_bytes(),
                              "physical_debug_diff": ''.join(difflib.unified_diff(physical_text, production_text,
                                                                                  fromfile="physical-native", tofile="production")),
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
    return {"compiler": compiler, "version": version, "platform": sys.platform, "integrated": integrated,
            "candidate_mode": ("flat" if flat else "original-basename") if candidate else None,
            "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", nargs="?", default="cc")
    parser.add_argument("--require-debug", action="store_true")
    parser.add_argument("--require-physical-debug", action="store_true")
    parser.add_argument("--module-alias", action="store_true")
    parser.add_argument("--integrated", action="store_true")
    parser.add_argument("--macro-read", action="store_true")
    parser.add_argument("--nested-read", action="store_true")
    parser.add_argument("--capture-object", action="store_true")
    parser.add_argument("--require-captured-object", action="store_true")
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--flat", action="store_true",
                        help="I test directory and basename remaps without copying the retained unit")
    args = parser.parse_args()
    if args.flat and not args.candidate:
        parser.error("I require --candidate with --flat")
    if args.require_physical_debug and args.candidate:
        parser.error("I compare the physical control with production, not a candidate")
    if args.require_captured_object and not args.capture_object:
        parser.error("I require --capture-object for captured-object acceptance")
    if args.nested_read and not args.macro_read:
        parser.error("I require --macro-read with --nested-read")
    compiler = shutil.which(args.compiler)
    if not compiler: raise SystemExit("I need a C compiler")
    result = measure(compiler, candidate=args.candidate, flat=args.flat,
                     module_alias=args.module_alias, macro_read=args.macro_read, capture_object=args.capture_object,
                     nested_read=args.nested_read, integrated=args.integrated)
    print(json.dumps(result, indent=2))
    if args.require_debug and any(case["native"] != case["candidate" if args.candidate else "production"] or case["answer"] != 42
                                  or not case["generation_reused"] or (args.candidate and not case["candidate_object_identical"])
                                  for case in result["cases"]):
        raise SystemExit("I did not retain the native debug evidence and reusable result")
    if args.require_physical_debug and any(case["physical_native"] != case["production"] or
                                          not case["physical_object_identical"] or case["answer"] != 42 or
                                          not case["generation_reused"] for case in result["cases"]):
        raise SystemExit("I did not retain the physical-source native object and debug evidence")
    if args.require_captured_object and any(case["physical_native"] != case["captured_object"] or
                                           not case["captured_object_identical"] or case["captured_object_value"] != 42 or
                                           len(case["captured_object_removed_inputs"]) != 2 + args.macro_read + args.nested_read
                                           for case in result["cases"]):
        raise SystemExit("I did not retain the native object, debug evidence and source-free execution")
