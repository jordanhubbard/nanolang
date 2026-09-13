"""I measure restored source changes; exit zero is not cache acceptance.

Run with python3 -m tests.characterize_source_snapshot [C-compiler].
Add --require-consistent to fail when cold or warm answers differ from fresh.
Add --assembler to include external binary input read by inline assembly.
Add --external-assembler to measure Clang's non-integrated assembler mode.
Add --alternate-assembler to measure GNU alternate-macro input capture.
Add --assembler-units to measure mixed C and standalone .s/.S sources.
Add --split-search to forward assembler operands in separate metadata entries.
Add --response to include compiler arguments read from a response file.
Add --response-large to exercise argument lists beyond inline capture limits.
Add --link-response to exercise driver response files in linker metadata.
Add --forwarded-response to exercise response files forwarded to the linker.
I execute the production builder and load each library in a fresh process.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile

from tests.characterize_linker_inputs import run
from tests import test_bytecode_shadows as shadows


def measure(compiler, kinds=("source", "header"), payload_name=None, remove_input=False, split_search=False):
    probe = shadows.ROOT / "obj/test_module_generation_probe"
    if not probe.is_file():
        raise RuntimeError("I need make obj/test_module_generation_probe")
    cases = []
    for kind in kinds:
        for shared in (False, True):
            with tempfile.TemporaryDirectory(prefix="nano-source42-snapshot-" if kind.startswith("link-response") else "nano-source-snapshot-") as tmp:
                directory = Path(tmp)
                module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
                env.pop("NANO_VERBOSE_BUILD", None)
                if shared:
                    env["NANO_BUILD_CACHE"] = str(directory / "cache")
                source = module / "answer.c"
                target = source
                fresh_flags = []
                extra_sources = []
                if kind.startswith("link-response"):
                    for value in (42, 43):
                        member, obj = directory / "member.c", directory / "member.o"
                        member.write_text(f"long long selected(void) {{ return {value}; }}\n")
                        run([compiler, "-fPIC", "-c", member, "-o", obj], directory)
                        run(["ar", "rcs", directory / f"selected{value}.a", obj], directory)
                    target = module / "link.rsp"
                    target.write_text(str(directory / "selected42.a") + "\n")
                    source.write_text("extern long long selected(void);\n"
                                      "long long nano_build_answer(void) { return selected(); }\n")
                    fresh_flags = [("-Wl,@" if "-forwarded" in kind else "@") + str(target)]
                    metadata = {"name": "answer_native", "c_sources": ["answer.c"]}
                    if kind.endswith("-pkg"):
                        metadata["pkg_config"] = ["link-fixture"]
                        pkg = directory / "pkg-config"
                        pkg.write_text(f"#!{sys.executable}\nimport sys\n"
                                       f"if '--libs' in sys.argv: print({fresh_flags[0]!r})\n")
                        pkg.chmod(0o700)
                        env["PKG_CONFIG"] = str(pkg)
                    else:
                        field = ("ldflags_macos" if sys.platform == "darwin" else "ldflags_linux") if kind.endswith("-platform") else "ldflags"
                        metadata[field] = fresh_flags
                    (module / "module.json").write_text(json.dumps(metadata))
                elif kind == "header":
                    target = module / "answer.h"
                    target.write_text("#define ANSWER 42\n")
                    source.write_text('#include <stdint.h>\n#include "answer.h"\n'
                                      'int64_t nano_build_answer(void) { return ANSWER; }\n')
                elif kind in ("response", "response-large"):
                    target = module / "flags.rsp"
                    padding = "-DNANO_PADDING=1\n" * 600 if kind == "response-large" else ""
                    target.write_text(padding + "-DANSWER=42\n")
                    source.write_text('long long nano_build_answer(void) { return ANSWER; }\n')
                    metadata = json.loads((module / "module.json").read_text())
                    fresh_flags = ["@" + str(target)]
                    metadata["cflags"] = fresh_flags
                    (module / "module.json").write_text(json.dumps(metadata))
                elif kind.startswith("assembler"):
                    if kind.startswith("assembler-external"):
                        metadata = json.loads((module / "module.json").read_text())
                        fresh_flags = ["-fno-integrated-as"]
                        if kind.endswith("-debug"): fresh_flags += ["-O2", "-g"]
                        metadata["cflags"] = fresh_flags
                        (module / "module.json").write_text(json.dumps(metadata))
                    target = module / (payload_name if payload_name is not None else "answer.bin")
                    if payload_name is not None:
                        env["NANO_AS_CAPTURE_HELPER"] = str(shadows.ROOT / "bin/nano_as_capture.so")
                    target.write_bytes(b"42")
                    symbol = "_snapshot_payload" if sys.platform == "darwin" else "snapshot_payload"
                    directive = f'.incbin {json.dumps(str(target), ensure_ascii=False)}'
                    if kind in ("assembler-alternate", "assembler-external-alternate"):
                        # Angle-bracket macro arguments require GNU alternate mode.
                        # I do not enable .altmacro in source: the driver flag must work.
                        include = module / "alternate.s"
                        include.write_text('.macro payload file\n.incbin "\\file"\n.endm\npayload <' + str(target) + '>\n')
                        directive = f'.include "{include}"'
                        fresh_flags.extend(["-Xassembler", "--alternate"] if split_search
                                           else ["-Wa,--alternate"])
                        metadata = json.loads((module / "module.json").read_text())
                        metadata["cflags"] = [shlex.quote(flag) for flag in fresh_flags]
                        (module / "module.json").write_text(json.dumps(metadata))
                        env["NANO_AS_CAPTURE_HELPER"] = str(shadows.ROOT / "bin/nano_as_capture.so")
                    elif kind in ("assembler-search", "assembler-external-search"):
                        search = module / "assembler includes"
                        search.mkdir()
                        (search / "selected.s").write_text(directive + "\n")
                        directive = '.include "selected.s"'
                        fresh_flags.extend(["-Xassembler", "-I", "-Xassembler", str(search)] if split_search
                                           else ["-Wa,-I," + str(search)])
                        metadata = json.loads((module / "module.json").read_text())
                        metadata["cflags"] = [shlex.quote(flag) for flag in fresh_flags]
                        (module / "module.json").write_text(json.dumps(metadata))
                        env["NANO_AS_CAPTURE_HELPER"] = str(shadows.ROOT / "bin/nano_as_capture.so")
                    elif kind in ("assembler-nested", "assembler-include"):
                        target.rename(module / "payload with 'quotes'.bin")
                        target = module / "payload with 'quotes'.bin"
                        target.write_bytes(b"xx42yy")
                        inner, outer = module / "inner.s", module / "outer.s"
                        inner.write_text('.macro payload\n.incbin "' +
                            os.path.relpath(target, directory) + '", (1+1), (3-1)\n.endm\npayload\n')
                        outer.write_text(f'.include "{os.path.relpath(inner, directory)}"\n')
                        directive = f'.include "{os.path.relpath(outer, directory)}"'
                        if kind == "assembler-include":
                            inner.write_text('.ascii "42"\n')
                            target = inner
                    elif kind in ("assembler-fallback", "assembler-macro", "assembler-external-macro", "assembler-external-macro-debug", "assembler-external-macro-query-failure"):
                        include = module / "macro.s"
                        include.write_text('.macro payload file\n.incbin "\\file"\n.endm\npayload "' + str(target) + '"\n')
                        directive = f'.include "{include}"'
                        env["NANO_AS_CAPTURE_HELPER"] = str(
                            directory / "missing-helper.so" if kind == "assembler-fallback"
                            else shadows.ROOT / "bin/nano_as_capture.so")
                    assembly = f'.data\n.globl {symbol}\n{symbol}:\n{directive}\n.text\n'
                    standalone = kind in ("assembler-unit", "assembler-preprocessed-unit")
                    if standalone:
                        unit = module / ("payload.S" if kind == "assembler-preprocessed-unit" else "payload.s")
                        if kind == "assembler-preprocessed-unit":
                            assembly = '#define PAYLOAD ' + json.dumps(str(target)) + '\n' + assembly.replace(directive, '.incbin PAYLOAD')
                        unit.write_text(assembly)
                        extra_sources.append(unit)
                        metadata = json.loads((module / "module.json").read_text())
                        metadata["c_sources"].append(unit.name)
                        (module / "module.json").write_text(json.dumps(metadata))
                        env["NANO_AS_CAPTURE_HELPER"] = str(shadows.ROOT / "bin/nano_as_capture.so")
                    source.write_text('extern const unsigned char snapshot_payload[];\n'
                        + ('' if standalone else '__asm__(' + json.dumps(assembly) + ');\n') +
                        'long long nano_build_answer(void) {\n'
                        'return (snapshot_payload[0] - 48) * 10 + snapshot_payload[1] - 48;\n}\n')
                    if kind in ("assembler-alternate", "assembler-external-alternate"):
                        ordinary = subprocess.run([compiler, "-fPIC", "-c", source,
                            *(["-fno-integrated-as"] if kind.startswith("assembler-external") else []),
                            "-o", directory / "ordinary.o"], cwd=directory,
                            capture_output=True, timeout=20)
                        if ordinary.returncode == 0:
                            raise RuntimeError("I need a fixture that requires alternate macro syntax")
                original, stamp = target.read_bytes(), target.stat()
                wrapper, marker = directory / "cc", directory / "mutated"
                calls = directory / "calls"
                wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
if {kind == "assembler-external-macro-query-failure"!r} and "-###" in sys.argv:
    sys.exit(1)
if {kind == "capture-failure"!r} and any(arg in sys.argv for arg in ("-S", "-E")):
    print("I failed the requested capture phase", file=sys.stderr)
    sys.exit(1)
if "-S" in sys.argv or "-E" in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write(("S" if "-S" in sys.argv else "E") + "\\n")
if "-c" in sys.argv and "-###" not in sys.argv and "-S" not in sys.argv:
    with open({str(calls)!r}, "a") as log: log.write("C\\n")
    if "assembler" in sys.argv:
        with open({str(calls)!r}, "a") as log:
            log.write(("external" if "-fno-integrated-as" in sys.argv else "integrated") + "\\n")
if {"(('-shared' in sys.argv or '-dynamiclib' in sys.argv) and not any(a in sys.argv for a in ('-Wl,--version', '-Wl,-version_details')))" if kind.startswith("link-response") else "('-c' in sys.argv and '-###' not in sys.argv and '-S' not in sys.argv)"}:
    marker = pathlib.Path({str(marker)!r})
    if not marker.exists() and os.getenv("NANO_AS_CAPTURE_PHASE") != "capture" and ({not extra_sources!r} or any(pathlib.Path(arg).name in {[path.name for path in extra_sources]!r} for arg in sys.argv[1:])):
        target = pathlib.Path({str(target)!r})
        original, stamp = target.read_bytes(), target.stat()
        changed = original.replace({(b"selected42.a" if kind.startswith("link-response") else b"42")!r},
                                   {(b"selected43.a" if kind.startswith("link-response") else b"43")!r})
        assert changed != original and len(changed) == len(original)
        try:
            if {remove_input!r}:
                target.unlink()
            else:
                target.write_bytes(changed)
                os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            result = subprocess.run([{compiler!r}] + sys.argv[1:])
        finally:
            target.write_bytes(original)
            os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        marker.write_text(str(result.returncode))
        sys.exit(result.returncode)
os.execv({compiler!r}, [{compiler!r}] + sys.argv[1:])
''')
                wrapper.chmod(0o700)
                env["NANO_CC"] = str(wrapper)

                def query(mode):
                    return Path(run([probe, mode, module], directory, env).stdout.decode().strip())

                def answer(library):
                    result = run([sys.executable, "-c",
                        "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
                        "lib.nano_build_answer.restype=ctypes.c_int64; "
                        "print(lib.nano_build_answer())", library], directory)
                    return int(result.stdout)

                native_answer = None
                if extra_sources:
                    native = directory / ("native.dylib" if sys.platform == "darwin" else "native.so")
                    run([compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                         "-fPIC", source, *extra_sources, *fresh_flags, "-o", native], directory)
                    native_answer = answer(native)
                    if native_answer != 42:
                        raise RuntimeError("I need the mixed-source native baseline to return 42")

                if kind in ("assembler-external-macro-query-failure", "assembler-fallback", "capture-failure",
                            "assembler-unit", "assembler-preprocessed-unit"):
                    built = subprocess.run([probe, "build", module], cwd=directory, env=env, capture_output=True, timeout=20)
                    if built.returncode:
                        root = query("root")
                        cases.append({
                            "input": kind, "cache": "shared" if shared else "local", "build_failed": True,
                            "native_answer": native_answer,
                            "diagnostic": built.stderr.decode(), "mutation_started": marker.exists(),
                            "total_object_compilations": calls.read_text().splitlines().count("C") if calls.exists() else 0,
                            "published_generations": len(list(root.glob(".nano-gen-*"))),
                            "leaked_stages": len(list(root.glob(".nano-build-*"))),
                            "current_exists": os.path.lexists(root / "current"),
                            "bytes_restored": target.read_bytes() == original,
                            "size_preserved": target.stat().st_size == stamp.st_size,
                            "mtime_preserved": target.stat().st_mtime_ns == stamp.st_mtime_ns,
                        })
                        continue
                else:
                    query("build")
                cold_generation = query("directory")
                cold_answer = answer(query("library"))
                cold_calls = calls.read_text().splitlines()
                record = (cold_generation / "source_hashes.json").is_file()
                query("build")
                warm_generation = query("directory")
                warm_answer = answer(query("library"))
                fresh = directory / ("fresh.dylib" if sys.platform == "darwin" else "fresh.so")
                run([compiler, "-dynamiclib" if sys.platform == "darwin" else "-shared",
                     "-fPIC", source, *extra_sources, *fresh_flags, "-o", fresh], directory)
                if marker.read_text() != "0" or target.read_bytes() != original:
                    raise RuntimeError("I did not complete and restore the controlled compilation")
                cases.append({
                    "input": kind, "cache": "shared" if shared else "local",
                    "input_bytes": len(original),
                    "bytes_restored": target.read_bytes() == original,
                    "size_preserved": target.stat().st_size == stamp.st_size,
                    "mtime_preserved": target.stat().st_mtime_ns == stamp.st_mtime_ns,
                    "cold_answer": cold_answer, "warm_answer": warm_answer,
                    "fresh_answer": answer(fresh), "reuse_record": record,
                    "generation_reused": cold_generation == warm_generation,
                    "retained_translation_unit": (cold_generation / "__snapshot_0_0.i").is_file(),
                    "retained_assembly": (cold_generation / "__snapshot_0_0.s").is_file(),
                    "retained_assembler_files": len(list(cold_generation.glob("__assembler_*"))),
                    "retained_read_manifest": (cold_generation / "__as_read_0_0.manifest0").is_file(),
                    "target_in_reuse_record": record and str(target) in
                        (cold_generation / "source_hashes.json").read_text(),
                    "cold_object_compilations": cold_calls.count("C"),
                    "total_object_compilations": calls.read_text().splitlines().count("C"),
                    "external_assembly_compilations": calls.read_text().splitlines().count("external"),
                    "cold_assembly_captures": cold_calls.count("S"),
                    "total_assembly_captures": calls.read_text().splitlines().count("S"),
                })
    return {"platform": sys.platform, "compiler": compiler,
            "compiler_version": run([compiler, "--version"], shadows.ROOT).stdout.decode().splitlines()[0],
            "cases": cases}


def require_consistent(result):
    if any(case.get("build_failed") for case in result["cases"]):
        raise SystemExit("I could not complete a characterized build.")
    if any(not (case["cold_answer"] == case["warm_answer"] == case["fresh_answer"])
           for case in result["cases"]):
        raise SystemExit("I compiled or reused code that differs from the restored inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", nargs="?", default="cc")
    parser.add_argument("--require-consistent", action="store_true")
    parser.add_argument("--assembler", action="store_true")
    parser.add_argument("--external-assembler", action="store_true")
    parser.add_argument("--alternate-assembler", action="store_true")
    parser.add_argument("--assembler-units", action="store_true")
    parser.add_argument("--split-search", action="store_true")
    parser.add_argument("--response", action="store_true")
    parser.add_argument("--response-large", action="store_true")
    parser.add_argument("--link-response", action="store_true")
    parser.add_argument("--forwarded-response", action="store_true")
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if not compiler:
        raise SystemExit("I need a C compiler executable")
    kinds = ("source", "header")
    if args.assembler: kinds += ("assembler",)
    if args.assembler_units: kinds += ("assembler-unit", "assembler-preprocessed-unit")
    if args.external_assembler: kinds += ("assembler-external",)
    if args.alternate_assembler:
        kinds += ("assembler-external-alternate" if args.external_assembler else "assembler-alternate",)
    if args.response: kinds += ("response",)
    if args.response_large: kinds += ("response-large",)
    if args.link_response: kinds += ("link-response", "link-response-platform", "link-response-pkg")
    if args.forwarded_response: kinds += ("link-response-forwarded", "link-response-forwarded-platform", "link-response-forwarded-pkg")
    result = measure(compiler, kinds, split_search=args.split_search)
    print(json.dumps(result, indent=2))
    if args.require_consistent:
        require_consistent(result)
