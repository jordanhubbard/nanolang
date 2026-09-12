"""I measure linker evidence, not cache acceptance.

Run with python3 -m tests.characterize_linker_inputs [C-compiler].
I use temporary C/archive fixtures and the production module-builder probe.
Exit zero means I completed the experiment, not that reuse is safe.
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from tests import test_bytecode_shadows as shadows


def run(argv, directory, env=None, required=True):
    result = subprocess.run([str(value) for value in argv], cwd=directory,
                            env=env, capture_output=True, timeout=30)
    if required and result.returncode:
        raise RuntimeError(f"I could not run {argv!r}: {result.stderr!r}")
    return result


def measure(compiler, probe=None):
    probe = probe or shadows.ROOT / "obj/test_module_generation_probe"
    if not probe.is_file():
        raise RuntimeError("I need make obj/test_module_generation_probe")
    archiver = shutil.which("ar")
    if not archiver:
        raise RuntimeError("I need an archive tool")
    with tempfile.TemporaryDirectory(prefix="nano-link-evidence-") as tmp:
        directory = Path(tmp)
        module, _, env = shadows.BytecodeShadows().foreign_build_fixture(directory)
        env["NANO_CC"] = compiler
        env.pop("NANO_VERBOSE_BUILD", None)
        early, late = directory / "early", directory / "late"
        early.mkdir()
        late.mkdir()
        member_source, member_object = directory / "member.c", directory / "member.o"
        archive = late / "libselected.a"
        caller = module / "answer.c"
        caller.write_text("extern long long selected_answer(void);\n"
                          "long long nano_build_answer(void) { return selected_answer(); }\n")
        caller_object = directory / "caller.o"
        run([compiler, "-fPIC", "-c", caller, "-o", caller_object], directory)
        link_flags = ["-L" + str(early), "-L" + str(late), "-lselected"]
        (module / "module.json").write_text(json.dumps({
            "name": "answer_native", "c_sources": ["answer.c"],
            "ldflags": [shlex.quote(flag) for flag in link_flags]}))
        shared_flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]

        def build_archive(path, value):
            member_source.write_text(f"long long selected_answer(void) {{ return {value}; }}\n")
            run([compiler, "-fPIC", "-c", member_source, "-o", member_object], directory)
            run([archiver, "rcs", path, member_object], directory)

        def build_module():
            run([probe, "build", module], shadows.ROOT, env)
            generation = run([probe, "directory", module], shadows.ROOT, env).stdout.decode().strip()
            library = run([probe, "library", module], shadows.ROOT, env).stdout.decode().strip()
            return Path(generation), Path(library)

        def answer(library):
            # I load each library in a fresh process; loader handle caching
            # must not hide an on-disk replacement in this experiment.
            script = ("import ctypes, sys; library = ctypes.CDLL(sys.argv[1]); "
                      "library.nano_build_answer.restype = ctypes.c_int64; "
                      "print(library.nano_build_answer())")
            return int(run([sys.executable, "-c", script, library], directory).stdout)

        def clean(value):
            return value.decode(errors="backslashreplace").replace(str(directory), "$FIXTURE")

        def link_evidence(name, inputs):
            library = directory / (name + (".dylib" if sys.platform == "darwin" else ".so"))
            base = [compiler, *shared_flags, caller_object, *inputs, "-o", library]
            plain = run(base, directory)
            plain_answer = answer(library)
            trace = run(base + ["-Wl,-t"], directory, required=False)
            report = {
                "plain_exit": plain.returncode,
                "trace_exit": trace.returncode,
                "trace_stdout_excerpt": clean(trace.stdout[:1200]),
                "trace_stderr_excerpt": clean(trace.stderr[:1200]),
                "trace_stdout_bytes": len(trace.stdout),
                "trace_stderr_bytes": len(trace.stderr),
                "answer": plain_answer,
                "traced_answer": answer(library) if trace.returncode == 0 else None,
            }
            # I record unsupported formats as observations, not fatal errors.
            dependency = directory / (name + ".dependencies")
            options = (["-Xlinker", "-dependency_info", "-Xlinker", dependency]
                       if sys.platform == "darwin" else
                       ["-Xlinker", "--dependency-file=" + str(dependency)])
            result = run(base + options, directory, required=False)
            report["dependency_exit"] = result.returncode
            report["dependency_stderr"] = clean(result.stderr)
            report["dependency_answer"] = answer(library) if result.returncode == 0 else None
            if result.returncode == 0 and dependency.is_file():
                data = dependency.read_bytes()
                if sys.platform == "darwin":
                    # I expose raw tagged NUL-delimited records without
                    # assigning undocumented meaning to the tag values.
                    records, offset = [], 0
                    while offset < len(data):
                        end = data.find(b"\0", offset + 1)
                        if end < 0:
                            break
                        records.append({"tag": data[offset], "value": clean(data[offset + 1:end])})
                        offset = end + 1
                    report["dependency_terminated"] = offset == len(data)
                    report["dependency_record_count"] = len(records)
                    report["dependency_tag_counts"] = {
                        str(tag): sum(record["tag"] == tag for record in records)
                        for tag in sorted({record["tag"] for record in records})}
                    report["dependency_fixture_records"] = [
                        record for record in records if "$FIXTURE" in record["value"] and
                        (record["tag"] != 17 or "selected" in record["value"])]
                    report["dependency_tool_records"] = [record for record in records if record["tag"] == 0]
                    report["dependency_system_examples"] = [
                        record for record in records if record["tag"] == 16 and
                        "$FIXTURE" not in record["value"]][:3]
                else:
                    report["dependency_text"] = clean(data)
            return report

        build_archive(archive, 42)
        first_generation, first_library = build_module()
        initial_answer = answer(first_library)
        if initial_answer != 42:
            raise RuntimeError(f"I could not establish the original answer: {initial_answer}")
        warm_generation, warm_library = build_module()
        warm_answer = answer(warm_library)
        first_record = first_generation / "source_hashes.json"
        record = json.loads(first_record.read_text()) if first_record.is_file() else {}
        before_evidence = link_evidence("before", link_flags)
        stamp, original = archive.stat(), archive.read_bytes()
        build_archive(archive, 43)
        os.utime(archive, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        changed_generation, changed_library = build_module()
        changed_answer = answer(changed_library)
        after_evidence = link_evidence("after", link_flags)

        # The old input list cannot name this newly available candidate.
        build_archive(early / "libselected.a", 44)
        early_generation, early_library = build_module()
        search_evidence = link_evidence("earlier", link_flags)

        # I preserve raw unusual path bytes in the argument vector. A line
        # trace may be ambiguous even when linking itself is unambiguous.
        unusual = directory / "space ' back\\slash\nline"
        unusual.mkdir()
        unusual_archive = unusual / "libselected.a"
        build_archive(unusual_archive, 45)
        unusual_evidence = link_evidence("unusual", [unusual_archive])

        # A response file is an indirect input, not just the selected archive.
        response = directory / "link.rsp"
        response.write_text(str(archive) + "\n")
        response_evidence = link_evidence("response", ["-Xlinker", "@" + str(response)])

        thin = directory / "libthin.a"
        thin_result = run([archiver, "rcsT", thin, member_object], directory, required=False)
        thin_evidence = {"archive_exit": thin_result.returncode,
                         "archive_stderr": clean(thin_result.stderr)}
        if thin_result.returncode == 0:
            thin_evidence["thin_magic"] = thin.read_bytes().startswith(b"!<thin>\n")
            if thin_evidence["thin_magic"]:
                thin_evidence["link"] = link_evidence("thin", [thin])

        return {
            "platform": sys.platform,
            "compiler": compiler,
            "compiler_version": clean(run([compiler, "--version"], directory).stdout).splitlines()[0],
            "archiver": archiver,
            "initial_answer": initial_answer,
            "unchanged_answer": warm_answer,
            "unchanged_generation_reused": first_generation == warm_generation,
            "reusable_record_created": bool(record),
            "selected_archive_recorded": ("dep:" + str(archive) in record or
                                           str(archive) in record.get("__link_inputs_v1", {})),
            "archive_changed": hashlib.sha256(original).digest() != hashlib.sha256(archive.read_bytes()).digest(),
            "archive_size_preserved": len(original) == archive.stat().st_size,
            "archive_timestamp_preserved": archive.stat().st_mtime_ns == stamp.st_mtime_ns,
            "cache_answer_after_archive_edit": changed_answer,
            "cache_generation_changed_after_archive_edit": changed_generation != first_generation,
            "cache_answer_after_earlier_library": answer(early_library),
            "cache_generation_changed_after_earlier_library": early_generation != changed_generation,
            "before": before_evidence,
            "after_archive_edit": after_evidence,
            "after_earlier_library": search_evidence,
            "unusual_path": unusual_evidence,
            "response_file": response_evidence,
            "thin_archive": thin_evidence,
        }


if __name__ == "__main__":
    compiler = shutil.which(sys.argv[1] if len(sys.argv) > 1 else "cc")
    if not compiler:
        raise SystemExit("I need a C compiler executable")
    print(json.dumps(measure(compiler), indent=2))
