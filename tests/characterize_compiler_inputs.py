"""I measure candidate compiler-input evidence; I do not certify cache safety.

Run with python3 -m tests.characterize_compiler_inputs [clang-compatible-compiler].
I print JSON observations. A nonzero exit means the experiment itself failed,
not that a known cache defect was repaired or that all cases are safe.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from tests import test_bytecode_shadows as shadows


def run(argv, directory):
    result = subprocess.run([str(arg) for arg in argv], cwd=directory,
                            capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"I could not run {argv!r}: {result.stderr.decode(errors='replace')}")
    return result


def measure(compiler):
    support = shadows.BytecodeShadows()
    with tempfile.TemporaryDirectory(prefix="nano-input-evidence-") as tmp:
        directory = Path(tmp)
        module, source, env = support.foreign_build_fixture(directory)
        env["NANO_CC"] = compiler
        # I keep both names present, so a lossy path remains readable.
        actual = module / "hidden\\answer.h"
        alias = module / "hidden" / "answer.h"
        alias.parent.mkdir()
        actual.write_text("#define ANSWER 42\n")
        alias.write_text("#define ANSWER 17\n")
        translation_unit = module / "answer.c"
        translation_unit.write_text(
            '#include <stdint.h>\n#include "hidden\\answer.h"\n'
            'int64_t nano_build_answer(void) { return ANSWER; }\n')
        first, output = support.compile(source, directory, "--run", env=env)
        if first.returncode != 42:
            raise RuntimeError(f"I could not establish the original value: {first.stderr!r}")
        generation = (module / ".build" / "current").resolve(strict=True)
        record_path = generation / "source_hashes.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else {}

        # I compare saved compiler input with standalone preprocessing, without
        # assuming that the two compiler modes produce identical bytes.
        saved_object = directory / "saved.o"
        flags = [compiler, "-fPIC"]
        if sys.platform != "darwin":
            flags.append("-D_POSIX_C_SOURCE=200809L")
        run(flags + ["-c", "-save-temps=obj", translation_unit, "-o", saved_object], directory)
        snapshots = sorted(directory.glob("*.i"))
        if len(snapshots) != 1:
            raise RuntimeError(f"I expected one saved C input, found {snapshots!r}")
        snapshot = snapshots[0].read_bytes()
        graph = directory / "dependencies.dot"
        run(flags + ["-c", "-Xclang", "-dependency-dot", "-Xclang", graph,
                     translation_unit, "-o", directory / "graph.o"], directory)
        graph_text = graph.read_text()
        trace = run(flags + ["-H", "-c", translation_unit, "-o", directory / "trace.o"], directory).stderr
        original_preprocessed = run(flags + ["-E", translation_unit], directory).stdout
        unchanged_preprocessed = run(flags + ["-E", translation_unit], directory).stdout
        timestamp = actual.stat()
        actual.write_text("#define ANSWER 43\n")
        os.utime(actual, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
        changed_preprocessed = run(flags + ["-E", translation_unit], directory).stdout
        second, _ = support.compile(source.replace(" 42)", " 43)"), directory, "--run", env=env)
        replay = support.execute(output, env=env)

        # I also measure a new earlier include, which a previous dependency
        # list cannot name because that file did not exist at the first build.
        early = directory / "early"
        late = directory / "late"
        early.mkdir()
        late.mkdir()
        (late / "selected.h").write_text("#define SELECTED 42\n")
        selection = directory / "selection.c"
        selection.write_text("#include <selected.h>\nint selected = SELECTED;\n")
        selection_flags = flags + ["-I", early, "-I", late, "-E", selection]
        before_selection = run(selection_flags, directory).stdout
        (early / "selected.h").write_text("#define SELECTED 43\n")
        after_selection = run(selection_flags, directory).stdout

        # I do not assume a preprocessed file is self-contained in PCH mode.
        pch_header = directory / "precompiled.h"
        pch_header.write_text("typedef char measured_type[42];\n")
        pch = directory / "precompiled.pch"
        pch_source = directory / "uses_pch.c"
        pch_source.write_text("int main(void) { return sizeof(measured_type); }\n")
        run(flags + ["-x", "c-header", pch_header, "-o", pch], directory)
        pch_command = flags + ["-E", "-include-pch", pch, pch_source]
        pch_before = run(pch_command, directory).stdout
        pch_timestamp = pch_header.stat()
        pch_header.write_text("typedef char measured_type[43];\n")
        os.utime(pch_header, ns=(pch_timestamp.st_atime_ns, pch_timestamp.st_mtime_ns))
        stale_pch_preprocessed = run(pch_command, directory).stdout
        pch_executable = directory / "pch-program"
        run(flags + ["-include-pch", pch, pch_source, "-o", pch_executable], directory)
        stale_pch_exit = subprocess.run([str(pch_executable)], cwd=directory,
                                        capture_output=True, timeout=10).returncode
        run(flags + ["-H", "-include-pch", pch, pch_source, "-o", pch_executable], directory)
        traced_pch_exit = subprocess.run([str(pch_executable)], cwd=directory,
                                         capture_output=True, timeout=10).returncode
        run(flags + ["-save-temps=obj", "-include-pch", pch, pch_source,
                     "-o", pch_executable], directory)
        saved_pch_exit = subprocess.run([str(pch_executable)], cwd=directory,
                                        capture_output=True, timeout=10).returncode
        run(flags + ["-x", "c-header", pch_header, "-o", pch], directory)
        pch_after = run(pch_command, directory).stdout

        return {
            "compiler": compiler,
            "compiler_version": run([compiler, "--version"], directory).stdout.decode(errors="replace").splitlines()[0],
            "reusable_record_created": bool(record),
            "actual_header_recorded": "dep:" + str(actual.resolve()) in record,
            "slash_alias_recorded": "dep:" + str(alias.resolve()) in record,
            "updated_shadow_compile_exit": second.returncode,
            "published_program_exit_after_header_edit": replay.returncode,
            "cache_observed_header_edit": second.returncode == 43 and replay.returncode == 43,
            "saved_input_equals_standalone_preprocessing": snapshot == original_preprocessed,
            "dependency_graph": graph_text.replace(str(directory), "$FIXTURE"),
            "include_trace": trace.decode(errors="replace").replace(str(directory), "$FIXTURE"),
            "unchanged_preprocessing_equal": original_preprocessed == unchanged_preprocessed,
            "preprocessing_detected_actual_header_edit": original_preprocessed != changed_preprocessed,
            "preprocessing_detected_new_earlier_include": before_selection != after_selection,
            "preprocessing_detected_rebuilt_pch": pch_before != pch_after,
            "preprocessing_changed_with_stale_pch": pch_before != stale_pch_preprocessed,
            "stale_pch_direct_program_exit": stale_pch_exit,
            "stale_pch_traced_program_exit": traced_pch_exit,
            "stale_pch_saved_input_program_exit": saved_pch_exit,
            "pch_preprocessor_output": pch_before.decode(errors="replace").replace(str(directory), "$FIXTURE"),
        }


if __name__ == "__main__":
    compiler = shutil.which(sys.argv[1] if len(sys.argv) > 1 else "cc")
    if not compiler:
        raise SystemExit("I need a C compiler executable")
    print(json.dumps(measure(compiler), indent=2))
