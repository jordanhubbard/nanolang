#!/usr/bin/env python3
"""I retain bounded host-tool reports for my user-guide capture gate."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".test_output/userguide-ci"
sys.path.insert(0, str(ROOT))
from tests.native_process import run_captured


def configured_compiler():
    metadata = json.loads((ROOT / "modules/compiler_support/module.json").read_text())
    if "NANO_CC" in os.environ:
        configured, origin = os.environ["NANO_CC"], "NANO_CC"
    elif isinstance(metadata.get("c_compiler"), str):
        configured, origin = metadata["c_compiler"], "compiler_support.c_compiler"
    elif "CC" in os.environ:
        configured, origin = os.environ["CC"], "CC"
    else:
        configured, origin = "cc", "default"
    return configured, origin


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configured, origin = configured_compiler()
    report = {"configured_compiler": configured, "selection_origin": origin,
              "queries": [], "limits": [
                  "I record metadata probes, not successful product admission or a complete native invocation.",
                  "I preserve query errors and change no compiler, assembler, capture or shadow allowance."]}

    def query(label, argv):
        entry = {"label": label, "argv": argv}
        if argv:
            executable = shutil.which(argv[0])
            entry["executable"] = executable
            if executable:
                try:
                    entry["executable_sha256"] = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
                except OSError as error:
                    entry["executable_hash_error"] = repr(error)
        try:
            if not argv:
                raise ValueError("I have no configured compiler command.")
            result = run_captured(argv, cwd=ROOT, text=True, errors="replace", timeout=10)
            entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            entry["error"] = repr(error)
        report["queries"].append(entry)
        return entry

    try:
        compiler = shlex.split(configured)
    except ValueError as error:
        compiler = []
        report["compiler_parse_error"] = repr(error)
    query("revision", ["git", "rev-parse", "HEAD"])
    query("system", ["uname", "-a"])
    query("compiler-version", compiler + ["--version"] if compiler else [])
    for tool in ("as", "ld"):
        selected = query("compiler-selects-" + tool,
                         compiler + ["-print-prog-name=" + tool] if compiler else [])
        if selected.get("returncode") == 0:
            path = selected["stdout"].strip()
            if path and "\n" not in path:
                query("compiler-" + tool + "-version", [path, "--version"])
        query("path-" + tool + "-version", [tool, "--version"])
    query("make-version", ["make", "--version"])
    (OUTPUT / "toolchain.json").write_text(json.dumps(report, indent=2) + "\n")
    print("I retain my bounded tool reports at .test_output/userguide-ci/toolchain.json.")


if __name__ == "__main__":
    main()
