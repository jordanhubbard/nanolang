"""I compare execution, not compiler exit status alone, across four paths."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("c-seed", "selfhost", "vm", "aot")


def load_cases(manifest):
    data = json.loads(manifest.read_text())
    if not isinstance(data, dict) or set(data) != {"schema", "cases"} or data["schema"] != "nanolang.language-contract.v1":
        raise ValueError("I need a language-contract.v1 manifest")
    cases = data["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("I need at least one contract case")
    if any(not isinstance(c, str) or not re.fullmatch(r"[0-9]{2}_[a-z_]+", c) for c in cases):
        raise ValueError("I need safe corpus case names")
    if len(set(cases)) != len(cases):
        raise ValueError("I reject duplicate contract cases")
    for case in cases:
        for extension in ("nano", "expected"):
            if not (ROOT / "tests/cross-backend" / f"{case}.{extension}").is_file():
                raise ValueError(f"I cannot find {case}.{extension}")
    return cases


def run(command, directory, timeout=60):
    result = subprocess.run(list(map(str, command)), cwd=ROOT,
                            env=dict(os.environ, TMPDIR=str(directory)),
                            capture_output=True, timeout=timeout)
    if result.returncode != 0:
        details = (result.stdout + result.stderr).decode(errors="replace")
        raise RuntimeError(f"I received exit {result.returncode} from {command[0]}: {details}")
    return result.stdout


def check_output(command, expected, directory):
    actual = run(command, directory, timeout=10)
    if actual != expected:
        raise RuntimeError(f"I expected stdout {expected!r}; I received {actual!r}")


def check_case(case, tools, cc, directory):
    source = ROOT / "tests/cross-backend" / f"{case}.nano"
    expected = source.with_suffix(".expected").read_bytes()
    results = {}
    for backend in ("c-seed", "selfhost"):
        try:
            binary = directory / backend
            run([tools[backend], source, "-o", binary], directory)
            check_output([binary], expected, directory)
            results[backend] = None
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            results[backend] = str(error)
    bytecode = directory / "program.nvm"
    try:
        run([tools["frontend"], source, "--emit-nvm", "-o", bytecode], directory)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        results.update(vm=str(error), aot=str(error))
        return results
    try:
        check_output([tools["vm"], bytecode], expected, directory)
        results["vm"] = None
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        results["vm"] = str(error)
    try:
        generated = directory / "program.c"
        binary = directory / "aot"
        run([tools["aot"], bytecode, "-o", generated], directory)
        run([*cc, "-std=c11", "-O2", generated, "-o", binary], directory)
        check_output([binary], expected, directory)
        results["aot"] = None
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        results["aot"] = str(error)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "tests/language-contract/v1.json")
    args = parser.parse_args()
    try:
        cases = load_cases(args.manifest)
        cc = shlex.split(os.environ.get("CC", "cc"))
        if not cc:
            raise ValueError("I need a host C compiler")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    tools = {
        "c-seed": ROOT / "bin/nanoc_c",
        "selfhost": Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve(),
        "frontend": ROOT / "bin/nano_virt",
        "vm": ROOT / "bin/nano_vm",
        "aot": ROOT / "bin/nvm2c",
    }
    failures = 0
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="nano-contract-") as scratch:
            results = check_case(case, tools, cc, Path(scratch))
        for backend in BACKENDS:
            error = results[backend]
            print(f"{'FAIL' if error else 'PASS'} {case} {backend}")
            if error:
                failures += 1
                print(error)
    print(f"I checked {len(cases) * len(BACKENDS)} execution rows; {failures} failed.")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
