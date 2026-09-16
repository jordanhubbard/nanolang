"""I compare every selected program, retaining exact output and failure logs."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


def execute(command, timeout, log):
    with log.open("wb") as output:
        process = subprocess.Popen(command, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise RuntimeError(f"I timed out: {command[0]}; log: {log}")
    # I snapshot the available bytes; a surviving descendant cannot keep a
    # pipe open or extend this read indefinitely. This is not process isolation.
    with log.open("rb") as output:
        return process.returncode, output.read(os.fstat(output.fileno()).st_size)


def compare(compiler, vm_goto, vm_switch, sources, logs, timeout):
    logs.mkdir(parents=True, exist_ok=True)
    if not sources:
        raise RuntimeError("I refuse an empty dispatch-equivalence corpus")
    same = failed = 0
    for index, source in enumerate(sources):
        stem = logs / f"{index:04d}-{source.stem}"
        artifact = stem.with_suffix(".nvm")
        artifact.unlink(missing_ok=True)
        try:
            status, _ = execute([str(compiler), str(source), "-o", str(artifact),
                                 "--emit-nvm"], timeout, stem.with_suffix(".compile.log"))
            if status or not artifact.is_file():
                raise RuntimeError(f"I could not compile {source}; log: {stem}.compile.log")
            left = execute([str(vm_goto), str(artifact)], timeout, stem.with_suffix(".goto.log"))
            right = execute([str(vm_switch), str(artifact)], timeout, stem.with_suffix(".switch.log"))
            if left[0] < 0 or right[0] < 0:
                raise RuntimeError(f"I observed a VM signal for {source}; logs: {stem}.*.log")
            left = (left[0], left[1].replace(os.fsencode(vm_goto), b"<VM>"))
            right = (right[0], right[1].replace(os.fsencode(vm_switch), b"<VM>"))
            if left != right:
                raise RuntimeError(f"I found differing output/status for {source}; logs: {stem}.*.log")
            same += 1
        except (OSError, RuntimeError) as error:
            failed += 1
            print(error, file=sys.stderr)
    print(f"{len(sources)} selected, {same} identical, {failed} failed, 0 skipped")
    return 0 if same and not failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--vm-goto", type=Path, required=True)
    parser.add_argument("--vm-switch", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("sources", nargs="*", type=Path)
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("I require a timeout in (0, 600] seconds")
    try:
        return compare(args.compiler.resolve(), args.vm_goto.resolve(),
                       args.vm_switch.resolve(), args.sources, args.logs.resolve(), args.timeout)
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
