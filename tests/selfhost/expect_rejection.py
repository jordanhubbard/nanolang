#!/usr/bin/env python3
"""I accept only a bounded, diagnostic-matched compiler rejection."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("I require a compiler command after --")
    if args.timeout <= 0:
        parser.error("I require a positive timeout")
    if not args.require:
        parser.error("I require at least one diagnostic fragment")
    return args


def retain_log(path: Path, output: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def fail(message: str) -> int:
    print(f"I cannot accept this rejection: {message}", file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    if args.output.exists():
        return fail(f"the output already exists: {args.output}")

    try:
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as error:
        retain_log(args.log, str(error).encode(errors="replace") + b"\n")
        return fail(f"I could not launch {args.command[0]}: {error}")

    try:
        output, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        final_output, _ = process.communicate()
        retained = final_output or error.output or b""
        retain_log(args.log, retained)
        return fail(f"the compiler exceeded {args.timeout:g} seconds")

    retain_log(args.log, output)
    if process.returncode < 0:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return fail(f"the compiler died from signal {-process.returncode}")
    if process.returncode == 0:
        return fail("the compiler accepted the invalid program")
    if args.output.exists():
        return fail(f"the rejected compile published an output: {args.output}")

    decoded = output.decode(errors="replace")
    missing = [fragment for fragment in args.require if fragment not in decoded]
    if missing:
        return fail("the compiler did not emit: " + ", ".join(repr(item) for item in missing))

    print(f"I observed the required semantic rejection in {args.log}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
