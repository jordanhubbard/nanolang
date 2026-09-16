"""I require every selected source to compile and pass verification-only mode."""
import argparse
from pathlib import Path
import sys

from dispatch_equivalence import execute


def verify(compiler, vm, sources, logs, timeout):
    logs.mkdir(parents=True, exist_ok=True)
    if not sources:
        raise RuntimeError("I refuse an empty verifier corpus")
    passed = failed = 0
    for index, source in enumerate(sources):
        stem = logs / f"{index:04d}-{source.stem}"
        artifact = stem.with_suffix(".nvm")
        try:
            artifact.unlink(missing_ok=True)
            status, _ = execute([str(compiler), str(source), "-o", str(artifact),
                                 "--emit-nvm"], timeout, stem.with_suffix(".compile.log"))
            if status or not artifact.is_file():
                raise RuntimeError(f"I could not compile {source}; log: {stem}.compile.log")
            status, _ = execute([str(vm), "--verify-only", str(artifact)],
                                timeout, stem.with_suffix(".verify.log"))
            if status:
                raise RuntimeError(f"I could not verify {source}; log: {stem}.verify.log")
            passed += 1
        except (OSError, RuntimeError) as error:
            failed += 1
            print(error, file=sys.stderr)
    print(f"{len(sources)} selected, {passed} verified, {failed} failed, 0 skipped")
    return 0 if passed and not failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--vm", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("sources", nargs="*", type=Path)
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("I require a timeout in (0, 600] seconds")
    try:
        return verify(args.compiler.resolve(), args.vm.resolve(), args.sources,
                      args.logs.resolve(), args.timeout)
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
