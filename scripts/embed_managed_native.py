#!/usr/bin/env python3
"""I retain the qualified managed runtime in standalone generated native C."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = tuple(ROOT / 'src/nanoisa' / name for name in (
    'managed_strings.h', 'binary64_parse.h', 'managed_strings.c'))
OUTPUT = ROOT / 'src/nanoisa/managed_native_source.h'
REMOVED = (b'#include "managed_strings.h"\n', b'#include "binary64_parse.h"\n')


def literal(data: bytes) -> str:
    """I preserve bytes on my supported ASCII-compatible C toolchains."""
    escapes = {10: r'\n', 13: r'\r', 9: r'\t', 34: r'\"', 92: r'\\'}
    return '"' + ''.join(escapes.get(byte, chr(byte) if 32 <= byte < 127
                                   else f'\\{byte:03o}') for byte in data) + '"\n'


def generate() -> bytes:
    sources = [path.read_bytes() for path in SOURCES]
    for path, source in zip(SOURCES, sources):
        if not source.endswith(b'\n'):
            raise SystemExit(f'I require a final newline in {path.name}.')
    runtime = sources[-1]
    for include in REMOVED:
        if runtime.splitlines(keepends=True).count(include) != 1:
            raise SystemExit('I require each named runtime include exactly once.')
        runtime = runtime.replace(include, b'', 1)
    joined = sources[0] + sources[1] + runtime
    # I require explicit review before adding another embedded dependency.
    if any(line.lstrip().startswith(b'#include "') for line in joined.splitlines()):
        raise SystemExit('I require every local managed dependency to be embedded explicitly.')
    hashes = ''.join(f' * {path.name} SHA256 {hashlib.sha256(source).hexdigest()}\n'
                     for path, source in zip(SOURCES, sources))
    header = ('/* I generate exact managed native source; do not edit.\n' + hashes +
              ' * I remove only the two named local includes from managed_strings.c.\n' +
              f' * Assembled SHA256 {hashlib.sha256(joined).hexdigest()}\n */\n' +
              '#ifndef NANOISA_MANAGED_NATIVE_SOURCE_H\n#define NANOISA_MANAGED_NATIVE_SOURCE_H\n' +
              'static const char nms_native_source[] =\n')
    return (header + ''.join(literal(line) for line in joined.splitlines(keepends=True)) +
            ';\n#endif\n').encode('ascii')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = generate()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != expected:
            raise SystemExit('I require regeneration of my embedded managed native runtime.')
    else:
        OUTPUT.write_bytes(expected)


if __name__ == '__main__':
    main()
