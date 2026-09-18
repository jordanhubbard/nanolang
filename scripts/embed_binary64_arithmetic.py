#!/usr/bin/env python3
"""I embed one exact scalar arithmetic policy in standalone generated C."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/binary64_arithmetic.h'
OUTPUT = ROOT / 'src/binary64_arithmetic_source.h'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = SOURCE.read_bytes()
    text = ('/* I generate this from binary64_arithmetic.h; do not edit. SHA256 ' +
            hashlib.sha256(source).hexdigest() + ' */\n'
            '#ifndef NANOLANG_BINARY64_ARITHMETIC_SOURCE_H\n'
            '#define NANOLANG_BINARY64_ARITHMETIC_SOURCE_H\n'
            'static const char nl_binary64_arithmetic_source[] =\n' +
            ''.join(json.dumps(line) + '\n' for line in source.decode().splitlines(keepends=True)) +
            ';\n#endif\n')
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            raise SystemExit('I require regeneration of my embedded binary64 arithmetic.')
    else:
        OUTPUT.write_text(text)

if __name__ == '__main__':
    main()
