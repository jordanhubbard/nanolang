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
    nano = ('# I generate this exact runtime source from src/binary64_arithmetic.h.\n'
            'pub fn gen_binary64_arithmetic_runtime() -> string {\n    return ' +
            json.dumps(source.decode()) + '\n}\n'
            'shadow gen_binary64_arithmetic_runtime {\n'
            '    assert (str_contains (gen_binary64_arithmetic_runtime) "nano_rt_f64_div")\n'
            '    assert (str_contains (gen_binary64_arithmetic_runtime) "0x7ff8000000000000")\n}\n')
    outputs = {OUTPUT: text, ROOT / 'src_nano/compiler/binary64_arithmetic_runtime.nano': nano}
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit('I require regeneration of my embedded binary64 arithmetic.')
        else:
            path.write_text(content)

if __name__ == '__main__':
    main()
