#!/usr/bin/env python3
"""I derive self-contained AOT payload text from my existing binary-string implementation."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BEGIN = '/* I begin my mechanically shared native binary-string payload. */'
END = '/* I end my mechanically shared native binary-string payload. */'
OUTPUT = ROOT/'src/nanoisa/nvm2c_bstring_payload.inc'


def region(path):
    text = path.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError('I require one explicit binary-string payload region per input.')
    return text.split(BEGIN)[1].split(END)[0].strip()+'\n'


def payload_source():
    header = region(ROOT/'src/runtime/nl_string.h')
    implementation = region(ROOT/'src/runtime/nl_string.c')
    # I preserve every selected implementation body and change only linkage/names.
    combined = header+'\n'+implementation
    combined = re.sub(r'\bnl_string_', 'nbpayload_', combined)
    combined = re.sub(r'\butf8_sequence_length\b', 'nbpayload_sequence_length', combined)
    lines = []
    for line in combined.splitlines():
        if re.match(r'^(?:nbpayload_t\s*\*|size_t\s+|char\s+|bool\s+|int64_t\s+|int32_t\s+|void\s+|const char\s*\*|const void\s*\*)\s*nbpayload_\w+\s*\(', line):
            line = 'NBPAYLOAD_INLINE '+line
        line = line.replace('static int nbpayload_sequence_length(', 'NBPAYLOAD_INLINE int nbpayload_sequence_length(')
        lines.append(line)
    return ('#include <stddef.h>\n#include <stdint.h>\n#include <stdbool.h>\n'
            '#include <stdlib.h>\n#include <string.h>\n'
            '#if defined(__GNUC__) || defined(__clang__)\n'
            '#define NBPAYLOAD_INLINE static inline __attribute__((unused))\n'
            '#else\n#define NBPAYLOAD_INLINE static inline\n#endif\n'
            'extern bool nl_utf8_validate(const char *, size_t, size_t *);\n'
            +'\n'.join(lines)+'\n#undef NBPAYLOAD_INLINE\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--source', action='store_true')
    options = parser.parse_args()
    source = payload_source()
    if options.source:
        print(source, end='')
        return
    output = ('/* I derive this payload from runtime/nl_string.h and runtime/nl_string.c. */\n'
              +''.join(json.dumps(line+'\n')+'\n' for line in source.splitlines()))
    if options.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != output:
            raise SystemExit('I require regenerated native binary-string payload text.')
    else:
        OUTPUT.write_text(output)


if __name__ == '__main__':
    main()
