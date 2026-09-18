#!/usr/bin/env python3
"""I provide the exact shared binary64 formatter to my selfhost emitter."""
import argparse
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    source=(ROOT/'src/binary64_format.h').read_text()
    text=('# I generate this exact runtime source from src/binary64_format.h.\n'
          'pub fn gen_binary64_format_runtime() -> string {\n    return '+json.dumps(source)+'\n}\n'
          'shadow gen_binary64_format_runtime {\n'
          '    assert (str_contains (gen_binary64_format_runtime) "nano_rt_f64_nonfinite")\n'
          '    assert (str_contains (gen_binary64_format_runtime) "nano_rt_f64_print")\n'
          '}\n')
    output=ROOT/'src_nano/compiler/binary64_format_runtime.nano'
    if args.check:
        if not output.exists() or output.read_text()!=text:
            raise SystemExit('I require regeneration of my embedded binary64 formatter.')
    else:output.write_text(text)

if __name__=='__main__':main()
