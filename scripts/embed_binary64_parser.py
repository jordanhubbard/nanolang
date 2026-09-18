#!/usr/bin/env python3
"""I retain one exact parser source in standalone generated C output."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'src/nanoisa/binary64_parse.h'
OUTPUT = ROOT/'src/nanoisa/binary64_parse_source.h'
source = SOURCE.read_bytes()
text = ('/* I generate this from binary64_parse.h; do not edit. SHA256 '+
        hashlib.sha256(source).hexdigest()+' */\n'
        'static const char nbp_parser_source[] =\n'+
        ''.join(json.dumps(line)+'\n' for line in source.decode().splitlines(keepends=True))+';\n')
parser=argparse.ArgumentParser()
parser.add_argument('--check',action='store_true')
args=parser.parse_args()
if args.check:
    if not OUTPUT.exists() or OUTPUT.read_text()!=text:
        raise SystemExit('I require regeneration of my embedded binary64 parser.')
else:
    OUTPUT.write_text(text)
