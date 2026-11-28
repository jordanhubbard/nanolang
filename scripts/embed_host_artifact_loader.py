#!/usr/bin/env python3
"""I embed my exact digest and deployment loader in standalone native output."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT/'src/nanoisa'/name for name in ('host_artifact_digest.h', 'host_artifact_loader.h')]
source = ''.join(p.read_text() for p in paths)
output = ROOT/'src/nanoisa/host_artifact_source.h'
text = ('/* I generate my host loader; do not edit. SHA256 '+hashlib.sha256(source.encode()).hexdigest()+' */\n'
        'static const char nano_host_artifact_source[] =\n'+
        ''.join(json.dumps(line)+'\n' for line in source.splitlines(keepends=True))+';\n')
parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
if args.check:
    if not output.exists() or output.read_text() != text:
        raise SystemExit('I require regeneration of my embedded host loader.')
else:
    output.write_text(text)
