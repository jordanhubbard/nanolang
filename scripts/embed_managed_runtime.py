#!/usr/bin/env python3
"""I package target-specific runtime IR without enabling executable admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'src/nanoisa/managed_strings.c'
HEADER = 'src/nanoisa/managed_strings.h'


def invoke(args, **kwargs):
    result = subprocess.run(args, capture_output=True, text=True,
                            timeout=60, cwd=ROOT, **kwargs)
    if result.returncode:
        raise RuntimeError('I could not build or verify runtime IR: '+
                           shlex.join(args)+'\n'+result.stdout+result.stderr)
    return result.stdout


def generate(clang, opt):
    version = invoke(clang + ['--version']).splitlines()[0]
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in (SOURCE, HEADER, 'scripts/embed_managed_runtime.py')}
    variants = {}
    with tempfile.TemporaryDirectory(prefix='nano-runtime-ir-') as directory:
        for target in ('native', 'wasm32'):
            flags = ['-std=c11', '-O2', '-ffreestanding', '-fno-builtin',
                     '-fno-ident', '-Wall', '-Wextra', '-Werror', '-S', '-emit-llvm']
            if target == 'wasm32':
                flags += ['--target=wasm32-unknown-unknown']
            else:
                flags += shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS', ''))
            output = Path(directory)/(target+'.ll')
            invoke(clang + flags + [SOURCE, '-o', str(output)])
            invoke(opt + ['-passes=verify', '-disable-output', str(output)])
            ir = output.read_text()
            triple = re.search(r'^target triple = "([^"]+)"$', ir, re.M).group(1)
            layout = re.search(r'^target datalayout = "([^"]+)"$', ir, re.M).group(1)
            if target == 'wasm32':
                if triple != 'wasm32-unknown-unknown' or 'p:32:32' not in layout:
                    raise ValueError('I require the explicit wasm32 runtime ABI')
                if re.search(r'^declare .*@(malloc|free)\(', ir, re.M):
                    raise ValueError('I refuse allocator imports in my Wasm runtime')
            elif triple.startswith('wasm'):
                raise ValueError('I require a native 64-bit runtime target')
            if 'nms_test_' in ir:
                raise ValueError('I refuse test hooks in packaged runtime IR')
            variants[target] = {'ir': ir, 'triple': triple, 'layout': layout,
                                'sha256': hashlib.sha256(ir.encode()).hexdigest(),
                                'flags': flags}
    manifest = {'schema': 1, 'clang': version, 'sources': hashes,
                'variants': {key: {k: v for k, v in value.items() if k != 'ir'}
                             for key, value in variants.items()}}
    lines = ['/* I generate this runtime package; edit its C source, not this file. */',
             '#ifndef NANOISA_MANAGED_RUNTIME_IR_H', '#define NANOISA_MANAGED_RUNTIME_IR_H']
    for target, value in variants.items():
        lines.append('static const char nms_runtime_ir_'+target+'[] =')
        lines += [json.dumps(line+'\n') for line in value['ir'].splitlines()]
        lines.append(';')
    lines += ['#endif', '']
    return '\n'.join(lines), manifest, variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clang', default='clang')
    parser.add_argument('--opt', default='opt')
    parser.add_argument('--header', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    header, manifest, _ = generate(shlex.split(args.clang), shlex.split(args.opt))
    # I finish compilation and validation before publishing either output.
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(header)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
