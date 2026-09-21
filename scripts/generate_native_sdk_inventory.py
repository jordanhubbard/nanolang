#!/usr/bin/env python3
"""I derive my installed path and Make object inventories from committed inputs."""
import argparse
import json
from pathlib import Path

ROLES = ('nano_virt', 'nano_vm', 'nano_cop', 'nano_vmd', 'nanoisa', 'nvm2c',
         'nanoc', 'nanoc_c', 'nanoc_stage1')


def inputs(root):
    source = json.loads((root / 'scripts/native_sdk_inputs.json').read_text())
    objects = json.loads((root / 'scripts/native_sdk_objects.json').read_text())
    for rows in (source, objects):
        if not rows or rows != sorted(set(rows)) or len(rows) > 8192:
            raise ValueError('I require a bounded ordered SDK inventory')
        for name in rows:
            if (not isinstance(name, str) or not name or name.startswith('/') or
                    any(p in ('', '.', '..') for p in name.split('/')) or
                    any(ord(c) < 32 or ord(c) == 127 for c in name) or
                    '\\' in name or len(name.encode()) > 2048):
                raise ValueError('I require an exact relative SDK path')
    if any(not p.startswith('obj/') or not p.endswith('.o') for p in objects):
        raise ValueError('I require SDK objects under the canonical obj directory')
    paths = source + objects + ['bin/' + name for name in ROLES]
    if len(paths) != len(set(paths)):
        raise ValueError('I refuse overlapping SDK input roles')
    return sorted(paths), objects


def outputs(root):
    paths, objects = inputs(root)
    lines = ['/* I derive this file from scripts/native_sdk_{inputs,objects}.json. */',
             'static const char *const sdk_required_inputs[] = {']
    for name in sorted(paths + ['bin/nano_as_capture.so']):
        if name == 'bin/nano_as_capture.so':
            lines += ['#ifdef __linux__', '    "bin/nano_as_capture.so",', '#endif']
        else:
            lines.append('    ' + json.dumps(name, ensure_ascii=True) + ',')
    lines += ['};', '']
    make = '# I derive this list from scripts/native_sdk_objects.json.\n'
    make += 'NATIVE_SDK_OBJECTS = \\\n' + ' \\\n'.join('  $(OBJ_DIR)/' + p[4:] for p in objects) + '\n'
    return {'src/runtime/native_sdk_inventory.inc': '\n'.join(lines),
            'scripts/native_sdk_objects.mk': make}


def check(root):
    for name, expected in outputs(root).items():
        if (root / name).read_text() != expected:
            raise ValueError('I require fresh generated SDK inventory: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check(args.root)
    else:
        for name, value in outputs(args.root).items():
            (args.root / name).write_text(value)


if __name__ == '__main__':
    main()
