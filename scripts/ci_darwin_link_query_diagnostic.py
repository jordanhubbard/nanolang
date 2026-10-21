#!/usr/bin/env python3
"""I retain actual selected-linker evidence without broadening admission."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '.test_output/darwin-link-query'


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    commands = []
    compiler = shutil.which('cc')
    subprocess.run([compiler, '-x', 'c', '-c', '/dev/null', '-o', str(OUT / 'empty.o')], check=True, timeout=30)
    link = [compiler, '-shared', str(OUT / 'empty.o'), '-o', str(OUT / 'query.dylib')]
    for label, argv in (
        ('compiler', [compiler, '--version']),
        ('selected-linker', ['xcrun', '--find', 'ld']),
        ('driver-linker', [compiler, '-print-prog-name=ld']),
        ('linker-details', [*link, '-Wl,-version_details']),
        ('linker-driver-trace', [*link, '-###', '-Wl,-version_details']),
        ('production-grammar', [str(ROOT / 'obj/test_module_generation_probe'),
                                'link-response-grammar', shlex.join(link)]),
    ):
        result = subprocess.run(argv, cwd=OUT, capture_output=True, timeout=30)
        (OUT / (label + '.stdout')).write_bytes(result.stdout)
        (OUT / (label + '.stderr')).write_bytes(result.stderr)
        commands.append({'label': label, 'argv': argv, 'returncode': result.returncode})
        if label == 'selected-linker' and result.returncode == 0:
            path = Path(result.stdout.decode().strip()).resolve()
            commands[-1].update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')


def capture_cases():
    sys.path.insert(0, str(ROOT))
    from tests.test_source_snapshots import SourceSnapshots
    class SelectedResponseCapture(SourceSnapshots):
        pass
    names = []
    for label, suffix, shared in (('embedded', None, False), ('raw', '.s', False),
                                  ('preprocessed', '.S', False), ('shared_raw', '.s', True),
                                  ('shared_preprocessed', '.S', True)):
        def case(self, suffix=suffix, shared=shared):
            self.assembler_search_order_phases_and_recovery(
                ('xassembler-split',), suffix=suffix, shared_unit=shared,
                cases=tuple(('xassembler-split', 'common', external, cache)
                            for external in (False, True) for cache in (False, True)))
        name = 'test_selected_' + label
        setattr(SelectedResponseCapture, name, case)
        names.append(name)
    suite = unittest.TestSuite(SelectedResponseCapture(name) for name in names)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


if __name__ == '__main__':
    if sys.argv[1:] == ['capture']:
        raise SystemExit(0 if capture_cases() else 1)
    if sys.argv[1:]:
        raise SystemExit('I require no argument or capture.')
    report()
