"""I verify retained native assembler source, images and complete producer outcomes."""
import hashlib
import json
from pathlib import Path
import platform
import re
from scripts import ci_build_capture_assembler as builder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_fixed_profile(directory, machine=None):
    fixed = Path(directory).resolve(strict=True)
    build = json.loads((fixed / 'build.json').read_text())
    if (not build.get('completed') or build.get('profile') != 'fixed-as246' or
            build.get('source_sha256') != builder.FIXED_SHA256 or
            build.get('fixed_source_sha256') != builder.AFTER):
        raise RuntimeError('I refuse an incomplete or different fixed-source producer.')
    for name, expected in (('binutils-2.46.0.tar.xz', builder.FIXED_SHA256),
                           ('input-scrub-before.c', builder.BEFORE),
                           ('input-scrub-fixed.c', builder.AFTER),
                           ('bin/as', build['assembler_sha256'])):
        if digest(fixed / name) != expected:
            raise RuntimeError('I refuse changed retained source or assembler bytes.')
    image = (fixed / 'bin/as').read_bytes()[:20]
    expected_machine = {'x86_64': 62, 'aarch64': 183, 'arm64': 183}.get(machine or platform.machine())
    if (expected_machine is None or image[:6] != b'\x7fELF\x02\x01' or
            int.from_bytes(image[18:20], 'little') != expected_machine):
        raise RuntimeError('I refuse a fixed assembler for another native architecture.')
    outcomes = []
    for name in ('baseline', 'fixed'):
        outcomes.append([line for line in (fixed / (name + '-gas.sum')).read_text().splitlines()
            if re.match(r'^(PASS|FAIL|XFAIL|XPASS|KFAIL|KPASS|UNSUPPORTED|UNRESOLVED|ERROR):', line)])
    if (outcomes[0] != outcomes[1] or len(outcomes[1]) != build.get('gas_cases') or
            sum(line.startswith('PASS:') for line in outcomes[1]) != build.get('gas_expected_passes') or
            build.get('gas_expected_passes', 0) < 1000 or any(line.startswith(
                ('FAIL:', 'XPASS:', 'UNRESOLVED:', 'ERROR:')) for line in outcomes[1])):
        raise RuntimeError('I require identical complete passing native GAS outcomes.')
    witnesses = json.loads((fixed / 'witness/results.json').read_text())
    pairs = {(case, tool) for case in ('actual-markers.s', 'without-directory.s',
             'without-predef.s', 'source-marker-only.s') for tool in ('baseline', 'fixed')}
    if (witnesses != build.get('witnesses') or len(witnesses) != 8 or any(
            item['original_source_identity'] != item['expected'] or item['expected'] !=
            (item['tool'] == 'fixed' or item['case'] != 'actual-markers.s') for item in witnesses) or
            {(item['case'], item['tool']) for item in witnesses} != pairs):
        raise RuntimeError('I require all eight original native source-identity witnesses.')
    for item in witnesses:
        path = fixed / 'witness' / (item['tool'] + '-' + Path(item['case']).stem + '.o')
        if digest(path) != item['object_sha256']:
            raise RuntimeError('I refuse a changed native witness object.')
    return build
