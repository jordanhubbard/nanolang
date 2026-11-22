"""I read exact immutable capture fixtures from their retained evidence manifest."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/capture-complete-hosted-20261122'


def retained_bytes(logical):
    entries = json.loads((EVIDENCE / 'manifest.json').read_text())['files']
    entry = next(item for item in entries if item['path'] == logical)
    path = (EVIDENCE / entry['artifact']).resolve(strict=True)
    if not path.is_relative_to(ROOT / 'docs/evidence'):
        raise ValueError('I refuse a retained fixture outside my evidence directory.')
    packed = path.read_bytes()
    raw = gzip.decompress(packed)
    if (len(packed) != entry['compressed_bytes'] or len(raw) != entry['bytes'] or
            hashlib.sha256(packed).hexdigest() != entry['compressed_sha256'] or
            hashlib.sha256(raw).hexdigest() != entry['sha256']):
        raise ValueError('I refuse a changed retained capture fixture.')
    return raw


def fixed_profile(directory):
    prefix = 'observer/hosted-fixed-45009129801/fixed-as246/'
    names = ['build.json', 'binutils-2.46.0.tar.xz', 'input-scrub-before.c',
             'input-scrub-fixed.c', 'bin/as', 'baseline-gas.sum', 'fixed-gas.sum',
             'witness/results.json']
    for case in ('actual-markers', 'without-directory', 'without-predef', 'source-marker-only'):
        for tool in ('baseline', 'fixed'):
            names.append('witness/' + tool + '-' + case + '.o')
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(retained_bytes(prefix + name))


def pre_admission_inputs():
    """I keep historical qualifier premises separate from current product tests."""
    return {name: retained_bytes('corrected/' + name) for name in
            ('src/module_builder.c', 'tests/test_source_snapshots.py')}
