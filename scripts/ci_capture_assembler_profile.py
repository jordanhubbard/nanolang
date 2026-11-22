"""I verify retained native assembler source, images and complete producer outcomes."""
import hashlib
import gzip
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


ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / 'docs/evidence/capture-complete-hosted-20261122'
RETAINED_MANIFEST_SHA256 = '1e9c5d0da4b187f976607d487fcf8c6cdec203516060ac4d88c7817a6878c489'
RETAINED_AS_SHA256 = '7e9edb4b7fc0c7ce97fbcdd4ecb63e3eb1c1663e2452cd32ce0249110b253bf5'
RETAINED_CC_SHA256 = '864711037b900acdb7a449cf48d912cefa935b527fb60c767ccb956779e66136'
RETAINED_PREFIX = 'observer/hosted-fixed-45009129801/fixed-as246/'


def retained_capture_bytes(logical):
    manifest = (RETAINED / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != RETAINED_MANIFEST_SHA256:
        raise RuntimeError('I refuse a changed retained native evidence manifest.')
    entries = [entry for entry in json.loads(manifest)['files'] if entry['path'] == logical]
    if len(entries) != 1:
        raise RuntimeError('I require one exact retained native input.')
    entry = entries[0]
    path = (RETAINED / entry['artifact']).resolve(strict=True)
    if not path.is_relative_to(ROOT / 'docs/evidence'):
        raise RuntimeError('I refuse a retained native input outside my evidence directory.')
    packed = path.read_bytes()
    raw = gzip.decompress(packed)
    if (len(packed) != entry['compressed_bytes'] or len(raw) != entry['bytes'] or
            hashlib.sha256(packed).hexdigest() != entry['compressed_sha256'] or
            hashlib.sha256(raw).hexdigest() != entry['sha256']):
        raise RuntimeError('I refuse changed retained native input bytes.')
    return raw


def retained_profile_metadata():
    metadata = json.loads(retained_capture_bytes(RETAINED_PREFIX + 'build.json'))
    if (metadata.get('assembler_sha256') != RETAINED_AS_SHA256 or
            metadata.get('compiler_sha256') != RETAINED_CC_SHA256):
        raise RuntimeError('I refuse an unidentified retained native producer.')
    return metadata


def materialize_retained_profile(directory, machine=None):
    # I read ELF as data here; only the separately guarded Linux selector executes it.
    if (machine or platform.machine()) != 'x86_64':
        raise RuntimeError('I require qualification of a retained profile for this native architecture.')
    directory = Path(directory)
    if directory.is_symlink() or (directory.exists() and any(directory.iterdir())):
        raise RuntimeError('I refuse to overwrite a retained native profile directory.')
    expected = retained_profile_metadata()
    directory.mkdir(parents=True, exist_ok=True)
    names = ['build.json', 'binutils-2.46.0.tar.xz', 'input-scrub-before.c',
             'input-scrub-fixed.c', 'bin/as', 'baseline-gas.sum', 'fixed-gas.sum',
             'witness/results.json']
    for case in ('actual-markers', 'without-directory', 'without-predef', 'source-marker-only'):
        for tool in ('baseline', 'fixed'):
            names.append('witness/' + tool + '-' + case + '.o')
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(retained_capture_bytes(RETAINED_PREFIX + name))
    actual = read_fixed_profile(directory, machine=machine)
    if actual != expected:
        raise RuntimeError('I refuse changed retained producer metadata after materialization.')
    return actual
