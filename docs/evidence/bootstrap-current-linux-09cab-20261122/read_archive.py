#!/usr/bin/env python3
"""I check my exact Linux reference-C bootstrap without claiming pure cutover."""
from pathlib import Path, PurePosixPath
import gzip
import hashlib
import io
import json
import shlex
import struct
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for base in manifest['base_archives']:
        assert sha((HERE / base['manifest']).read_bytes()) == base['sha256']
    files, cache, descriptors = {}, {}, {}
    for row in manifest['files']:
        assert row['path'] not in files
        descriptor = {key: row[key] for key in ('bytes', 'artifact', 'encoding',
                                               'compressed_bytes', 'compressed_sha256')}
        if row['sha256'] in descriptors:
            assert descriptor == descriptors[row['sha256']]
        else:
            descriptors[row['sha256']] = descriptor
        if row['sha256'] not in cache:
            path = HERE / row['artifact']
            assert path.resolve().is_relative_to((ROOT / 'docs/evidence').resolve())
            encoded = path.read_bytes()
            assert len(encoded) == row['compressed_bytes']
            assert sha(encoded) == row['compressed_sha256']
            assert row['encoding'] == 'gzip'
            data = gzip.decompress(encoded)
            assert len(data) == row['bytes'] and sha(data) == row['sha256']
            cache[row['sha256']] = data
        files[row['path']] = cache[row['sha256']]
    assert len(files) == manifest['counts']['records']

    def observation(name):
        return json.loads(files['observation/' + name])

    terminal, before = observation('terminal.json'), observation('before.json')
    audit = observation('audit.json')
    receipt = json.loads(files['bin/nanoc_bootstrap.json'])
    assert receipt == terminal['receipt']
    assert terminal['status'] == 'PASS' and terminal['returncode'] == manifest['make_returncode'] == 0
    assert terminal['pin'] == manifest['source_pin'] == before['source_pin']
    assert terminal['sources_unchanged']
    assert receipt['sources'] == terminal['sources_after'] == before['sources']
    assert receipt['stage_timeout_seconds'] == 1800 and receipt['shadow_timeout_seconds'] == 30
    labels = ['seed', 'seed-verify', 'stage1', 'stage1-verify', 'stage1-translate',
              'stage1-native', 'stage1-hello-compile', 'stage1-hello-execute',
              'stage2', 'stage2-verify', 'stage2-translate', 'stage2-native',
              'stage2-hello-compile', 'stage2-hello-execute', 'seed-final-verify',
              'stage1-final-verify', 'stage2-final-verify']
    assert [row['label'] for row in receipt['steps']] == labels
    prefix = 'bootstrap-work/' + PurePosixPath(receipt['work']).name + '/'
    for step in receipt['steps']:
        assert step['exit_code'] == 0
        assert sha(files[prefix + PurePosixPath(step['log']).name]) == step['log_sha256']
    assert files['bin/nanoc_stage1.nvm'] == files['bin/nanoc_stage2.nvm']
    for name, expected in receipt['artifacts'].items():
        assert sha(files['bin/' + name]) == expected
    for name in ('nanoc_stage1', 'nanoc_stage2'):
        data = files['bin/' + name]
        assert data[:4] == b'\x7fELF' and data[4:6] == bytes([2, 1])
        assert struct.unpack_from('<H', data, 18)[0] == 183
    assert files['bin/nanoc'] == files['bin/nanoc_stage2']
    assert sha(files['bin/nanoc_c']) == before['tools']['nanoc_c']
    assert terminal['nanoc_link'] == 'nanoc_stage2' and all(terminal['sentinels'].values())
    assert receipt['phase'] == 'stage2' and receipt['raw_stage1_stage2_equal']
    for label in ('stage1', 'stage2'):
        step = next(row for row in receipt['steps'] if row['label'] == label + '-hello-execute')
        assert files[prefix + PurePosixPath(step['log']).name] == b'Hello from NanoLang!\n'
    log = files['observation/make-bootstrap3.log']
    assert sha(log) == terminal['log_sha256']
    assert b'I installed bin/nanoc from the verified Stage 2 module.' in log
    assert b'bin/nanoc succeeded with bin/nanoc_c removed' in log
    assert b'byte-identical in this build.' in log
    assert len(receipt['hosts']) == 6
    for path, expected in receipt['hosts'].items():
        relative = str(PurePosixPath(path).relative_to(PurePosixPath(receipt['root'])))
        assert sha(files['host-libraries/' + relative]) == expected
    for name, expected in receipt['tools'].items():
        assert sha(files['bin/' + name]) == expected == terminal['tools_after'][name]
    for label in ('seed', 'stage1', 'stage2'):
        dump = files['observation/' + label + '-final.dump'].decode()
        imports = {shlex.split(line)[1] for line in dump.splitlines()
                   if line.startswith('.import ') and shlex.split(line)[1]}
        assert imports == set(receipt['hosts'])
    for row in manifest['guard_allowlist']:
        key = 'retained-host-inputs/' + row['path']
        if row['present']:
            assert sha(files[key]) == row['sha256']
        else:
            assert key not in files
    expected = dict(receipt['sources'])
    for path, value in expected.items():
        assert sha(files['selected-sources/' + path]) == value
    expected.update({name.removeprefix('extra-source-inputs/'): sha(data)
                     for name, data in files.items() if name.startswith('extra-source-inputs/')})
    tracked = set(subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', manifest['source_pin']], cwd=ROOT, text=True).splitlines())
    archive = subprocess.check_output(['git', 'archive', manifest['source_pin'], '--',
                                       *sorted(set(expected) & tracked)], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        actual = {entry.name: sha(source.extractfile(entry).read())
                  for entry in source if entry.isfile()}
    assert actual == {name: value for name, value in expected.items() if name in tracked}
    assert set(expected) - tracked == {'modules/index.json'}
    assert not audit['pure_translator_admission']
    assert not audit['paired_current_source_acceptance'] and not audit['release_acceptance']
    print(json.dumps(dict(status='INTEGRITY_PASS', source_pin=manifest['source_pin'],
                          records=len(files), source_inputs=len(receipt['sources']),
                          original_steps=17, raw_module_bytes=len(files['bin/nanoc_stage1.nvm']),
                          raw_module_sha256=sha(files['bin/nanoc_stage1.nvm']),
                          native_products=2, installed_no_c_seed=True,
                          pure_translator_admission=False, paired_current_source=False,
                          release_acceptance=False), indent=2))


if __name__ == '__main__':
    main()
