#!/usr/bin/env python3
"""I verify my constructor repair while retaining the full lifetime failures."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'].startswith('315d9375c4')
    for entry in manifest['base_archives']:
        assert sha((HERE/entry['path']).read_bytes()) == entry['sha256']
    files, objects, descriptors = {}, {}, {}
    for row in manifest['files']:
        assert row['path'] not in files
        descriptor = tuple(row[k] for k in ('artifact', 'bytes', 'compressed_bytes', 'compressed_sha256'))
        if row['sha256'] not in objects:
            path = (HERE/row['artifact']).resolve(strict=True)
            assert path.is_relative_to((ROOT/'docs/evidence').resolve())
            packed = path.read_bytes()
            assert (len(packed), sha(packed)) == (row['compressed_bytes'], row['compressed_sha256'])
            raw = gzip.decompress(packed)
            assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
            objects[row['sha256']] = raw
            descriptors[row['sha256']] = descriptor
        assert descriptors[row['sha256']] == descriptor
        files[row['path']] = objects[row['sha256']]
    counts = manifest['counts']
    assert len(files) == counts['records']
    assert len(objects) == counts['unique_objects']
    assert sum(map(len, files.values())) == counts['logical_bytes']
    own = {Path(r['artifact']).name for r in manifest['files'] if Path(r['artifact']).parent == Path('objects')}
    assert own == {p.name for p in (HERE/'objects').iterdir()}
    paths = [r['path'] for r in manifest['git_sources']]
    tree = subprocess.check_output(['git', 'ls-tree', '-rz', manifest['source_pin'], '--', *paths], cwd=ROOT)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            meta, path = item.split(b'\t', 1)
            blobs[path.decode()] = meta.split()[2].decode()
    assert len(blobs) == len(paths) == counts['sources']
    for row in manifest['git_sources']:
        raw = files['source/'+row['path']]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob'] == blobs[row['path']]
    q = json.loads(files['journal/qualification.json'])
    assert q['source_pin'] == manifest['source_pin']
    for key in ('full_fixture_sanitizer_acceptance', 'repl_runtime_acceptance', 'field_retirement_enabled', 'original_strict_assertion_changed', 'deadline_changed'):
        assert q[key] is False
    assert [r['exit'] for r in q['normal']['commands']] == [0, 0, 0]
    assert [r['exit'] for r in q['sanitizers']['commands']] == [0, 2, 2]
    assert [r['exit'] for r in q['neighbors']] == [0, 0, 0, 0, 0, 1, 0, 0]
    for group in ('normal', 'sanitizers'):
        for name, identity in q[group]['files'].items():
            assert sha(files['source/'+name]) == identity
    assert q['nvm2c_sha256'] == sha(files['source/src/nanoisa/nvm2c.c']) == '12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67'
    assert q['nvm2c_image_sha256'] == sha(files['images/bin/nvm2c']) == 'c94adddd92eefc081ee5cb28df971a36fe2f7cbe8e9730ec68ca6b1c3d24a7ea'
    base = json.loads((HERE/manifest['base_archives'][0]['path']).read_text())
    def baseline_raw(name):
        row = next(r for r in base['files'] if r['path'] == 'source/'+name)
        path = (HERE/manifest['base_archives'][0]['path']).parent/row['artifact']
        packed = path.read_bytes()
        assert (len(packed), sha(packed)) == (row['compressed_bytes'], row['compressed_sha256'])
        raw = gzip.decompress(packed)
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        return raw
    assert files['source/tests/test_eval_callable_lifetimes.c'] == baseline_raw('tests/test_eval_callable_lifetimes.c')
    assert files['source/src/env.c'] == baseline_raw('src/env.c')
    header_comment = b'/* I return owned callable name/signature metadata; my caller retires both.\n * Other result kinds retain their existing lifetime rules. */\n'
    assert files['source/src/nanolang.h'].count(header_comment) == 1
    assert files['source/src/nanolang.h'].replace(header_comment, b'') == baseline_raw('src/nanolang.h')
    for prefix in ('test-eval-callable-lifetimes', 'asan-test-eval-callable-lifetimes'):
        text = files['journal/'+prefix+'.stdout']
        assert b'I retain 0 callable names across nine 64-call routes.' in text
        assert len(re.findall(rb'0 outstanding after teardown\.', text)) == 9
        assert files['journal/'+prefix+'.stderr'] == b''
    for prefix in ('test-eval-callable-expression-ownership', 'asan-test-eval-callable-expression-ownership'):
        text = files['journal/'+prefix+'.stdout']
        assert text.count(b'I own an independent callable result from ') == 5
        for label, count in (('discarded', 320), ('replacement', 192), ('handled', 64), ('async_case', 192)):
            assert f'I observe {label}: {count} allocations and {count} releases.'.encode() in text
    for name, expected in (('expression-ownership', b'572 byte(s) leaked in 26 allocation(s)'), ('record-escape', b'3720 byte(s) leaked in 159 allocation(s)')):
        report = files['journal/asan-test-eval-callable-'+name+'.stderr']
        assert b'LeakSanitizer: detected memory leaks' in report and expected in report
        assert b'heap-use-after-free' not in report
    assert files['journal/primitive-asan-run.stderr'] == b''
    primitive = files['journal/primitive-only.c']
    original = files['source/tests/test_eval_callable_expression_ownership.c']
    assert primitive == original.replace(b'int main(void) {', b'int complete_callable_ownership_main(void) {') + b'\nint main(void) { source_temporaries(); return 0; }\n'
    assert b's_metadata_allocated - before' in files['journal/old-providers-run.stdout']
    assert b'FAILED:' in files['journal/old-providers-run.stdout']
    for prefix in ('test-eval-callable-record-escape', 'asan-test-eval-callable-record-escape'):
        assert b'run every fixture shadow.' in files['journal/'+prefix+'.stdout']
    assert b'All eval tests passed!' in files['journal/final-eval-run.stdout']
    assert b'All typechecker tests passed!' in files['journal/final-typechecker-run.stdout']
    assert json.loads(files['journal/repl-original.json'])['exit'] == 1
    assert json.loads(files['journal/phase1/ordinary-result.json'])[-1]['exit'] == 2
    for prefix in ('repl-original', 'repl-make'):
        report = files['journal/'+prefix+'.stderr']
        assert b'parent' in report and b'g_mod_count' in report and b'cmd_modules' in report
    assert len(q['provider_identities']) > 200
    assert all(raw[:6] == b'\x7fELF\x02\x01' for name, raw in files.items() if name.startswith('products/'))
    print(f"PASS: {len(files)} records/{len(paths)} Git inputs; original strict audit zero normal/ASan; complete LSan and REPL failures retained.")


if __name__ == '__main__':
    main()
