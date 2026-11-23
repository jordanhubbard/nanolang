#!/usr/bin/env python3
"""I verify integrated capture evidence without granting a new bootstrap or release."""
from pathlib import Path
import ast
import gzip
import hashlib
import io
import json
import struct
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    files, decoded, descriptors = {}, {}, {}
    for base in manifest['base_archives']:
        assert sha((HERE / base['path']).read_bytes()) == base['sha256']
    for row in manifest['files']:
        assert row['path'] not in files
        descriptor = {key: row[key] for key in ('bytes', 'artifact', 'compressed_bytes', 'compressed_sha256')}
        if row['sha256'] in descriptors:
            assert descriptor == descriptors[row['sha256']]
        else:
            descriptors[row['sha256']] = descriptor
            path = HERE / row['artifact']
            assert path.resolve().is_relative_to((ROOT / 'docs/evidence').resolve())
            encoded = path.read_bytes()
            assert len(encoded) == row['compressed_bytes'] and sha(encoded) == row['compressed_sha256']
            raw = gzip.decompress(encoded)
            assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
            decoded[row['sha256']] = raw
        files[row['path']] = decoded[row['sha256']]
    assert len(files) == manifest['counts']['records']

    def observe(name):
        return json.loads(files['observer/' + name])

    before, after = observe('before.json'), observe('after.json')
    assert after['guards'] == before
    assert observe('post-source-tools.json')['unchanged']
    controls = observe('controls.json')
    assert controls == dict(discovered=38, ran=38, errors=0, failures=0, skipped=[])
    report = observe('source.json')
    assert report['discovered'] == report['ran'] == 128
    assert report['errors'] == report['failures'] == 0 and len(report['skipped']) == 24
    matrix = observe('response-matrix.json')
    assert len(matrix) == 600
    for row in matrix:
        assert row['captured_returncode'] == 0 and row['native_returncode'] == row['replay_returncode']
        assert row['native_stdout'] == row['replay_stdout'] and row['native_stderr'] == row['replay_stderr']
    steps = observe('steps.json')
    assert [row['label'] for row in steps] == ['controls', 'build', 'gnu-queries', 'response', 'source']
    assert all(row['returncode'] == 0 for row in steps)
    assert steps[-1]['timeout'] == 1500 and steps[-2]['timeout'] == 600
    for row in steps:
        for suffix in ('stdout', 'stderr'):
            assert sha(files['observer/' + row['label'] + '.' + suffix]) == row[suffix + '_sha256']
    assert b'Ran 3 tests' in files['observer/gnu-queries.stderr']
    assert observe('producer-integrity.json')['records'] == 1449
    assert [row['records'] for row in observe('additional-peer-integrity.json')] == [1476, 1838, 3638]
    cold = observe('cold-result.json')
    assert cold['tools_unchanged'] and cold['observation']['returncode'] == 0
    assert cold['observation']['supported_version_decision'] == 0
    fresh = observe('fresh-result.json')
    assert fresh['pin'] == manifest['source_pin']
    assert fresh['ordinary_shadow_override'] is None and fresh['guide_snippets'] == 40
    assert fresh['compiler_link'] == 'nanoc_c' and fresh['bootstrap_state_absent'] and fresh['git_status_clean']
    assert fresh['selection']['settings_unchanged'] and not fresh['selection']['environment_changed']
    guide = observe('fresh-steps.json')[-1]
    assert guide['label'] == 'fresh-guide' and guide['returncode'] == 0 and guide['timeout'] == 1800
    assert b'userguide-check: 40 snippet(s) passed' in files['observer/fresh-guide.stdout']
    image = files['fresh/bin/nanoc_c']
    assert image[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H', image, 18)[0] == 183
    assert sha(image) == fresh['compiler_sha256']
    assert sha(files['source/src/module_builder.c']) == '971b242622730d2581bd5d576bec8dc7ce48a05c67e4917765d4f347f346bb09'
    assert sha(files['selected/tests/test_source_snapshots.py']) == '7e80518ce26a983ec316fe161cbbf7c5a77a64724554ba3dc592f690ab7ee259'
    expected = dict(before['sources'])
    expected.update(before['selected'])
    expected.update(before['protected'])
    expected['Makefile.gnu'] = before['make']
    for name, value in before['sources'].items():
        assert sha(files['source/' + name]) == value
    tracked = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', manifest['source_pin']], cwd=ROOT, text=True).splitlines())
    archive = subprocess.check_output(['git', 'archive', manifest['source_pin'], '--', *sorted(set(expected) & tracked)], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        actual = {entry.name: sha(source.extractfile(entry).read()) for entry in source if entry.isfile()}
    assert actual == {name: value for name, value in expected.items() if name in tracked}
    assert set(expected) - tracked == {'modules/index.json'}
    for row in observe('fixture-ast-comparison.json'):
        assert row['unchanged']
    baseline = observe('selection.json')['source_before']
    for path in ('tests/test_source_snapshots.py', 'tests/test_response_driver_grammar.py'):
        old = ast.parse(subprocess.check_output(['git', 'show', baseline + ':' + path], cwd=ROOT, text=True))
        new = ast.parse(files['selected/' + path].decode())
        def assertions(tree):
            return sorted(ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                          if isinstance(node, ast.Assert) or isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute)
                          and (node.func.attr.startswith('assert') or node.func.attr == 'skipTest'))
        assert assertions(old) == assertions(new)
    hosted = False
    supplement = HERE / 'verification/hosted-manifest.json'
    if supplement.exists():
        declaration = json.loads(supplement.read_text())
        artifacts = {}
        for row in declaration['files']:
            encoded = (HERE / row['artifact']).read_bytes()
            assert len(encoded) == row['compressed_bytes'] and sha(encoded) == row['compressed_sha256']
            raw = gzip.decompress(encoded)
            assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
            artifacts[row['path']] = raw
        run = json.loads(artifacts['hosted-run.json'])
        jobs = json.loads(artifacts['hosted-jobs.json'])['jobs']
        summary = json.loads(artifacts['hosted-audit.json'])
        selection = json.loads(artifacts['artifact/.test_output/capture-assembler-ci/result.json'])
        assert run['conclusion'] == 'success' and run['head_sha'] == manifest['source_pin']
        assert summary['actual_merge'] == '64a604f41409af72b2a9b2d88bee89a77d0f5e0d'
        assert summary['actual_merge'].encode() in artifacts['hosted-guide.log']
        assert b'userguide-check: 40 snippet(s) passed' in artifacts['hosted-guide.log']
        build = next(job for job in jobs if job['name'] == 'build')
        assert build['conclusion'] == 'success'
        for label in ('I select my verified native capture assembler', 'Validate user guide snippets', 'Build user guide HTML'):
            assert next(step for step in build['steps'] if step['name'] == label)['conclusion'] == 'success'
        assert next(job for job in jobs if job['name'] == 'deploy')['conclusion'] == 'skipped'
        assert selection['compiler_before'] == selection['compiler_after'] and selection['settings_unchanged']
        assert selection['compiler_before']['sha256'] == '864711037b900acdb7a449cf48d912cefa935b527fb60c767ccb956779e66136'
        assert selection['capture_source_sha256'] == sha(files['source/src/module_builder.c'])
        assert selection['profile_origin']['producer_run'] == 45009129801
        assert selection['selected_assembler']['sha256'] == '7e9edb4b7fc0c7ce97fbcdd4ecb63e3eb1c1663e2452cd32ce0249110b253bf5'
        assert sha(artifacts['artifact/.test_output/capture-assembler-ci/selected-as']) == selection['selected_assembler']['sha256']
        assert sha(artifacts['artifact/obj/test_module_generation_probe']) == selection['probe_sha256']
        for name, expected in summary['source_blobs'].items():
            assert subprocess.check_output(['git', 'rev-parse', manifest['source_pin'] + ':' + name], cwd=ROOT, text=True).strip() == expected
        hosted = True
    print(json.dumps(dict(status='INTEGRITY_PASS', source_pin=manifest['source_pin'], records=len(files),
                          controls=38, gnu_queries=3, source_methods=128, source_conditional_skips=24,
                          response_equivalences=600, ordinary_guide_snippets=40, shadow_override=None,
                          hosted_completion=hosted, later_source_bootstrap=False, current_darwin=False,
                          release_acceptance=False), indent=2))


if __name__ == '__main__':
    main()
