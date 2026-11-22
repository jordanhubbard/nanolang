#!/usr/bin/env python3
"""I qualify current production capture with the actual declared Clang compiler."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ci_qualify_as246 as qualifier
from scripts import ci_source_capture_linux as cold_capture
from scripts.ci_fixed_clang_inputs import native_closure
from scripts.ci_capture_assembler_profile import retained_capture_bytes

OUT = ROOT / '.test_output/capture-compiler'
EXPECTED = {
    'src/module_builder.c': '971b242622730d2581bd5d576bec8dc7ce48a05c67e4917765d4f347f346bb09',
    'tests/test_source_snapshots.py': '7e80518ce26a983ec316fe161cbbf7c5a77a64724554ba3dc592f690ab7ee259',
    'src/nanolang.h': 'b31f07ee4664a8bea9c19da5fba905aff4ba7daad18165647fd91b92fc826aa6',
    'obj/test_module_generation_probe': 'c61fac6137641dfd5c0e3f2e6dd8ce0b098220e0b86b7e99274051f5533c5070',
    'bin/nano_as_capture.so': '678c89bbd9a0b9cdfc01be19f281726139fef337303e5bb7639b4b1061a0ce0e',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def identity():
    return {name: digest(ROOT / name) for name in EXPECTED}


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    selected = json.loads((ROOT / '.test_output/capture-assembler-ci/result.json').read_text())
    profile = json.loads((ROOT / '.test_output/capture-assembler-ci/validated-profile.json').read_text())
    compiler = selected['compiler_before']
    if compiler['configured'] != 'clang' or compiler['sha256'] != '78e9dd2ccc9e88859b3a29109a44bfab00e3d6cb70f00f3daae063eed3d05251':
        raise RuntimeError('I require the actual previously qualified Clang21 image.')
    if identity() != EXPECTED:
        save('unmatched-native-identities.json', identity())
        raise RuntimeError('I require my exact retained current source, fixture, header, probe and helper bytes.')
    producers = []
    for index, path in enumerate(sorted((ROOT / 'obj').rglob('*.o'))):
        result = qualifier.command(['readelf', '-p', '.comment', path],
            'object-producer-' + str(index), 30, output=OUT)
        if b'clang version' not in result.stdout.lower() or b'GCC:' in result.stdout:
            raise RuntimeError('I require every freshly built project object to identify Clang.')
        producers.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    if not producers:
        raise RuntimeError('I require actual project object producer observations.')
    record = {'source_pin': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'selected_clang': compiler,
        'fixed_backend': {'assembler': selected['selected_assembler']['path'], 'compiler': profile['compiler']},
        'objects': producers, 'native_identities_before': identity()}
    record['native_closure_before'] = native_closure(record, ROOT, OUT, 'before')
    names = subprocess.check_output(['git', 'ls-files', '-z', 'src', 'src_nano', 'tests', 'scripts',
        'modules', 'std', 'stdlib', '.github', 'examples', 'userguide', 'GNUmakefile', 'Makefile.gnu'], cwd=ROOT).split(b'\0')
    inputs = {os.fsdecode(name): digest(ROOT / os.fsdecode(name)) for name in names if name}
    record['source_inputs'] = inputs
    snapshots = OUT / 'source-inputs'
    snapshots.mkdir()
    for name, expected in inputs.items():
        target = snapshots / (expected + '.gz')
        if not target.exists():
            target.write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
    save('before.json', record)
    native = OUT / 'native'
    native.mkdir()
    for name in ('obj/test_module_generation_probe', 'bin/nano_as_capture.so'):
        shutil.copy2(ROOT / name, native / Path(name).name)
    tools = Path(os.environ['RUNNER_TEMP']) / 'nanolang-capture-clang-driver'
    tools.mkdir(exist_ok=False)
    (tools / 'cc').symlink_to(compiler['path'])
    with open(os.environ['GITHUB_PATH'], 'a') as stream:
        stream.write(str(tools) + '\n')


def phase(name):
    record = json.loads((OUT / 'before.json').read_text())
    if Path(shutil.which('cc')).resolve() != Path(record['selected_clang']['path']):
        raise RuntimeError('I require the explicit diagnostic cc alias to select the same actual Clang image.')
    qualifier.OUT = OUT
    if name == 'source':
        qualifier.command([sys.executable, '-c', qualifier.original_source_runner(), OUT / 'source.json'],
                          'original-tests', 1500)
        observed = json.loads((OUT / 'source.json').read_text())
        previous = json.loads(retained_capture_bytes('observer/hosted-clang-45009127688/as246-candidate/tests.json'))
        if observed['skipped'] != previous['skipped']:
            raise RuntimeError('I preserve the original complete Clang conditional skip tuples.')
    elif name == 'response':
        runner = qualifier.original_source_runner().replace('tests.test_source_snapshots',
            'tests.test_response_driver_grammar').replace('discovered==128 and result.testsRun==128',
            'discovered==1 and result.testsRun==1')
        qualifier.command([sys.executable, '-c', runner, OUT / 'response.json'], 'response-matrix', 600)
    elif name == 'debug':
        save('debug.json', qualifier.clang_native_comparisons(record['selected_clang']['path']))
    elif name == 'cold':
        cold_capture.OUT = OUT / 'cold-compiler-support'
        cold_capture.main()
        observed = json.loads((cold_capture.OUT / 'observation.json').read_text())
        if observed['returncode'] != 0 or observed['compiler'] != record['selected_clang']['path']:
            raise RuntimeError('I require actual selected-Clang cold CompilerSupport publication.')
    elif name == 'guide':
        qualifier.original_userguide(record)
        if record['ordinary_userguide_environment']['inherited_shadow_override'] is not None:
            raise RuntimeError('I require the original ordinary guide environment without an inherited override.')
        save('guide.json', record)
    else:
        raise ValueError('I require a supported current production capture phase.')


def finish():
    record = json.loads((OUT / 'before.json').read_text())
    qualifier.OUT = OUT
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    record['native_closure_after'] = native_closure(record, ROOT, OUT, 'after')
    record['all_source_hashes_unchanged'] = all(digest(ROOT / p) == h for p, h in record['source_inputs'].items())
    record['native_identities_after'] = identity()
    reports = {}
    for name in ('source', 'response', 'debug', 'guide'):
        path = OUT / (name + '.json')
        reports[name] = json.loads(path.read_text()) if path.is_file() else None
    record['reports'] = reports
    record['errors'] = sorted(p.name for p in OUT.glob('error-*.json'))
    matrix = ROOT / '.test_output/response-driver-grammar-matrix.json'
    if matrix.is_file():
        shutil.copy2(matrix, OUT / matrix.name)
    observations = json.loads(matrix.read_text()) if matrix.is_file() else []
    record['response_cases'] = len(observations)
    record['response_returncode_stdout_matches'] = all(
        item['captured_returncode'] == 0 and
        (item['native_returncode'], item['native_stdout']) ==
        (item['replay_returncode'], item['replay_stdout']) for item in observations)
    cold_path = OUT / 'cold-compiler-support/observation.json'
    cold = json.loads(cold_path.read_text()) if cold_path.is_file() else {}
    record['cold_observation'] = cold
    guide_stdout = OUT / 'original-userguide.stdout'
    guide_passed = (guide_stdout.is_file() and
        b'userguide-check: 40 snippet(s) passed' in guide_stdout.read_bytes())
    source, response = reports['source'], reports['response']
    success = (not record['errors'] and source is not None and response is not None and
        source['ran'] == source['discovered'] == 128 and source['errors'] == source['failures'] == 0 and
        response['ran'] == response['discovered'] == 1 and response['errors'] == response['failures'] == 0 and
        response['skipped'] == [] and reports['debug'] is not None and len(reports['debug']) == 9 and
        reports['guide'] is not None and reports['guide']['original_userguide_passed'] and
        guide_passed and cold.get('returncode') == 0 and
        cold.get('compiler') == record['selected_clang']['path'] and
        len(observations) == 600 and record['response_returncode_stdout_matches'] and
        record['all_source_hashes_unchanged'] and record['native_identities_after'] == EXPECTED and
        record['native_closure_before'] == record['native_closure_after'])
    record['success'] = bool(success)
    record['scope'] = 'I qualify the actual current production Clang capture profile. Full compiler stages/platform/release gates remain separate.'
    save('result.json', record)
    if not success:
        raise RuntimeError('I retain the incomplete or failed current production capture qualification.')


def main():
    if len(sys.argv) != 2:
        raise ValueError('I require one explicit current-capture phase.')
    name = sys.argv[1]
    try:
        if name == 'prepare':
            prepare()
        elif name == 'finish':
            finish()
        else:
            phase(name)
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save('error-' + name + '.json', {'phase': name, 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
