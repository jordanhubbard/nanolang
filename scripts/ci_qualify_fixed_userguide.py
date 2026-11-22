#!/usr/bin/env python3
"""I run the missing ordinary guide gate after matching completed private capture."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ci_qualify_as246 as qualifier
from scripts import ci_source_capture_linux as capture


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior_reports(prior):
    result = json.loads((prior / 'result.json').read_text())
    reports = {}
    for name, count in (('tests.json', 128), ('response-tests.json', 1)):
        report = json.loads((prior / name).read_text())
        if (report.get('discovered') != count or report.get('ran') != count or
                report.get('errors') != 0 or report.get('failures') != 0):
            raise RuntimeError('I require both complete passing original capture reports.')
        reports[name] = report
    if (not result.get('fixed_as246_complete') or
            not result.get('fixed_identified_profile_qualification_passed') or
            not result.get('checkout_sources_restored') or
            result.get('complete_corpus') != {'source': reports['tests.json'],
                'response': reports['response-tests.json'], 'separate_invocations': True,
                'total_methods': 129}):
        raise RuntimeError('I require the completed matching private fixed capture cohort.')
    cold = json.loads((prior / 'cold-compiler-support/observation.json').read_text())
    if cold.get('returncode') != 0 or cold.get('supported_version_decision') != 0:
        raise RuntimeError('I require the actual preceding cold publication.')
    return result, cold


def require_same_source(pin):
    if not re.fullmatch('[0-9a-f]{40}', pin):
        raise RuntimeError('I require an exact retained source commit.')
    result = qualifier.command(['git', 'diff', '--name-only', '-z', pin, 'HEAD', '--'],
                               'preceding-source-delta', 30)
    changed = [os.fsdecode(p) for p in result.stdout.split(b'\0') if p]
    permitted = {'docs/ROADMAP.md', '.github/workflows/source-capture-toolchain-diagnostic.yml',
        'scripts/ci_qualify_as246.py', 'scripts/ci_qualify_fixed_userguide.py',
        'tests/test_capture_qualification_modes.py', 'tests/test_fixed_userguide_qualification.py'}
    if any(name not in permitted and not name.startswith('docs/evidence/') for name in changed):
        raise RuntimeError('I refuse production source drift after the qualified capture cohort.')
    return changed


def main():
    out = ROOT / '.test_output/source-capture-diagnostic/ordinary-userguide'
    out.mkdir(parents=True, exist_ok=False)
    qualifier.OUT = out
    original = {ROOT / name: (ROOT / name).read_bytes() for name in
                ('src/module_builder.c', 'tests/test_source_snapshots.py')}
    record = {'scope': 'I execute only the missing ordinary user-guide cohort. I reuse no '
                      'earlier hosted shadow allowance as ten-second default acceptance.'}
    try:
        qualifier.command(['git', 'diff', '--exit-code', 'HEAD', '--'], 'initial-clean-tree', 30)
        prior = Path(os.environ['NANO_PRIOR_CAPTURE_DIRECTORY']).resolve(strict=True) / 'as246-candidate'
        previous, cold = prior_reports(prior)
        record['prior_source_pin'] = previous['source_pin']
        record['diagnostic_only_source_changes'] = require_same_source(previous['source_pin'])
        record['prior_reports_sha256'] = {name: digest(prior / name) for name in
            ('result.json', 'tests.json', 'response-tests.json', 'cold-compiler-support/observation.json')}
        current = json.loads((capture.OUT / 'observation.json').read_text())
        source, test = original
        _, _, backend = qualifier.fixed_assembler_candidate(
            original[source].decode(), original[test].decode(), current)
        compiler = backend['compiler']
        if (compiler != cold['compiler'] or digest(compiler) != cold['tool_sha256'][compiler] or
                backend['assembler_sha256'] != previous['fixed_backend']['assembler_sha256'] or
                backend['assembler_sha256'] != cold['tool_sha256'][cold['assembler']]):
            raise RuntimeError('I require the same actual compiler and qualified assembler images.')
        record['fixed_backend'] = backend
        patch = prior / 'candidate.patch'
        if digest(patch) != previous['candidate_patch_sha256']:
            raise RuntimeError('I refuse a changed qualified private source patch.')
        summary = qualifier.command(['git', 'apply', '--numstat', '-z', str(patch)],
                                    'candidate-patch-paths', 30).stdout
        paths = {os.fsdecode(row.split(b'\t', 2)[2]) for row in summary.split(b'\0') if row}
        if paths != {'src/module_builder.c', 'tests/test_source_snapshots.py'}:
            raise RuntimeError('I refuse a private patch outside the qualified source/fixture pair.')
        qualifier.command(['git', 'apply', '--check', str(patch)], 'candidate-patch-check', 30)
        qualifier.command(['git', 'apply', str(patch)], 'candidate-patch-apply', 30)
        for path in original:
            if path.read_bytes() != (prior / path.name).read_bytes():
                raise RuntimeError('I require the exact previously qualified private source bytes.')
            shutil.copy2(path, out / path.name)
        shutil.copy2(patch, out / 'candidate.patch')
        qualifier.command(['make', '-f', 'Makefile.gnu', '-j2',
            'obj/test_module_generation_probe', 'bin/nano_as_capture.so'], 'build', 600)
        for name in ('obj/test_module_generation_probe', 'bin/nano_as_capture.so'):
            shutil.copy2(ROOT / name, out / Path(name).name)
        qualifier.original_userguide(record)
        record['ordinary_default_userguide_passed'] = True
        record['preceding_capture_cohorts_repeated'] = False
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        for path, raw in original.items():
            path.write_bytes(raw)
        record['checkout_sources_restored'] = all(p.read_bytes() == raw for p, raw in original.items())
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1:]:
        raise SystemExit('I require the declared retained capture environment.')
    main()
