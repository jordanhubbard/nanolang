#!/usr/bin/env python3
"""I reuse a completed fixed-GCC producer for one explicit Clang consumer."""
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ci_qualify_as246 as qualifier
from scripts import ci_qualify_fixed_userguide as guide
from scripts.ci_capture_assembler_profile import read_fixed_profile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_prior(prior, run, machine=None):
    candidate = prior / 'as246-candidate'
    result, cold = guide.prior_reports(candidate)
    if (run.get('status') != 'completed' or run.get('conclusion') != 'success' or
            run.get('head_sha') != result['source_pin'] or
            run.get('path') != '.github/workflows/source-capture-toolchain-diagnostic.yml'):
        raise RuntimeError('I require the successful matching source-capture producer run.')
    if (not result.get('original_userguide_passed') or
            result.get('ordinary_userguide_environment', {}).get('shadow_override', 'missing') is not None):
        raise RuntimeError('I require the preceding ordinary default guide acceptance.')
    if 'userguide-check: 40 snippet(s) passed' not in (candidate / 'original-userguide.stdout').read_text():
        raise RuntimeError('I require the actual forty-snippet ordinary guide terminal.')
    fixed = prior / 'fixed-as246'
    build = read_fixed_profile(fixed, machine)
    if (build['assembler_sha256'] != result['fixed_backend']['assembler_sha256'] or
            build['assembler_sha256'] != guide.retained_tool_digest(cold, cold['assembler']) or
            build['compiler'] != cold['compiler'] or
            build['compiler_sha256'] != guide.retained_tool_digest(cold, cold['compiler'])):
        raise RuntimeError('I require matching producer and preceding consumer native identities.')
    for name, key in (('test_module_generation_probe', 'obj/test_module_generation_probe'),
                      ('nano_as_capture.so', 'bin/nano_as_capture.so')):
        if digest(candidate / name) != guide.retained_tool_digest(cold, key):
            raise RuntimeError('I refuse a changed preceding native probe or capture helper.')
    if digest(candidate / 'candidate.patch') != result['candidate_patch_sha256']:
        raise RuntimeError('I refuse a changed qualified consumer patch.')
    return {'producer_run': run['id'], 'source_pin': result['source_pin'],
        'assembler_sha256': build['assembler_sha256'], 'native_machine': machine or platform.machine(),
        'producer_compiler_sha256': build['compiler_sha256'],
        'preceding_probe_sha256': digest(candidate / 'test_module_generation_probe'),
        'preceding_helper_sha256': digest(candidate / 'nano_as_capture.so'),
        'source_sha256': build['source_sha256'], 'fixed_source_sha256': build['fixed_source_sha256'],
        'gas_cases': build['gas_cases'], 'witnesses': len(build['witnesses'])}


def require_candidate(prior, source, test, patch):
    for path in (source, test):
        if path.read_bytes() != (prior / path.name).read_bytes():
            raise RuntimeError('I require byte-identical previously qualified consumer source and tests.')
    if patch != (prior / 'candidate.patch').read_bytes():
        raise RuntimeError('I require the exact previously qualified consumer patch.')
    return {'source_sha256': digest(source), 'test_sha256': digest(test),
            'patch_sha256': hashlib.sha256(patch).hexdigest()}


def native_closure(record, root, out, label):
    assembler = Path(record['fixed_backend']['assembler'])
    images = [str(assembler), record['selected_clang']['path'],
              str(root / 'obj/test_module_generation_probe'), str(root / 'bin/nano_as_capture.so')]
    compiler = record['fixed_backend'].get('compiler')
    if compiler and compiler not in images:
        images.append(compiler)
    dependencies = []
    for index, image in enumerate(images):
        result = qualifier.command(['ldd', image],
            'fixed-clang-dependencies-' + label + '-' + str(index), 30, output=out)
        text = result.stdout.decode()
        paths = re.findall(r'^\s*(?:\S+\s+=>\s+)?(/\S+)\s', text, re.MULTILINE)
        if 'not found' in text or not paths:
            raise RuntimeError('I require the actual native tool dependency closure.')
        dependencies.extend(paths)
    return {str(Path(p).resolve(strict=True)): digest(p) for p in [*images, *dependencies]}


def require_source_continuity(pin, out):
    qualifier.command(['git', 'fetch', '--no-tags', '--depth=1', 'origin', pin],
                      'producer-source-fetch', 120, output=out)
    changed = qualifier.command(['git', 'diff', '--name-only', '-z', pin, 'HEAD'],
                                'producer-source-delta', 30, output=out).stdout
    names = [os.fsdecode(p) for p in changed.split(b'\0') if p]
    permitted = {'docs/ROADMAP.md', 'Makefile.gnu', '.github/workflows/userguide_pages.yml',
        '.github/workflows/source-capture-toolchain-diagnostic.yml', 'scripts/ci_qualify_as246.py',
        'scripts/ci_fixed_clang_inputs.py', 'tests/test_capture_qualification_modes.py',
        'tests/test_fixed_clang_inputs.py', 'scripts/ci_profile_capture_pair.py',
        'scripts/ci_capture_assembler_profile.py', 'scripts/ci_select_capture_assembler.py',
        'scripts/ci_userguide_capture_diagnostics.py', 'scripts/ci_fixed_gcc_inputs.py',
        '.github/actions/setup-capture-assembler/action.yml',
        'tests/retained_capture_evidence.py', 'tests/test_capture_assembler_profile.py',
        'tests/test_ci_select_capture_assembler.py', 'tests/assembler_search_fixture.py',
        'tests/test_assembler_search_fixture.py', 'tests/test_fixed_gcc_inputs.py',
        'tests/test_source_snapshots.py'}
    if any(name not in permitted and not name.startswith('docs/evidence/') for name in names):
        raise RuntimeError('I refuse unrelated source drift after the qualified producer.')
    if 'tests/test_source_snapshots.py' in names:
        from tests.assembler_search_fixture import upgrade_source
        previous_fixture = qualifier.command(['git', 'show', pin + ':tests/test_source_snapshots.py'],
            'producer-source-fixture', 30, output=out).stdout.decode()
        if (ROOT / 'tests/test_source_snapshots.py').read_text() != upgrade_source(previous_fixture):
            raise RuntimeError('I refuse source-fixture drift beyond the declared package protocol replacement.')
    for name in ('Makefile.gnu', '.github/workflows/userguide_pages.yml'):
        if name not in names:
            continue
        previous = qualifier.command(['git', 'show', pin + ':' + name],
            'producer-' + Path(name).name, 30, output=out).stdout.decode()
        current = (ROOT / name).read_text()
        old, new = ('tests.test_response_driver_grammar\n',
                    'tests.test_response_driver_grammar tests.test_gnu_tool_queries\n') if name == 'Makefile.gnu' else (
            '  group: "userguide-pages"\n',
            "  group: userguide-pages-${{ github.event_name == 'pull_request' && github.event.pull_request.number || 'deploy' }}\n")
        if previous.count(old) != 1 or current != previous.replace(old, new):
            raise RuntimeError('I refuse a build or guide change beyond the reviewed handoff.')
    previous = qualifier.command(['git', 'show', pin + ':scripts/ci_qualify_as246.py'],
        'producer-command-helper', 30, output=out).stdout.decode()
    def command_body(source):
        return ast.dump(next(node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == 'command'), include_attributes=False)
    if command_body(previous) != command_body((ROOT / 'scripts/ci_qualify_as246.py').read_text()):
        raise RuntimeError('I refuse a changed producer process-supervision helper.')
    return names


def prepare_prior(out, record):
    record['free_workspace_bytes'] = shutil.disk_usage(ROOT).free
    record['free_temporary_bytes'] = shutil.disk_usage(tempfile.gettempdir()).free
    if min(record['free_workspace_bytes'], record['free_temporary_bytes']) < 5 * 1024**3:
        raise RuntimeError('I require five GiB of consumer build headroom.')
    qualifier.command(['git', 'diff', '--exit-code', 'HEAD', '--'],
                      'initial-clean-tree', 30, output=out)
    prior = Path(os.environ['NANO_PRIOR_CAPTURE_DIRECTORY']).resolve(strict=True)
    run_id = os.environ['NANO_PRIOR_CAPTURE_RUN']
    if not re.fullmatch('[1-9][0-9]*', run_id):
        raise RuntimeError('I require a numeric completed producer run.')
    run = json.loads(qualifier.command(['gh', 'api', 'repos/' + os.environ['GITHUB_REPOSITORY'] +
        '/actions/runs/' + run_id], 'producer-run', 30, output=out).stdout)
    record['producer'] = require_prior(prior, run)
    record['source_changes'] = require_source_continuity(record['producer']['source_pin'], out)
    return prior


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('I prepare this consumer only in its isolated GitHub Actions checkout.')
    out = ROOT / '.test_output/source-capture-diagnostic/fixed-clang-inputs'
    out.mkdir(parents=True, exist_ok=False)
    record = {'scope': 'I prepare only an explicit Clang/r1 consumer. I repeat no producer '
                      'build/GAS cohort and change no default CI compiler or production admission.'}
    try:
        prior = prepare_prior(out, record)
        clang = Path(shutil.which('clang')).resolve(strict=True)
        version = qualifier.command([str(clang), '--version'], 'actual-clang-version', 30, output=out)
        if not version.stdout.startswith(b'Ubuntu clang version 21.1.8 (6ubuntu1)\n'):
            raise RuntimeError('I require the actual qualified Clang21 image.')
        tools = out / 'tools'
        tools.mkdir()
        for name in ('cc', 'clang'):
            (tools / name).symlink_to(clang)
        with open(os.environ['GITHUB_PATH'], 'a') as stream:
            stream.write(str(tools) + '\n')
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            stream.write('NANO_FIXED_AS246_DIRECTORY=' + str(prior / 'fixed-as246') + '\n')
            stream.write('NANO_FIXED_CLANG_PRIOR=' + str(prior / 'as246-candidate') + '\n')
        record['clang_sha256'] = digest(clang)
        record['prepared'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
