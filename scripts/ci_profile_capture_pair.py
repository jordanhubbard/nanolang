#!/usr/bin/env python3
"""I measure original native assertions across retained tool axes."""
import hashlib
import ast
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ci_qualify_as246 as process
from scripts.ci_capture_assembler_profile import read_fixed_profile
from scripts.ci_fixed_clang_inputs import native_closure
from tests.native_process import run_captured

METHOD = 'tests.test_source_snapshots.SourceSnapshots.test_clang_external_assembler_cache_restored_inputs'
CHILD = r'''
import hashlib,json,os,resource,shutil,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
config=json.loads(Path(sys.argv[1]).read_text());root=Path(config['root']);out=Path(config['output'])
sys.path.insert(0,str(root));os.chdir(root)
native_run=subprocess.run;records=[];fixtures=[]
def cpu():
 r=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime
def observe(argv,**kwargs):
 start=time.monotonic();before=cpu()
 item={'argv':list(map(str,argv)) if not isinstance(argv,str) else argv,
  'parent_timeout':kwargs.get('timeout'),'started_monotonic':start}
 def persist(event):
  with (out/'commands.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
 persist({'event':'start',**item})
 try:
  result=native_run(argv,**kwargs)
 except BaseException as error:
  item.update(error=repr(error),wall_seconds=time.monotonic()-start,child_cpu_seconds=cpu()-before)
  records.append(item);persist({'event':'exception',**item});raise
 item.update(wall_seconds=time.monotonic()-start,child_cpu_seconds=cpu()-before,
  returncode=result.returncode,stdout=result.stdout.decode(errors='replace') if isinstance(result.stdout,bytes) else result.stdout,
  stderr=result.stderr.decode(errors='replace') if isinstance(result.stderr,bytes) else result.stderr)
 records.append(item);persist({'event':'complete',**item})
 return result

def retain(self):
 self._finalizer.detach();fixtures.append(self.name)
suite=unittest.defaultTestLoader.loadTestsFromName(config['method']);discovered=suite.countTestCases()
start=time.monotonic();before=cpu()
with (out/'unittest.log').open('w') as log,patch('subprocess.run',side_effect=observe),patch.object(tempfile.TemporaryDirectory,'cleanup',retain):
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
elapsed=time.monotonic()-start;used=cpu()-before
for name in fixtures:
 path=Path(name)
 if path.is_dir():shutil.copytree(path,out/'fixtures'/path.name,symlinks=True)
(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
report={'discovered':discovered,'ran':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
 'skipped':[(case.id(),reason) for case,reason in result.skipped], 'wall_seconds':elapsed,'child_cpu_seconds':used,
 'original_method':config['method'],'imported_source':str(sys.modules['tests.test_source_snapshots'].__file__),
 'fixture_paths':fixtures,'scope':'I preserve original assertions and subprocess deadlines. Cleanup-only observation retains original temporary fixture paths.'}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(0 if discovered==result.testsRun==1 and result.wasSuccessful() and not result.skipped else 1)
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu():
    value = resource.getrusage(resource.RUSAGE_CHILDREN)
    return value.ru_utime + value.ru_stime


def image(path, expected):
    if digest(path) != expected:
        raise RuntimeError('I refuse a changed retained native observation image.')
    return path


def input_digests(root):
    return {str(path.relative_to(root)): digest(path) for directory in ('src', 'tests', 'obj', 'bin')
            for path in sorted((root / directory).rglob('*')) if path.is_file()}


def scheduling(out, label):
    command = ['/bin/sh', '-c', 'i=0; while [ "$i" -lt 1000000 ]; do i=$((i+1)); done; printf ready']
    start, before = time.monotonic(), cpu()
    result = run_captured(command, timeout=15)
    elapsed, used = time.monotonic() - start, cpu() - before
    if result.returncode or result.stdout != b'ready':
        raise RuntimeError('I require my bounded CPU-share control to complete.')
    record = {'command': command, 'wall_seconds': elapsed, 'child_cpu_seconds': used,
              'cpu_wall_share': used / elapsed, 'scope': 'I observe one shell CPU loop, not a NanoVM or throughput guarantee.'}
    (out / ('scheduling-' + label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    if not sys.platform.startswith('linux') or platform.machine() != 'x86_64':
        raise RuntimeError('I require native Linux x86_64 for these retained images.')
    mode = os.environ.get('NANO_CAPTURE_PROFILE_METHOD', 'external-cache')
    methods = {'external-cache': METHOD,
        'split-search': 'tests.test_source_snapshots.SourceSnapshots.test_split_assembler_search_order_phases_and_recovery'}
    if mode not in methods:
        raise RuntimeError('I require a declared original diagnostic method.')
    method = methods[mode]
    out = ROOT / '.test_output/source-capture-diagnostic/capture-pair-profile'
    out.mkdir(parents=True, exist_ok=False)
    failed = ROOT / '.test_output/source-capture-diagnostic/profile-failed'
    supported = ROOT / '.test_output/source-capture-diagnostic/profile-supported'
    current, old = failed / 'as246-candidate', supported / 'as246-candidate'
    prior = failed / 'prior-fixed-clang/as246-candidate'
    record = {'scope': 'I measure one unchanged original method, not a complete qualification or admission.',
              'mode': mode, 'original_method': method, 'cases': []}
    try:
        if min(shutil.disk_usage(ROOT).free, shutil.disk_usage('/tmp').free) < 5 * 1024**3:
            raise RuntimeError('I require five GiB of profiling headroom.')
        previous = json.loads((current / 'result.json').read_text())
        if not previous.get('fixed_clang_complete') or not previous.get('checkout_sources_restored'):
            raise RuntimeError('I require the actual restored failed combined consumer.')
        clang = Path(shutil.which('clang')).resolve(strict=True)
        image(clang, previous['selected_clang']['sha256'])
        fixed = failed / 'prior-fixed-clang/fixed-as246'
        fixed_record = read_fixed_profile(fixed)
        supported_record = json.loads((supported / 'qualified-as242/build.json').read_text())
        if (not supported_record.get('completed') or supported_record.get('source_sha256') !=
                'f6e4d41fd5fc778b06b7891457b3620da5ecea1006c6a4a41ae998109f85a800'):
            raise RuntimeError('I require the actual supported native assembler producer.')
        assemblers = {'fixed': image(fixed / 'bin/as', fixed_record['assembler_sha256']),
            'supported': image(supported / 'qualified-as242/bin/as', supported_record['assembler_sha256'])}
        old_cold = json.loads((prior / 'cold-compiler-support/observation.json').read_text())
        probes = {'gcc': image(prior / 'test_module_generation_probe', old_cold['tool_sha256']['obj/test_module_generation_probe']),
            'clang': image(current / 'test_module_generation_probe', previous['native_closure_before'][
                '/home/runner/work/nanolang/nanolang/obj/test_module_generation_probe'])}
        helpers = {'gcc': image(prior / 'nano_as_capture.so', old_cold['tool_sha256']['bin/nano_as_capture.so']),
            'clang': image(current / 'nano_as_capture.so', previous['native_closure_before'][
                '/home/runner/work/nanolang/nanolang/bin/nano_as_capture.so'])}
        for name in ('module_builder.c', 'test_source_snapshots.py', 'candidate.patch'):
            if (current / name).read_bytes() != (prior / name).read_bytes():
                raise RuntimeError('I require the exact same qualified candidate source across compiler axes.')
        def original_method(directory):
            tree = ast.parse((directory / 'test_source_snapshots.py').read_text())
            return ast.dump(next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                and node.name == method.rsplit('.', 1)[1]), include_attributes=False)
        if original_method(current) != original_method(old):
            raise RuntimeError('I refuse different original assertions between retained controls.')
        tools = out / 'tools'
        tools.mkdir()
        for name in ('cc', 'clang'):
            (tools / name).symlink_to(clang)
        for tool in assemblers.values():
            tool.chmod(0o700)
        record['host'] = {'uname': list(os.uname()), 'cpu_count': os.cpu_count(),
            'affinity': sorted(os.sched_getaffinity(0)),
            'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                               if line.startswith('model name')), None),
            'image_os': os.environ.get('ImageOS'), 'image_version': os.environ.get('ImageVersion')}
        record['images'] = {name: {key: {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
            for key, path in values.items()} for name, values in
            (('probes', probes), ('helpers', helpers), ('assemblers', assemblers))}
        record['scheduling_before'] = scheduling(out, 'before')
        combinations = list(itertools.product(('gcc', 'clang'), ('gcc', 'clang'), ('supported', 'fixed')))
        reference = image(old / 'test_module_generation_probe', json.loads(
            (old / 'cold-compiler-support/observation.json').read_text())['tool_sha256']['obj/test_module_generation_probe'])
        combinations.insert(0, ('reference', 'gcc', 'supported'))
        if mode == 'split-search':
            combinations = [('reference', 'gcc', 'supported'), ('clang', 'clang', 'fixed')]
        all_images = [clang, *probes.values(), *helpers.values(), *assemblers.values(), reference]
        before_images = {str(path): digest(path) for path in all_images}
        started = time.monotonic()
        for probe_name, helper_name, backend in combinations:
            if time.monotonic() - started > 1200:
                raise RuntimeError('I retain my bounded partial diagnostic instead of extending it.')
            name = probe_name + '-' + helper_name + '-' + backend
            case = out / name
            root = case / 'root'
            results = case / 'observation'
            root.mkdir(parents=True)
            results.mkdir()
            for directory in ('src', 'tests'):
                shutil.copytree(ROOT / directory, root / directory,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            candidate = old if probe_name == 'reference' else current
            shutil.copy2(candidate / 'module_builder.c', root / 'src/module_builder.c')
            shutil.copy2(candidate / 'test_source_snapshots.py', root / 'tests/test_source_snapshots.py')
            (root / 'obj').mkdir()
            (root / 'bin').mkdir()
            shutil.copy2(reference if probe_name == 'reference' else probes[probe_name], root / 'obj/test_module_generation_probe')
            shutil.copy2(helpers[helper_name], root / 'bin/nano_as_capture.so')
            (root / 'obj/test_module_generation_probe').chmod(0o700)
            config = {'root': str(root), 'output': str(results), 'method': method}
            config_path = case / 'config.json'
            config_path.write_text(json.dumps(config, indent=2) + '\n')
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                       CC='cc', COMPILER_PATH=str(assemblers[backend].parent), PYTHONDONTWRITEBYTECODE='1')
            guarded = {'fixed_backend': {'assembler': str(assemblers[backend])},
                       'selected_clang': {'path': str(clang)}}
            source_before = input_digests(root)
            closure_before = native_closure(guarded, root, results, 'before')
            (results / 'inputs-before.json').write_text(json.dumps(source_before, indent=2) + '\n')
            try:
                process.command([sys.executable, '-c', CHILD, str(config_path)], name,
                                600 if mode == 'split-search' else 180,
                                cwd=root, env=env, output=results)
            except Exception as error:
                record['cases'].append({'name': name, 'error': repr(error)})
            else:
                observed = json.loads((results / 'result.json').read_text())
                if Path(observed['imported_source']).resolve() != root / 'tests/test_source_snapshots.py':
                    raise RuntimeError('I refuse an observation importing another source fixture.')
                source_after = input_digests(root)
                closure_after = native_closure(guarded, root, results, 'after')
                (results / 'inputs-after.json').write_text(json.dumps(source_after, indent=2) + '\n')
                observed['source_inputs_unchanged'] = source_before == source_after
                observed['native_closure_before'] = closure_before
                observed['native_closure_after'] = closure_after
                if source_before != source_after or closure_before != closure_after:
                    raise RuntimeError('I refuse changed observation source, native images or dependencies.')
                record['cases'].append({'name': name, **observed})
            (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        record['scheduling_after'] = scheduling(out, 'after')
        record['retained_images_unchanged'] = before_images == {str(path): digest(path) for path in all_images}
        record['completed'] = len(record['cases']) == len(combinations) and not any('error' in case for case in record['cases'])
        if not record['completed'] or not record['retained_images_unchanged']:
            raise RuntimeError('I preserve an incomplete or changed-image diagnostic without a qualification claim.')
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
