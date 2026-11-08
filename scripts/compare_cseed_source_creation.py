#!/usr/bin/env python3
"""I compare fixed, alternating source creations without replacing release gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('original', 'modified', 'modified', 'original', 'original', 'modified')
PROVIDERS = ('bin/nanoisa', 'bin/nano_vm', 'bin/nvm2c', 'bin/nanoc_c',
             'bin/nano_virt', 'bin/nano_aot_runtime.o')
FACTORY_FILES = ('tests/test_bstring_source_vm.py', 'tests/test_bstring_vm_contracts.py',
                 'tests/test_numeric_builtin_selection.py', 'tests/native_process.py',
                 'scripts/diagnose_bstring_contracts.py')


def digest(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
        return value.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def shared_inputs(root):
    paths = list((root/'src_nano').rglob('*.nano'))
    paths += list((root/'modules').rglob('*.nano'))
    paths += [root/p for p in FACTORY_FILES]
    return {str(p.relative_to(root)): digest(p) for p in sorted(set(paths))}


def require_same_inputs(original, modified):
    left, right = shared_inputs(original), shared_inputs(modified)
    if left != right:
        changed = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
        raise ValueError('I refuse different comparison inputs: ' + ', '.join(changed))
    return left


def health(output, label):
    disks = []
    cache = Path.home()/'Library/Caches/Homebrew' if sys.platform == 'darwin' else Path('/var/cache/apt')
    for location in (ROOT, Path.home(), Path(tempfile.gettempdir()),
                     Path(os.environ.get('RUNNER_TEMP', tempfile.gettempdir())), cache):
        existing = location
        while not existing.exists():
            existing = existing.parent
        usage = shutil.disk_usage(existing)
        disks.append({'path': str(location), 'free': usage.free, 'total': usage.total})
    if sys.platform == 'darwin':
        raw = subprocess.check_output(['vm_stat'], text=True, timeout=20)
        pages = {line.split(':')[0]: int(line.split(':')[1].strip().rstrip('.'))
                 for line in raw.splitlines()[1:] if ':' in line}
        page_size = int(subprocess.check_output(['sysctl', '-n', 'hw.pagesize'], text=True, timeout=20))
        total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True, timeout=20))
        available = page_size * sum(pages.get(k, 0) for k in ('Pages free', 'Pages inactive', 'Pages speculative'))
    else:
        memory = {line.split(':')[0]: int(line.split()[1])*1024
                  for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith(('MemTotal:', 'MemAvailable:'))}
        total, available = memory['MemTotal'], memory['MemAvailable']
    report = {'time_ns': time.time_ns(), 'cpus': os.cpu_count(), 'load_average': os.getloadavg(),
              'disks': disks, 'memory': {'total': total, 'available': available}}
    write(output/'health'/f'{label}.json', report)
    if any(d['free'] < 20*1024**3 or d['free']/d['total'] < .10 for d in disks):
        print('I hold comparison builds for disk pressure and inspect cleanup candidates.', flush=True)
        candidates = [p for p in (ROOT/'obj', ROOT/'bin', ROOT/'.test_output',
                      Path.home()/'.worktrees/nanolang', cache) if p.exists()]
        try:
            inspected = subprocess.run(['du', '-sk', *map(str, candidates)],
                capture_output=True, text=True, timeout=20)
            investigation = {'argv': ['du', '-sk', *map(str, candidates)],
                'returncode': inspected.returncode, 'stdout': inspected.stdout,
                'stderr': inspected.stderr, 'deleted': False}
        except subprocess.TimeoutExpired:
            investigation = {'status': 'unknown: bounded size inspection timed out', 'deleted': False}
        write(output/'health'/f'{label}-cleanup-candidates.json', investigation)
        raise RuntimeError('I hold comparison builds for insufficient disk headroom.')
    if available/total <= .10:
        raise RuntimeError('I hold comparison builds for memory pressure.')


def command(argv, cwd, log, env, timeout):
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    with log.open('wb') as stream:
        child = subprocess.Popen(list(map(str, argv)), cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            # I let the factory's native-process wrapper reap its separate C session.
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'argv': list(map(str, argv)), 'cwd': str(cwd), 'returncode': code,
            'wall_seconds': time.monotonic()-started,
            'child_user_seconds': after.ru_utime-usage.ru_utime,
            'child_system_seconds': after.ru_stime-usage.ru_stime}


def shadow_trace(text, root):
    events = [json.loads(m.group()) for m in re.finditer(
        r'\{"event":"shadow-(?:start|complete)"[^\n]*?\}', text)]
    starts = [e for e in events if e['event'] == 'shadow-start']
    ends = [e for e in events if e['event'] == 'shadow-complete']
    identity = lambda e: (e['name'], e['source'], e['line'])
    if list(map(identity, ends)) != list(map(identity, starts[:len(ends)])) or len(ends) > len(starts):
        raise ValueError('I refuse unpaired shadow completion events.')
    return {'starts': len(starts), 'completes': len(ends),
            'failures': sum(e['failures'] for e in ends),
            'summed_body_seconds': sum(e['elapsed_seconds'] for e in ends),
            'selection_prefix': [(e['name'], e['source'].replace(str(root)+'/', ''), e['line'])
                                 for e in starts]}



def check_selection(prefix, full_selection, selection, succeeded):
    common = min(len(selection), len(prefix))
    if selection[:common] != prefix[:common]:
        raise ValueError('I refuse a changed selected shadow order.')
    if succeeded:
        if full_selection is not None and selection != full_selection:
            raise ValueError('I refuse a shorter or different successful shadow selection.')
        full_selection = selection
    if full_selection is not None and prefix != full_selection[:len(prefix)]:
        raise ValueError('I refuse a completed selection shorter than an earlier observed prefix.')
    if full_selection is not None and selection != full_selection[:len(selection)]:
        raise ValueError('I refuse a selection outside the completed original selection.')
    return (selection if len(selection) > len(prefix) else prefix), full_selection


def run(output, baseline):
    if not re.fullmatch('[0-9a-f]{40}', baseline):
        raise ValueError('I require a full lowercase baseline commit SHA.')
    if os.environ.get('NANO_SHADOW_TIMEOUT_SECONDS') not in (None, '10'):
        raise ValueError('I require the unchanged ten-second shadow limit.')
    if (output/'plan.json').exists():
        raise ValueError('I refuse to overwrite an existing comparison.')
    sys.path.insert(0, str(ROOT))
    from tests.native_process import native_compile_timeout
    native_budget = native_compile_timeout()
    candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    parent = Path.home()/'.worktrees/nanolang'
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='cseed-profile-', dir=parent))
    roots = {name: workspace/name for name in ('original', 'modified')}
    plan = {'scope': 'I measure first source creations; I do not replace complete component/platform/release qualification.',
            'baseline': baseline, 'candidate': candidate, 'order': ORDER, 'worktrees': {k: str(v) for k,v in roots.items()},
            'shadow_timeout_seconds': 10, 'native_build_budget': native_budget,
            'native_lifecycle_admitted': False, 'fresh_stages_qualified': False}
    write(output/'plan.json', plan)
    base_env = dict(os.environ, NANO_SHADOW_TRACE='1', NANO_NVM2C_SHAPE_TRACE='1',
                    NANO_STRING_OPERAND_COMPILERS='nanoc_c,nano_virt',
                    NANO_PRINT_STATEMENT_COMPILERS='nanoc_c,nano_virt',
                    NANO_EMPTY_RECORD_PROVIDERS='nanoc_c,nano_virt',
                    NANO_NUMERIC_PROVIDERS='nano_virt',
                    NANO_MAP_CONTEXT_PROVIDERS='nanoc_c,nano_virt')
    records, providers, environments = {}, {}, {}
    for name, revision in (('original', baseline), ('modified', candidate)):
        subprocess.run(['git', 'worktree', 'add', '--detach', str(roots[name]), revision], cwd=ROOT, check=True)
    common = require_same_inputs(roots['original'], roots['modified'])
    write(output/'shared-inputs.json', common)
    cc = os.environ.get('CC', 'cc')
    tool = Path(shutil.which(cc) or cc).resolve(strict=True)
    tool_hash = digest(tool)
    version = subprocess.check_output([cc, '--version'], text=True, timeout=20)
    write(output/'compiler.json', {'path': str(tool), 'sha256': tool_hash, 'version': version,
                                  'environment': {k: os.environ[k] for k in ('CC','CFLAGS','CPPFLAGS','LDFLAGS','CMD_TIMEOUT','DEVELOPER_DIR','SDKROOT') if k in os.environ}})
    for name, root in roots.items():
        health(output, 'before-build-'+name)
        env = dict(base_env, NANOLANG_SELFHOST_COMPILER=str(root/'bin/nano_virt'), NANO_NATIVE_TEST_CC=cc)
        environments[name] = env
        argv = ['make', '-f', 'Makefile.gnu', '-j2', 'CC='+cc, 'nanoisa_dump', 'nano_vm',
                'nvm2c', 'bin/nanoisa', 'bin/nanoc_c', 'nano_virt', 'nvm2c-runtime']
        records[name] = command(argv, root, output/'builds'/f'{name}.log', env, 1500)
        write(output/'builds.json', records)
        if records[name]['returncode']:
            raise RuntimeError('I require both actual C providers to build.')
        if shared_inputs(root) != common:
            raise RuntimeError('I refuse source inputs changed by provider creation.')
        providers[name] = {p: digest(root/p) for p in PROVIDERS}
        providers[name]['src/runtime/gc.c'] = digest(root/'src/runtime/gc.c')
        providers[name]['src/runtime/gc.h'] = digest(root/'src/runtime/gc.h')
        destination = output/'providers'/name
        destination.mkdir(parents=True)
        shutil.copy2(root/'bin/nanoc_c', destination/'nanoc_c')
        if digest(destination/'nanoc_c') != providers[name]['bin/nanoc_c']:
            raise RuntimeError('I refuse a changed copied provider.')
    write(output/'providers-before.json', providers)
    samples, prefix, full_selection = [], [], None
    for index, name in enumerate(ORDER, 1):
        health(output, f'before-sample-{index:02}')
        root = roots[name]
        factory = root/'.test_output/bstring-source-compiler'
        prior = set(factory.iterdir()) if factory.exists() else set()
        if digest(tool) != tool_hash:
            raise RuntimeError('I refuse a changed host compiler.')
        result = command([sys.executable, Path(__file__).resolve(), 'sample', str(root)], root,
                         output/'samples'/f'{index:02}-{name}.log', environments[name], native_budget + 60)
        created = set(factory.iterdir())-prior
        if len(created) != 1:
            raise RuntimeError('I require exactly one actual factory attempt per planned sample.')
        evidence = created.pop()
        copied = output/'samples'/f'{index:02}-{name}'
        shutil.copytree(evidence, copied)
        inputs = json.loads((evidence/'inputs.json').read_text())
        if inputs['before'] != inputs['after'] or shared_inputs(root) != common:
            raise RuntimeError('I refuse changed factory inputs.')
        for path, value in providers[name].items():
            if digest(root/path) != value:
                raise RuntimeError('I refuse a changed provider or runtime input.')
        trace = shadow_trace((evidence/'source-build.stderr').read_text(), root)
        selection = trace['selection_prefix']
        completed = result['returncode'] == 0
        prefix, full_selection = check_selection(prefix, full_selection, selection, completed)
        if completed and (not trace['starts'] or trace['starts'] != trace['completes'] or trace['failures'] or
                          not inputs['product_before_run'] or
                          inputs['product_before_run'] != inputs['product_after_run']):
            raise RuntimeError('I refuse an incomplete successful source creation.')
        for path in evidence.rglob('*'):
            if path.is_file() and digest(path) != digest(copied/path.relative_to(evidence)):
                raise RuntimeError('I refuse changed copied evidence.')
        samples.append({'index': index, 'variant': name, 'process': result, 'creation_succeeded': completed,
                        'trace': trace, 'factory_directory': str(evidence), 'copied_directory': str(copied),
                        'product_sha256': inputs['product_before_run']})
        summary = {'plan': plan, 'comparison_valid': False,
              'all_planned_samples_finished': len(samples)==len(ORDER),
              'all_creations_passed': all(s['creation_succeeded'] for s in samples), 'samples': samples}
        write(output/'summary.json', summary)
        print(json.dumps({'index': index, 'variant': name, 'creation_succeeded': completed,
                          'completes': trace['completes'], 'body_seconds': trace['summed_body_seconds']}), flush=True)
    after = {name: {p: digest(roots[name]/p) for p in paths} for name, paths in providers.items()}
    write(output/'providers-after.json', after)
    if after != providers:
        raise RuntimeError('I refuse changed providers after comparison.')
    if any(shared_inputs(root) != common for root in roots.values()):
        raise RuntimeError('I refuse changed inputs after comparison.')
    summary['comparison_valid'] = True
    write(output/'summary.json', summary)
    return int(any(not s['creation_succeeded'] for s in samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    check = sub.add_parser('health'); check.add_argument('--output', type=Path, required=True)
    compare = sub.add_parser('run'); compare.add_argument('--output', type=Path, required=True)
    compare.add_argument('--baseline', required=True)
    sample = sub.add_parser('sample'); sample.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.mode == 'sample':
        sys.path.insert(0, str(args.root.resolve()))
        os.chdir(args.root)
        from tests.test_bstring_source_vm import BstringSourceVmContracts
        BstringSourceVmContracts.setUpClass()
        return 0
    if args.mode == 'health':
        health(args.output.resolve(), 'preflight')
        return 0
    return run(args.output.resolve(), args.baseline)


if __name__ == '__main__':
    raise SystemExit(main())
