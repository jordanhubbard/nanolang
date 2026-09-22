#!/usr/bin/env python3
"""I schedule every resolved sanitizer unit target and require every result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

CFLAGS = '-Wall -Wextra -Werror -std=c99 -g -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer'
LDFLAGS = '-lm -lcrypto -fsanitize=address,undefined'
FLAGS = ['CFLAGS=' + CFLAGS, 'LDFLAGS=' + LDFLAGS]
NATIVE_CFLAGS = '-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
DEDICATED = ('test-forth-session', 'test-nanoisa-src-nano')
PROVIDERS = ['nanoisa_emit', 'nano_virt', 'nano_vm', 'nvm2c', 'nvm2c-runtime', 'nanoisa_dump']


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def validate_targets(targets):
    if not targets or len(targets) != len(set(targets)) or any(
            not re.fullmatch(r'test-[a-z0-9-]+', item) for item in targets):
        raise ValueError('I refuse ambiguous or unsupported unit prerequisites.')
    for required in (*DEDICATED, 'test-units-tail'):
        if required not in targets:
            raise ValueError('I require the original dedicated workers and trailing recipe.')
    return targets


def parse_database(text):
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith('test-units:')]
    if len(starts) != 1:
        raise ValueError('I require one resolved test-units rule.')
    start = starts[0]
    targets = lines[start].split()[1:]
    if 'test-units-tail' in targets:
        raise ValueError('I require the ordinary tail after all prerequisites, not parallel with them.')
    recipe = []
    for line in lines[start + 1:]:
        if not line:
            break
        if line.startswith('\t'):
            recipe.append(line)
        elif not line.startswith('#'):
            raise ValueError('I cannot identify the complete ordinary unit recipe.')
    if recipe != ['\t+@$(MAKE) test-units-tail']:
        raise ValueError('I require the exact sole post-prerequisite tail invocation.')
    return validate_targets([*targets, 'test-units-tail'])


def native_bootstrap_consumers(text, targets):
    """I derive compiler roles from resolved dependencies, including order-only edges."""
    graph = {}
    ambiguous = set()
    for line in text.splitlines():
        match = re.match(r'^([^\s:=#]+):[ \t]*(.*)$', line)
        if not match:
            continue
        target, body = match.groups()
        # Make prints target-specific variable assignments beside the real rule.
        if re.match(r'(?:(?:override|private|export)\s+)*[A-Za-z_][A-Za-z0-9_]*\s*[:+?]?=', body):
            continue
        dependencies = body.split()
        if any(not re.fullmatch(r'[A-Za-z0-9_./%+@-]+|\|', item) for item in dependencies):
            ambiguous.add(target)
            continue
        dependencies = [item for item in dependencies if item != '|']
        if target in graph and graph[target] != dependencies:
            ambiguous.add(target)
        graph[target] = dependencies
    markers = {'bootstrap', 'bootstrap1', 'bootstrap2', 'bootstrap3',
               '.bootstrap1.built', '.bootstrap2.built', '.bootstrap3.built'}
    selected = []
    for target in targets:
        if target not in graph:
            raise ValueError('I require the resolved rule for ' + target)
        pending = [target]
        seen = set()
        needed = False
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in ambiguous:
                raise ValueError('I cannot resolve compiler prerequisites for ' + current)
            needed = needed or current in markers
            pending.extend(graph.get(current, ()))
        if needed:
            selected.append(target)
    return selected


def resolve(output):
    command = ['make', '-qp', 'test-units', *FLAGS]
    result = subprocess.run(command, capture_output=True, timeout=60)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'make-database.stdout').write_bytes(result.stdout)
    (output / 'make-database.stderr').write_bytes(result.stderr)
    save(output / 'make-database.json', {'argv': command, 'returncode': result.returncode})
    if result.returncode not in (0, 1):
        raise ValueError('I could not resolve the actual Make unit inventory.')
    text = result.stdout.decode()
    targets = parse_database(text)
    return {'targets': targets, 'native_bootstrap_targets': native_bootstrap_consumers(text, targets)}


def plan(head, targets, native_bootstrap_targets=()):
    validate_targets(targets)
    native_bootstrap_targets = list(native_bootstrap_targets)
    if native_bootstrap_targets != [target for target in targets if target in native_bootstrap_targets]:
        raise ValueError('I require an ordered unique subset of bootstrap consumers.')
    workers = [{'id': 'forth', 'targets': [DEDICATED[0]]},
               {'id': 'source', 'targets': [DEDICATED[1]]}]
    remainder = [target for target in targets if target not in DEDICATED]
    workers.extend({'id': f'units-{index:02}', 'targets': remainder[index::14]}
                   for index in range(14))
    if any(not worker['targets'] for worker in workers):
        raise ValueError('I refuse empty unit partitions.')
    workers.append({'id': 'negative', 'targets': []})
    for worker in workers:
        worker['native_bootstrap'] = any(target in native_bootstrap_targets for target in worker['targets'])
    body = {'schema': 2, 'head': head, 'targets': targets, 'workers': workers,
            'native_bootstrap_targets': native_bootstrap_targets,
            'cflags': CFLAGS, 'ldflags': LDFLAGS, 'native_cflags': NATIVE_CFLAGS}
    return {**body, 'inventory_sha256': digest(body)}


def checked_plan(path):
    value = json.loads(Path(path).read_text())
    # Regeneration validates assignment, order, flags, identity and the digest.
    if value != plan(value['head'], value['targets'], value['native_bootstrap_targets']):
        raise ValueError('I refuse a modified partition manifest.')
    return value


def current_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def worker_from(manifest, name):
    matches = [worker for worker in manifest['workers'] if worker['id'] == name]
    if len(matches) != 1:
        raise ValueError('I require one exact named worker.')
    return matches[0]


def verify_local(manifest, output):
    if current_head() != manifest['head'] or resolve(output) != {key: manifest[key] for key in ('targets', 'native_bootstrap_targets')}:
        raise ValueError('I refuse a different source head or resolved target inventory.')


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def snapshot(output, name):
    sources = {}
    for raw in subprocess.check_output(['git', 'ls-files', '-z']).split(b'\0'):
        if not raw:
            continue
        path = os.fsdecode(raw)
        # Historical evidence is not a compiler/test source input.
        if path.startswith('docs/evidence/'):
            continue
        if Path(path).is_file():
            sources[path] = file_hash(path)
        else:
            raise ValueError('I cannot inventory a tracked input: ' + path)
    tools = {}
    for name_ in ('make', 'cc', 'gcc', 'ld', 'ar', 'nm', 'python3', 'perl', 'pkg-config'):
        executable = shutil.which(name_)
        if not executable:
            raise ValueError('I require my selected tool: ' + name_)
        tools[name_] = {'path': executable, 'resolved': os.path.realpath(executable),
                        'sha256': file_hash(executable)}
    for name_ in ('clang', 'opt', 'llvm-as', 'llvm-dis', 'lli', 'llc', 'llvm-nm', 'wasm-ld'):
        executable = shutil.which(name_)
        if executable:
            tools[name_] = {'path': executable, 'resolved': os.path.realpath(executable),
                            'sha256': file_hash(executable)}
    products = {}
    for base in ('bin', 'obj', 'obj-runtime', 'lib'):
        if Path(base).exists():
            for path in sorted(Path(base).rglob('*')):
                if path.is_file():
                    products[str(path)] = file_hash(path)
    value = {'head': current_head(), 'sources': sources, 'tools': tools, 'products': products}
    save(Path(output) / (name + '.json'), value)
    return value


def command_for(worker, phase):
    if phase == 'sanitize':
        return ['make', 'sanitize']
    if phase == 'bootstrap':
        return ['make', 'bootstrap3' if worker['native_bootstrap'] else 'build', *FLAGS]
    if phase == 'providers':
        if worker['id'] != 'source':
            raise ValueError('I prepare extra source-emitter providers only for their worker.')
        return ['make', *PROVIDERS, *FLAGS]
    if phase == 'tests':
        if worker['id'] == 'negative':
            return ['bash', 'tests/run_negative_tests.sh']
        return ['make', *worker['targets'], *FLAGS]
    raise ValueError('I refuse an unknown phase.')


def sanitizer_symbols(text):
    return {'asan': bool(re.search(r'\b__asan_(?:init|report_[A-Za-z0-9_]+)\b', text)),
            'ubsan': bool(re.search(r'\b__ubsan_handle_[A-Za-z0-9_]+\b', text))}


def instrumented_products(worker, output):
    nm = shutil.which('nm')
    if not nm:
        raise ValueError('I require nm to verify my actual compiler products.')
    paths = ['bin/nanoc_c']
    if worker['native_bootstrap']:
        paths += ['bin/nanoc_stage1', 'bin/nanoc_stage2']
    report = {'worker': worker['id'], 'native_cflags': NATIVE_CFLAGS,
              'nm': {'path': nm, 'sha256': file_hash(nm)}, 'products': {}, 'success': False}
    for name in paths:
        path = Path(name)
        if not path.is_file():
            save(Path(output) / 'instrumentation.json', report)
            raise ValueError('I require the actual compiler product: ' + name)
        before = file_hash(path)
        completed = subprocess.run([nm, str(path)], capture_output=True, timeout=60)
        prefix = Path(output) / (path.name + '-instrumentation')
        prefix.with_suffix('.stdout').write_bytes(completed.stdout)
        prefix.with_suffix('.stderr').write_bytes(completed.stderr)
        families = sanitizer_symbols(completed.stdout.decode(errors='replace'))
        report['products'][name] = {'sha256': before, 'returncode': completed.returncode, **families}
        save(Path(output) / 'instrumentation.json', report)
        if completed.returncode or not all(families.values()) or before != file_hash(path):
            raise ValueError('I require unchanged ASan and UBSan compiler instrumentation: ' + name)
    report['success'] = True
    save(Path(output) / 'instrumentation.json', report)
    return report


def instrumentation_stable(worker, output, prepared, after):
    path = Path(output) / 'instrumentation.json'
    if not path.is_file() or not prepared or not after:
        return False
    report = json.loads(path.read_text())
    required = ['bin/nanoc_c'] + (['bin/nanoc_stage1', 'bin/nanoc_stage2'] if worker['native_bootstrap'] else [])
    if (not report.get('success') or report.get('worker') != worker['id'] or
            report.get('native_cflags') != NATIVE_CFLAGS or set(report.get('products', {})) != set(required)):
        return False
    return all(product.get('returncode') == 0 and product.get('asan') and product.get('ubsan') and
               prepared['products'].get(name) == after['products'].get(name) == product.get('sha256')
               for name, product in report['products'].items())


def aggregate(manifest, root):
    actual = {}
    for path in Path(root).rglob('result.json'):
        value = json.loads(path.read_text())
        name = value['worker']
        if name in actual:
            raise ValueError('I refuse duplicate worker results.')
        actual[name] = value
    expected = {worker['id'] for worker in manifest['workers']}
    if set(actual) != expected:
        raise ValueError('I require every worker, including negative tests, exactly once.')
    for name, result in actual.items():
        worker = worker_from(manifest, name)
        if (result.get('head') != manifest['head'] or
                result.get('inventory_sha256') != manifest['inventory_sha256'] or
                result.get('targets') != worker['targets'] or
                result.get('native_bootstrap') != worker['native_bootstrap'] or
                not result.get('instrumentation_verified') or not result.get('success')):
            raise ValueError('I refuse failed, incomplete or mismatched worker evidence: ' + name)
    return {'success': True, 'head': manifest['head'],
            'inventory_sha256': manifest['inventory_sha256'], 'workers': sorted(actual)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'verify', 'command', 'snapshot', 'result', 'aggregate', 'instrumentation'))
    parser.add_argument('--manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker')
    parser.add_argument('--phase')
    parser.add_argument('--results')
    parser.add_argument('--github-output')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.action == 'plan':
        value = plan(current_head(), **resolve(output))
        save(output / 'plan.json', value)
        if args.github_output:
            with open(args.github_output, 'a') as stream:
                stream.write('matrix=' + json.dumps({'include': [{'id': w['id']} for w in value['workers']]}) + '\n')
        return
    manifest = checked_plan(args.manifest)
    if args.action == 'aggregate':
        if current_head() != manifest['head']:
            raise ValueError('I require the same aggregate checkout head.')
        save(output / 'aggregate.json', aggregate(manifest, args.results))
        return
    worker = worker_from(manifest, args.worker)
    if args.action == 'verify':
        verify_local(manifest, output)
    elif args.action == 'instrumentation':
        instrumented_products(worker, output)
    elif args.action == 'command':
        if os.environ.get('NANO_CFLAGS', NATIVE_CFLAGS) != NATIVE_CFLAGS:
            raise ValueError('I refuse different generated-native instrumentation flags.')
        os.environ['NANO_CFLAGS'] = NATIVE_CFLAGS
        os.environ['NANO_VERBOSE_BUILD'] = '1'
        command = command_for(worker, args.phase)
        if args.phase == 'tests' and worker['id'] == 'negative':
            os.environ['NANOLANG_COMPILER'] = './bin/nanoc_c'
            os.environ.pop('NANO_SHADOW_TIMEOUT_SECONDS', None)
        save(output / (args.phase + '-command.json'), {'argv': command, 'head': manifest['head'],
             'worker': worker, 'ASAN_OPTIONS': os.environ.get('ASAN_OPTIONS'),
             'NANO_SHADOW_TIMEOUT_SECONDS': os.environ.get('NANO_SHADOW_TIMEOUT_SECONDS'),
             'NANOLANG_COMPILER': os.environ.get('NANOLANG_COMPILER'),
             'NANO_CFLAGS': os.environ.get('NANO_CFLAGS'),
             'NANO_VERBOSE_BUILD': os.environ.get('NANO_VERBOSE_BUILD')})
        # The owning Actions step retains its original deadline and group cleanup.
        os.execvp(command[0], command)
    elif args.action == 'snapshot':
        value = snapshot(output, args.phase)
        if args.phase == 'prepared':
            original = json.loads((output / 'before.json').read_text())
            if any(original[key] != value[key] for key in ('head', 'sources', 'tools')):
                raise ValueError('I refuse source or tool drift before test execution.')
    elif args.action == 'result':
        steps = json.loads(os.environ['CI_SANITIZER_STEPS'])
        required = ['verify', 'before', 'sanitize', 'bootstrap', 'instrumentation', 'prepared', 'tests', 'after']
        if worker['id'] == 'source':
            required.append('providers')
        before = json.loads((output / 'before.json').read_text()) if (output / 'before.json').exists() else None
        after = json.loads((output / 'after.json').read_text()) if (output / 'after.json').exists() else None
        stable = bool(before and after and before['head'] == after['head'] == manifest['head'] and
                      before['sources'] == after['sources'] and before['tools'] == after['tools'])
        prepared = json.loads((output / 'prepared.json').read_text()) if (output / 'prepared.json').exists() else None
        instrumented = instrumentation_stable(worker, output, prepared, after)
        success = stable and instrumented and all(steps.get(name, {}).get('outcome') == 'success' for name in required)
        save(output / 'result.json', {'worker': worker['id'], 'targets': worker['targets'],
             'head': current_head(), 'inventory_sha256': manifest['inventory_sha256'],
             'source_tools_unchanged': stable, 'native_bootstrap': worker['native_bootstrap'],
             'instrumentation_verified': instrumented, 'steps': steps, 'success': success})
        if not success:
            raise ValueError('I retain a failed or incomplete sanitizer worker.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
