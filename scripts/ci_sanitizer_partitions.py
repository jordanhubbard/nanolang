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
import tarfile

CFLAGS = '-Wall -Wextra -Werror -std=c99 -g -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer'
LDFLAGS = '-lm -lcrypto -fsanitize=address,undefined'
FLAGS = ['CFLAGS=' + CFLAGS, 'LDFLAGS=' + LDFLAGS]
BOOTSTRAP_FLAGS = ['BOOTSTRAP2_SHADOW_FLAG=--root-shadows-only']
BOOTSTRAP_DRIVER_FLAGS = ['NANOC_STAGE1=bin/nanoc_stage1_driver']
NATIVE_CFLAGS = '-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
STAGE2_NATIVE_CFLAGS = '-O1 -gline-tables-only -fno-inline-functions -fsanitize=address,undefined -fno-omit-frame-pointer'
NATIVE_CC = 'clang'
STAGE2_NVM = 'build/sanitizer-stage2/compiler.nvm'
STAGE2_C = 'build/sanitizer-stage2/compiler.c'
STAGE2_OBJECT = 'build/sanitizer-stage2/compiler.o'
STAGE2_PROVIDER_OBJECTS = tuple(
    'obj/file-companion-plan/' + name + '.o' for name in
    ('file_companion_bridge', 'file_source_input', 'file_companion_snapshot',
     'nsi_file_binding', 'nsi_file_plan', 'nsi', 'file_source_catalog'))
DEDICATED = ('test-forth-session', 'test-nanoisa-src-nano', 'test-scalar-reconstruction')
PHASES = (('foundation', 'test-ci-foundation'),
          ('programs-language', 'test-ci-programs-language'),
          ('programs-app', 'test-ci-programs-app'),
          ('programs-unit', 'test-ci-programs-unit'),
          ('contracts', 'test-ci-contracts'),
          ('integrations', 'test-ci-integrations'),
          ('forth-complete', 'test-ci-forth'),
          ('runtime', 'test-ci-runtime'))
PROVIDERS = ['nanoisa_emit', 'nano_virt', 'nano_vm', 'nvm2c', 'nvm2c-runtime', 'nanoisa_dump']
UNIT_PARTITIONS = 28
BUNDLE_ROOTS = ('bin', 'obj', 'obj-runtime', 'lib', 'build')
BUNDLE_SENTINELS = ('.stage1.built', '.stage2.built', '.stage3.built',
                    '.bootstrap0.built', '.bootstrap1.built', '.bootstrap2.built', '.bootstrap3.built')
BUNDLE_PRODUCTS = {
    'base': ('bin/nanoc_c',),
    'stage1': ('bin/nanoc_c', 'bin/nanoc_stage1', 'bin/nanoc_stage1_driver',
               'bin/nano_vm', 'bin/nvm2c', 'bin/nano_aot_runtime.o', *STAGE2_PROVIDER_OBJECTS),
    'stage2-nvm': ('bin/nanoc_c', 'bin/nanoc_stage1', 'bin/nanoc_stage1_driver',
                   'bin/nano_vm', 'bin/nvm2c', 'bin/nano_aot_runtime.o',
                   *STAGE2_PROVIDER_OBJECTS, STAGE2_NVM),
    'stage2-c': ('bin/nanoc_c', 'bin/nanoc_stage1', 'bin/nanoc_stage1_driver',
                 'bin/nano_vm', 'bin/nvm2c', 'bin/nano_aot_runtime.o',
                 *STAGE2_PROVIDER_OBJECTS, STAGE2_NVM, STAGE2_C),
    'stage2-object': ('bin/nanoc_c', 'bin/nanoc_stage1', 'bin/nanoc_stage1_driver',
                      'bin/nano_vm', 'bin/nvm2c', 'bin/nano_aot_runtime.o',
                      *STAGE2_PROVIDER_OBJECTS, STAGE2_NVM, STAGE2_C, STAGE2_OBJECT),
    'stage2': ('bin/nanoc_c', 'bin/nanoc_stage1', 'bin/nanoc_stage1_driver', 'bin/nanoc_stage2'),
    'bootstrap': ('bin/nanoc_c', 'bin/nanoc_stage1', 'bin/nanoc_stage2',
                  'bin/nanoisa_emit', 'bin/nano_virt', 'bin/nano_vm', 'bin/nvm2c', 'bin/nanoisa_dump'),
}


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
    for required in (*DEDICATED, *(target for _, target in PHASES), 'test-units-tail'):
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
            if line.strip():
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
               {'id': 'source', 'targets': [DEDICATED[1]]},
               {'id': 'scalar', 'targets': [DEDICATED[2]]}]
    workers.extend({'id': name, 'targets': [target]} for name, target in PHASES)
    dedicated = set(DEDICATED) | {target for _, target in PHASES}
    remainder = [target for target in targets if target not in dedicated]
    workers.extend({'id': f'units-{index:02}', 'targets': remainder[index::UNIT_PARTITIONS]}
                   for index in range(UNIT_PARTITIONS))
    if any(not worker['targets'] for worker in workers):
        raise ValueError('I refuse empty unit partitions.')
    workers.append({'id': 'negative', 'targets': []})
    for worker in workers:
        worker['native_bootstrap'] = any(target in native_bootstrap_targets for target in worker['targets'])
    body = {'schema': 4, 'head': head, 'targets': targets, 'workers': workers,
            'native_bootstrap_targets': native_bootstrap_targets,
            'cflags': CFLAGS, 'ldflags': LDFLAGS, 'native_cflags': NATIVE_CFLAGS,
            'native_cc': NATIVE_CC, 'stage2_native_cflags': STAGE2_NATIVE_CFLAGS}
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


def bundle_create(output, manifest, stage):
    if stage not in BUNDLE_PRODUCTS or current_head() != manifest['head']:
        raise ValueError('I require an exact supported sanitizer bundle stage.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    required = BUNDLE_PRODUCTS[stage]
    if any(not Path(path).is_file() for path in required):
        raise ValueError('I require every sanitizer bundle product before publication.')
    selected = [Path(path) for path in (*BUNDLE_ROOTS, *BUNDLE_SENTINELS) if Path(path).exists()]
    archive = output / (stage + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        for path in selected:
            stream.add(path, arcname=str(path), recursive=True)
    metadata = {'schema': 1, 'head': manifest['head'], 'stage': stage,
                'inventory_sha256': manifest['inventory_sha256'],
                'archive': archive.name, 'archive_sha256': file_hash(archive),
                'products': {path: file_hash(path) for path in required}}
    save(output / (stage + '.json'), metadata)
    return metadata


def safe_bundle_member(member):
    path = Path(member.name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        return False
    if path.parts[0] not in set(BUNDLE_ROOTS) | set(BUNDLE_SENTINELS):
        return False
    if member.islnk():
        return False
    if member.issym():
        target = Path(member.linkname)
        if target.is_absolute() or '..' in target.parts:
            return False
    return member.isdir() or member.isfile() or member.issym()


def bundle_restore(output, manifest, stage, archive, metadata):
    if stage not in BUNDLE_PRODUCTS or current_head() != manifest['head']:
        raise ValueError('I require an exact supported sanitizer bundle stage.')
    archive, metadata = Path(archive), json.loads(Path(metadata).read_text())
    expected = {'schema': 1, 'head': manifest['head'], 'stage': stage,
                'inventory_sha256': manifest['inventory_sha256'], 'archive': archive.name}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('I refuse a sanitizer bundle from another source or plan.')
    if metadata.get('archive_sha256') != file_hash(archive):
        raise ValueError('I refuse a modified sanitizer build archive.')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        if not members or any(not safe_bundle_member(member) for member in members):
            raise ValueError('I refuse an unsafe sanitizer build archive.')
        stream.extractall('.', members=members)
    products = metadata.get('products', {})
    if set(products) != set(BUNDLE_PRODUCTS[stage]) or any(
            not Path(path).is_file() or file_hash(path) != digest_ for path, digest_ in products.items()):
        raise ValueError('I refuse incomplete sanitizer build products.')
    # A checkout in a dependent job is newer than the archived products. I
    # preserve bytes, then make the verified restored graph current for Make.
    for root in BUNDLE_ROOTS:
        path = Path(root)
        if path.exists():
            for child in path.rglob('*'):
                if child.is_file():
                    child.touch()
    for name in BUNDLE_SENTINELS:
        path = Path(name)
        if path.exists():
            path.touch()
    report = {**expected, 'archive_sha256': metadata['archive_sha256'],
              'products': products, 'success': True}
    save(Path(output) / ('restore-' + stage + '.json'), report)
    return report


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
        return ['make', 'bootstrap3' if worker['native_bootstrap'] else 'build',
                *BOOTSTRAP_FLAGS, *BOOTSTRAP_DRIVER_FLAGS, *FLAGS]
    if phase == 'bootstrap1':
        return ['make', 'bootstrap1', *FLAGS]
    if phase == 'bootstrap1-driver':
        return ['make', 'bootstrap1-driver', *BOOTSTRAP_FLAGS, *FLAGS]
    if phase == 'bootstrap1-companions':
        return ['make', 'file-companion-plan', *FLAGS]
    if phase == 'bootstrap2-nvm':
        return ['bin/nanoc_stage1', '--root-shadows-only', 'src_nano/nanoc_v06.nano',
                '--emit-nvm', '-o', STAGE2_NVM]
    if phase == 'bootstrap2-c':
        return ['bin/nvm2c', STAGE2_NVM, '-o', STAGE2_C]
    if phase == 'bootstrap2-object':
        return [NATIVE_CC, '-std=c11', *STAGE2_NATIVE_CFLAGS.split(), STAGE2_C,
                '-c', '-o', STAGE2_OBJECT]
    if phase == 'bootstrap2-native':
        return [NATIVE_CC, STAGE2_OBJECT, 'bin/nano_aot_runtime.o',
                *STAGE2_PROVIDER_OBJECTS, '-lm', '-Wl,--export-dynamic', '-ldl',
                '-fsanitize=address,undefined',
                '-o', 'bin/nanoc_stage2']
    if phase == 'bootstrap3':
        return ['make', 'bootstrap3', *BOOTSTRAP_FLAGS, *BOOTSTRAP_DRIVER_FLAGS, *FLAGS]
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
              'stage2_native_cflags': STAGE2_NATIVE_CFLAGS, 'native_cc': NATIVE_CC,
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
            report.get('native_cflags') != NATIVE_CFLAGS or
            report.get('stage2_native_cflags') != STAGE2_NATIVE_CFLAGS or
            report.get('native_cc') != NATIVE_CC or
            set(report.get('products', {})) != set(required)):
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
    parser.add_argument('action', choices=('plan', 'verify', 'command', 'snapshot', 'result', 'aggregate',
                                           'instrumentation', 'bundle', 'restore'))
    parser.add_argument('--manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker')
    parser.add_argument('--phase')
    parser.add_argument('--results')
    parser.add_argument('--github-output')
    parser.add_argument('--stage', choices=tuple(BUNDLE_PRODUCTS))
    parser.add_argument('--archive')
    parser.add_argument('--metadata')
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
    if args.action == 'bundle':
        bundle_create(output, manifest, args.stage)
        return
    if args.action == 'restore':
        bundle_restore(output, manifest, args.stage, args.archive, args.metadata)
        return
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
        if os.environ.get('NANO_CC', NATIVE_CC) != NATIVE_CC:
            raise ValueError('I refuse a different generated-native compiler.')
        os.environ['NANO_CFLAGS'] = NATIVE_CFLAGS
        os.environ['NANO_CC'] = NATIVE_CC
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
             'NANO_CC': os.environ.get('NANO_CC'),
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
        required = ['verify', 'restore', 'before', 'instrumentation', 'prepared', 'tests', 'after']
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
