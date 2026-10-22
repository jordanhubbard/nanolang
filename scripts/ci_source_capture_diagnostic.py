#!/usr/bin/env python3
"""I retain current selected assembler diagnostics without changing admission."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

out = Path('.test_output/source-capture-diagnostic').resolve()
out.mkdir(parents=True, exist_ok=True)
records = []
selected_environment = dict(os.environ)
backend_hashes = {}


def run(args, label):
    try:
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=30,
                                env=selected_environment)
        record = {'label': label, 'command': list(map(str, args)), 'returncode': result.returncode,
                  'stdout': result.stdout, 'stderr': result.stderr,
                  'developer_dir': selected_environment.get('DEVELOPER_DIR'),
                  'sdk_root': selected_environment.get('SDKROOT')}
    except subprocess.TimeoutExpired as error:
        record = {'label': label, 'command': list(map(str, args)), 'timeout': 30,
                  'stdout': repr(error.stdout), 'stderr': repr(error.stderr)}
    records.append(record)
    (out/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
    print(label, record.get('returncode', 'timeout'), flush=True)
    return record


identity = {'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
            'environment': {name: os.environ.get(name) for name in
                            ('ImageOS', 'ImageVersion', 'PATH', 'DEVELOPER_DIR', 'SDKROOT', 'CC')},
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
(out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
for tool in ('cc', 'clang', 'as', 'gcc', 'xcrun', 'xcode-select', 'brew'):
    path = shutil.which(tool)
    if path:
        run([path, '-p' if tool == 'xcode-select' else '--version'], tool+'-version')
if sys.platform == 'darwin':
    run(['xcrun', '--find', 'as'], 'selected-as')
    run(['xcrun', '--find', 'clang'], 'selected-clang')
    run(['xcrun', '--show-sdk-path'], 'selected-sdk')

compilers = [(shutil.which('cc'), None), (shutil.which('clang'), None)]
if Path('/opt/homebrew/opt/llvm/bin/clang').is_file():
    compilers.append(('/opt/homebrew/opt/llvm/bin/clang', None))
if sys.platform == 'darwin':
    installed = []
    for app in sorted(Path('/Applications').glob('Xcode*.app')):
        developer = app/'Contents/Developer'
        compiler = developer/'Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'
        if compiler.is_file():
            installed.append({'application': str(app), 'developer_dir': str(developer),
                              'compiler': str(compiler)})
            compilers.append((str(compiler), str(developer)))
    (out/'installed-xcodes.json').write_text(json.dumps(installed, indent=2)+'\n')
seen = set()
for number, (compiler, developer) in enumerate(compilers):
    if not compiler or (compiler, developer) in seen:
        continue
    seen.add((compiler, developer))
    selected_environment = dict(os.environ)
    if developer:
        selected_environment['DEVELOPER_DIR'] = developer
    work = out/str(number); work.mkdir(exist_ok=True)
    source = work/'answer.c'
    source.write_text('long long nano_build_answer(void) { return 42; }\n')
    if sys.platform == 'darwin':
        run(['xcrun', '--find', 'as'], f'{number}-selected-as')
        sdk = run(['xcrun', '--show-sdk-path'], f'{number}-selected-sdk')
        if sdk.get('returncode') == 0 and Path(sdk['stdout'].strip()).is_dir():
            selected_environment['SDKROOT'] = sdk['stdout'].strip()
    resolved = Path(compiler).resolve()
    (work/'compiler.json').write_text(json.dumps({'selected': compiler, 'resolved': str(resolved),
        'developer_dir': developer,
        'sdk_root': selected_environment.get('SDKROOT'),
        'sha256': hashlib.sha256(resolved.read_bytes()).hexdigest()}, indent=2)+'\n')
    version = run([compiler, '--version'], f'{number}-compiler-version')
    run([compiler, '-print-prog-name=as'], f'{number}-assembler-path')
    assembly = work/'answer.s'
    run([compiler, '-S', '-O0', source, '-o', assembly], f'{number}-emit-assembly')
    for integrated in (True, False):
        mode = 'integrated' if integrated else 'external'
        flags = [] if integrated else ['-fno-integrated-as']
        if 'clang' in version.get('stdout', '') and integrated:
            flags = ['-fintegrated-as']
        output = work/(mode+'.o')
        command = [compiler, *flags, '-c', '-x', 'assembler', assembly, '-o', output]
        dry = run([*command, '-###'], f'{number}-{mode}-dry-run')
        run([*command, '-v'], f'{number}-{mode}-assemble')
        run([compiler, *flags, '-fPIC', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
             source, '-o', work/(mode+('.dylib' if sys.platform == 'darwin' else '.so'))],
            f'{number}-{mode}-fresh-shared')
        # I execute only actual quoted argv reported by this selected installed
        # compiler, without a shell. This is a diagnostic, not admission proof.
        for index, line in enumerate(dry.get('stderr', '').splitlines()):
            if not line.lstrip().startswith('"'):
                continue
            args = shlex.split(line)
            if args and Path(args[0]).is_file():
                backend = Path(args[0]).resolve()
                if str(backend) not in backend_hashes:
                    backend_hashes[str(backend)] = hashlib.sha256(backend.read_bytes()).hexdigest()
                    (out/'backend-hashes.json').write_text(json.dumps(backend_hashes, indent=2)+'\n')
                run(args, f'{number}-{mode}-direct-backend-{index}')
                run([*args, '-###'], f'{number}-{mode}-backend-dry-run-{index}')
    if 'clang' in version.get('stdout', ''):
        debug = work/'debug.s'
        expanded = work/'debug-expanded.s'
        run([compiler, '-S', '-g', '-O0', source, '-o', debug], f'{number}-debug-assembly')
        dry = run([compiler, '-fintegrated-as', '-c', '-x', 'assembler', debug,
                   '-o', work/'debug.o', '-###'], f'{number}-debug-dry-run')
        jobs = [shlex.split(line) for line in dry.get('stderr', '').splitlines()
                if line.lstrip().startswith('"')]
        if len(jobs) == 1 and len(jobs[0]) > 1 and jobs[0][1] == '-cc1as':
            args = jobs[0]
            if args.count('-filetype') == args.count('-o') == 1:
                args[args.index('-filetype') + 1] = 'asm'
                args[args.index('-o') + 1] = str(expanded)
                run([*args, '-msave-temp-labels'], f'{number}-debug-expand')
                run([compiler, '-fintegrated-as', '-c', '-x', 'assembler', expanded,
                     '-o', work/'debug-replay.o'], f'{number}-debug-replay')
manifest = {str(p.relative_to(out)): {'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in out.rglob('*') if p.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
