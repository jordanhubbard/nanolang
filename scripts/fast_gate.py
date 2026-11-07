#!/usr/bin/env python3
"""I run strict product compiles and measure optional Make/private test profiles.

I read the compile commands from a Makefile dry run, so I use the real flags and cannot drift from the
build. I discard the objects, run the compiles in parallel and report only failures. A GCC compile with
optimization finds the maybe-uninitialized and unused-but-set errors that cost a hosted round trip."""
import argparse
import ast
import concurrent.futures
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.native_process import run_captured

TARGETS = ['nano_virt', 'nano_vm', 'nvm2c', 'nanoisa_dump', 'nvm2c-runtime']
MARKER = '__NANO_FAST_CC__'


def make_preview(targets, assignments=()):
    """I suppress recursive recipes, including recipes Make executes under -n."""
    text = (ROOT / 'Makefile.gnu').read_text()
    text = text.replace('$(MAKE)', '$(NANO_FAST_NO_RECURSE_MAKE)')
    text = text.replace('${MAKE}', '$(NANO_FAST_NO_RECURSE_MAKE)')
    if re.search(r'^\s*\.RECIPEPREFIX\s*[:?+]?=', text, re.M):
        raise ValueError('I require a reviewed preview for a custom Make recipe prefix.')
    text = re.sub(r'(?m)^\t([@+\-]*)',
                  lambda match: '\t' + match.group(1).replace('+', ''), text)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mk') as preview:
        preview.write(text)
        preview.flush()
        environment = {key: value for key, value in os.environ.items()
                       if key not in {'MAKEFLAGS', 'MFLAGS', 'MAKEOVERRIDES'}}
        result = run_captured(['make', '--no-print-directory', '-n', '-B', '-f', preview.name,
                               'NANO_FAST_NO_RECURSE_MAKE=make', f'CC={MARKER}',
                               *assignments, *targets], cwd=ROOT, text=True, timeout=300,
                              env=environment)
    if result.returncode:
        raise ValueError('I could not read the Make compile plan:\n' + result.stderr[-4000:])
    return result.stdout


def recipe_commands(text):
    """I read compiler argv without executing shell fragments or substitutions."""
    for line in re.sub(r'\\\n', ' ', text).splitlines():
        if MARKER not in line:
            continue
        # shlex's default wordchars split flags and paths unless whitespace mode is enabled.
        lexer = shlex.shlex(line, posix=True, punctuation_chars=';&|')
        lexer.whitespace_split = True
        lexer.commenters = ''
        words = list(lexer)
        for start, word in enumerate(words):
            if word != MARKER:
                continue
            end = next((i for i in range(start + 1, len(words))
                        if words[i] and set(words[i]) <= set(';&|')), len(words))
            yield words[start + 1:end]


def compile_profile(words):
    """I preserve preprocessing and diagnostic flags, and discard only link/output operands."""
    paired_drop = {'-o', '-MF', '-MT', '-MQ', '-L', '-l', '-framework', '-Xlinker',
                   '-undefined', '-install_name'}
    paired_keep = {'-I', '-isystem', '-iquote', '-include', '-imacros', '-isysroot',
                   '--sysroot', '-x', '-D', '-U'}
    flags, sources, i = [], [], 0
    while i < len(words):
        word = words[i]
        i += 1
        if '$' in word or '`' in word:
            raise ValueError('I require an explicit compile profile for shell operand: ' + word)
        if word in paired_drop | paired_keep:
            if i == len(words):
                raise ValueError('I require an argument after ' + word)
            if word in paired_keep:
                flags.extend([word, words[i]])
            i += 1
        elif word.endswith('.c'):
            if not (ROOT / word).is_file():
                raise ValueError('I cannot find compile input ' + word)
            sources.append(word)
        elif word in {'-c', '-S', '-MD', '-MMD', '-MP', '-shared', '-dynamiclib',
                      '-rdynamic', '-fsyntax-only'} or re.fullmatch(r'-O[0-3sg]?', word):
            continue
        elif word.startswith(('-l', '-L', '-Wl,')) or word.endswith(('.o', '.a', '.so', '.dylib')):
            continue
        elif word.startswith('-'):
            flags.append(word)
        else:
            raise ValueError('I cannot classify compile operand ' + word)
    return flags, sources


def private_union_profiles(base):
    """I read the literal flags used by my two private Python harnesses."""
    profiles = []
    for harness, source, whitebox in (
            ('tests/test_union_field_preparation.py', 'tests/nanoisa/test_union_field_preparation.c',
             ['-DVM_UF_WHITEBOX']),
            ('tests/test_union_field_vm.py', 'tests/nanoisa/test_union_field_vm.c',
             ['-DUNION_EXECUTION_WHITEBOX', '-DUNION_EXECUTION_ALLOC'])):
        tree = ast.parse((ROOT / harness).read_text())
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Attribute) and t.attr == 'flags'
                               for t in node.targets)]
        if len(assignments) != 1 or not isinstance(assignments[0].value, ast.List):
            raise ValueError('I require a reviewed private compile profile in ' + harness)
        literals = [x.value for x in assignments[0].value.elts
                    if isinstance(x, ast.Constant) and isinstance(x.value, str)]
        if '-DNANO_UNION_FIELD_PRIVATE_PREPARATION' not in literals:
            raise ValueError('I cannot find the private union define in ' + harness)
        flags, _ = compile_profile([*base, *literals])
        for dispatch in ([], ['-DNANO_NO_COMPUTED_GOTO']):
            profiles.extend(([*flags, *dispatch], unit) for unit in ('src/nanovm/vm.c', source))
            profiles.append(([*flags, *dispatch, *whitebox], source))
    return profiles


def test_profiles(compiler, family, header_root):
    """I cover Make-built units and private union modes; I report the remaining C inventory."""
    text = make_preview(['test-units'], [f'WEBSOCKET_PUBLIC_TEST_PREFIX={header_root}'])
    commands, skipped = [], []
    base = None
    for words in recipe_commands(text):
        # This optional recipe selects the same installed pkg-config provider as Make.
        if '$(pkg-config' in words:
            if 'tests/test_unicode_ffi.c' not in words:
                raise ValueError('I require an explicit profile for this pkg-config recipe.')
            provider = next((name for name in ('libutf8proc', 'utf8proc')
                             if shutil.which('pkg-config') and subprocess.run(
                                 ['pkg-config', '--exists', name], capture_output=True).returncode == 0), None)
            if not provider:
                skipped.append('unicode FFI: pkg-config libutf8proc/utf8proc is absent')
                continue
            cflags = shlex.split(subprocess.check_output(['pkg-config', '--cflags', provider], text=True))
            start = words.index('$(pkg-config')
            words[start:start + 3] = cflags
            start = words.index('$(pkg-config')
            del words[start:start + 3]  # I need no library operands for a translation-only compile.
        flags, sources = compile_profile(words)
        if base is None and sources:
            base = flags
        if '-fblocks' in flags and family != 'clang':
            skipped.extend(source + ': Darwin blocks require Clang' for source in sources)
            continue
        commands.extend((flags, source) for source in sources)
    if base is None:
        raise ValueError('I found no Make test profiles.')
    commands.extend(private_union_profiles(base))
    unique = {}
    for flags, source in commands:
        for instrumentation in ([], ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']):
            args = [compiler, *flags, '-O3', '-D_FORTIFY_SOURCE=2', *instrumentation, '-S', source]
            unique[tuple(args)] = args + ['-o', os.devnull]
    return list(unique.values()), skipped


def stage_public_headers(destination):
    """I copy exactly the Make-owned public header list into private temporary storage."""
    text = (ROOT / 'Makefile.gnu').read_text()
    match = re.search(r'^WEBSOCKET_PUBLIC_HEADERS = ([\s\S]*?)\n\.PHONY:', text, re.M)
    if not match:
        raise ValueError('I cannot find the public WebSocket header manifest.')
    for name in match.group(1).replace('\\\n', ' ').split():
        if '$' in name or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('I require a literal public header manifest: ' + name)
        target = Path(destination) / 'include/nanolang/websocket' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'src' / name, target)


def detect_compilers():
    found = {}
    for name in ('clang', 'cc'):
        path = shutil.which(name)
        if path and 'clang' in subprocess.run([path, '--version'], capture_output=True, text=True).stdout.lower():
            found['clang'] = path
            break
    versions = sorted((p for p in (shutil.which(f'gcc-{n}') for n in range(9, 30)) if p), reverse=True)
    gcc = versions[0] if versions else shutil.which('gcc')
    if gcc and 'gcc' in subprocess.run([gcc, '--version'], capture_output=True, text=True).stdout.lower() and \
            'clang' not in subprocess.run([gcc, '--version'], capture_output=True, text=True).stdout.lower():
        found['gcc'] = gcc
    return found


def compile_commands(compiler):
    commands = []
    for words in recipe_commands(make_preview(TARGETS)):
        if '-c' not in words:
            continue
        flags, sources = compile_profile(words)
        # I stop before assembly (-S): I need the compiler's diagnostics, and a host assembler can
        # reject valid code (Darwin GCC reports an unrelated relocation error for runtime/gc.c).
        commands.extend([compiler, *flags, '-S', source, '-o', os.devnull] for source in sources)
    return commands


def run(command):
    started = time.monotonic()
    try:
        result = run_captured(command, cwd=ROOT, text=True, timeout=180)
        return command, result.returncode, result.stderr or result.stdout, time.monotonic() - started
    except subprocess.TimeoutExpired:
        return command, 124, 'error: I exceeded the 180-second compile deadline.', time.monotonic() - started


def source_of(command):
    return next((word for word in command if re.search(r'\.c(pp)?$', word)), '?')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=os.environ.get('NANO_FAST_GATE_JOBS', os.cpu_count() or 4),
                        help='I bound compiler workers; NANO_FAST_GATE_JOBS sets my default.')
    parser.add_argument('--compilers', default='', help='I limit the run to a comma-separated subset of gcc,clang.')
    parser.add_argument('--optimize', default='-O2', help='I add this level so uninitialized analysis runs.')
    parser.add_argument('--test-profiles', action='store_true',
                        help='I also check Make test recipes and private union profiles at O3 with sanitizers.')
    parser.add_argument('--report', type=Path, help='I retain exact commands, results and uncovered test inputs.')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('I require at least one compiler worker.')
    available = detect_compilers()
    wanted = [name for name in (args.compilers.split(',') if args.compilers else available) if name]
    missing = [name for name in wanted if name not in available]
    if missing:
        print('I cannot find: ' + ', '.join(missing), file=sys.stderr)
        return 2
    failed = 0
    reports = []
    for name in wanted:
        started = time.monotonic()
        try:
            commands = compile_commands(available[name])
        except (ValueError, subprocess.SubprocessError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 2
        if not commands:
            print(f'{name}: I found no compile commands in the dry run.', file=sys.stderr)
            return 2
        # I replace any existing optimization level so GCC analyzes uninitialized use.
        commands = [[word for word in command if not re.fullmatch(r'-O[0-3sg]?', word)] + [args.optimize]
                    for command in commands]
        with tempfile.TemporaryDirectory(prefix='nanolang-fast-headers-') as headers:
            all_tests = {str(path.relative_to(ROOT)) for path in (ROOT / 'tests').rglob('*.c')}
            uncovered, skipped = sorted(all_tests), []
            if args.test_profiles:
                try:
                    stage_public_headers(headers)
                    extra, skipped = test_profiles(available[name], name, headers)
                except (ValueError, subprocess.SubprocessError, OSError) as error:
                    print(str(error), file=sys.stderr)
                    return 2
                covered = {source_of(command) for command in extra}
                uncovered = sorted(all_tests - covered)
                print(f'{name}: I cover {len(covered & all_tests)} test C files through Make/private '
                      f'profiles; {len(uncovered)} need a separate profile audit.', flush=True)
                for reason in sorted(set(skipped)):
                    print('  PROFILE SCOPE: ' + reason)
                commands.extend(extra)
            with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
                results = list(pool.map(run, commands))
        bad = [item for item in results if item[1] != 0]
        failed += len(bad)
        reports.append({'compiler': name, 'path': available[name],
                        'test_profiles_enabled': args.test_profiles,
                        'python_profile_audit_complete': False, 'uncovered_test_sources': uncovered,
                        'profile_scope': skipped, 'elapsed_seconds': time.monotonic() - started,
                        'results': [{'argv': command, 'returncode': status, 'diagnostics': output,
                                     'seconds': seconds} for command, status, output, seconds in results]})
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({'schema': 'nanolang/fast-gate-profiles@1',
                                              'compilers': reports}, indent=2) + '\n')
        print(f'{name} ({available[name]}): {len(results)} compiles, {len(bad)} failed, '
              f'{time.monotonic() - started:.1f}s')
        for command, _, output, _ in bad[:20]:
            errors = [line for line in output.splitlines() if 'error' in line][:3]
            print(f'  FAIL {source_of(command)}')
            for line in errors:
                print('    ' + line[:200])
    print('fast gate: ' + ('FAILED' if failed else 'ok'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
