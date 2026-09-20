#!/usr/bin/env python3
"""I retain direct managed-string qualification; I never prepare providers here."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
EMITTED = ['test_llvm_managed_strings', 'test_llvm_managed_decimal',
           'test_llvm_managed_format', 'test_managed_binary64_format',
           'test_managed_binary64_parse']
PHASES = {
    'original': EMITTED,
    'O2': EMITTED,
    'core-package': ['test_managed_string_core', 'test_managed_runtime_package'],
    'neighbors': ['test_llvm_scalar_globals', 'test_llvm_literal_strings',
                  'test_llvm_enum_scalars', 'test_llvm_generic_numeric',
                  'test_verifier_profiles'],
}
REQUIRED = ['clang', 'cc', 'opt', 'llc', 'llvm-as', 'lli', 'wasm-ld', 'node', 'wasmtime', 'python3']
PROVIDERS = ['bin/nanoisa', 'bin/nano_vm', 'bin/nvm2llvm', 'bin/nvm2wasm',
             'bin/nvm2c', 'obj/binary64_parser_vm', 'obj/scalar_global_lifetime',
             'obj/literal_string_aliases', 'obj/generic_numeric_bits',
             'obj/test_verifier_profiles']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def stop_group(process):
    # I bound both waits and report an unconfirmed cleanup as failure.
    outcome = {'pid': process.pid, 'errors': [], 'reaped': False, 'group_gone': False}
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except OSError as error:
            outcome['errors'].append(repr(error))
    send(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        outcome['term_wait_expired'] = True
    send(signal.SIGKILL)
    try:
        process.wait(timeout=5)
        outcome['reaped'] = True
    except subprocess.TimeoutExpired:
        outcome['kill_wait_expired'] = True
    until = time.monotonic() + 2
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            outcome['group_gone'] = True
            break
        except OSError as error:
            outcome['errors'].append(repr(error))
            break
        if time.monotonic() >= until:
            break
        time.sleep(0.02)
    outcome['confirmed'] = outcome['reaped'] and outcome['group_gone'] and not outcome['errors']
    return outcome


class Retention:
    def __init__(self, output, tools, sources, command_seconds):
        self.output, self.tools, self.sources = output, tools, sources
        self.command_seconds = command_seconds
        self.store = output / 'objects'
        self.store.mkdir()
        self.temporary = output / 'temporary'
        self.temporary.mkdir()
        self.counter = 0
        self.commands = []
        self.digest_cache = {}
        self.retention_failures = []
        self.raw_temporary = tempfile.TemporaryDirectory
        self.raw_run = subprocess.run

    def file_map(self, paths):
        result = {}
        for path in sorted(set(paths)):
            path = Path(path).absolute()
            if not path.is_file():
                result[str(path)] = {'missing': True}
                continue
            resolved = path.resolve()
            stat = resolved.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            key = (str(resolved), *identity)
            if key not in self.digest_cache:
                data = resolved.read_bytes()
                after = resolved.stat()
                if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise RuntimeError('I observed an input changing while hashing: ' + str(path))
                digest = hashlib.sha256(data).hexdigest()
                archive = self.store / digest
                if not archive.exists():
                    archive.write_bytes(data)
                self.digest_cache[key] = {'sha256': digest, 'bytes': len(data),
                    'object': 'objects/' + digest, 'resolved': str(resolved),
                    'stat_identity': list(identity)}
            result[str(path)] = self.digest_cache[key]
        return result

    def products(self, fresh=False):
        if fresh:
            self.digest_cache.clear()
        return self.file_map(p for p in self.temporary.rglob('*') if p.is_file())

    def inputs(self, fresh=False):
        if fresh:
            self.digest_cache.clear()
        providers = [p for directory in ('bin', 'obj')
                     for p in (ROOT / directory).rglob('*') if p.is_file()]
        return {'sources': self.file_map(self.sources),
                'providers': self.file_map(providers),
                'tools': self.file_map(self.tools)}

    def install(self):
        owner = self
        class KeptTemporary(self.raw_temporary):
            def __init__(self, *args, **kwargs):
                # These fixtures do not request a directory; I reject ambiguity.
                if args or kwargs.get('dir') is not None:
                    raise ValueError('I require keyword-only retained temporary directories')
                kwargs['dir'] = owner.temporary
                super().__init__(**kwargs)
                self._finalizer.detach()

            def cleanup(self):
                self._finalizer.detach()
        tempfile.TemporaryDirectory = KeptTemporary
        subprocess.run = self.run

    def restore(self):
        subprocess.run = self.raw_run
        tempfile.TemporaryDirectory = self.raw_temporary

    def run(self, args, **kwargs):
        # Every currently selected fixture uses run(capture_output=True).
        # I fail closed if a future caller requires another I/O contract.
        self.counter += 1
        directory = self.output / ('command-%05d' % self.counter)
        directory.mkdir()
        record = {'args': list(map(str, args)), 'cwd': str(kwargs.get('cwd', os.getcwd()))}
        self.commands.append(record)
        write_json(directory / 'request.json', record)
        allowed = {'capture_output', 'text', 'universal_newlines', 'timeout',
                   'env', 'cwd', 'check', 'input', 'encoding', 'errors'}
        if set(kwargs) - allowed or not kwargs.get('capture_output') or isinstance(args, (str, bytes)):
            self.retention_failures.append(str(directory) + ': unsupported subprocess contract')
            write_json(directory / 'status.json', {'error': 'unsupported subprocess contract'})
            raise ValueError('I require an explicit captured argv subprocess')
        before = self.inputs()
        write_json(directory / 'inputs-before.json', before)
        write_json(directory / 'products-before.json', self.products())
        env = kwargs.get('env', os.environ)
        record['environment'] = {k: env.get(k) for k in (
            'PATH', 'CC', 'NMS_NATIVE_CLANG_FLAGS', 'NMS_EMITTED_OPTIMIZATION',
            'NMS_CORE_LEAK_CHECK', 'NANOLANG_LLVM_VM', 'NANOLANG_LLVM_C',
            'NANOLANG_LLVM_TRANSLATOR', 'NANO_NVM2LLVM', 'NANO_LLC', 'NANO_WASM_LD', 'ASAN_OPTIONS', 'UBSAN_OPTIONS', 'LSAN_OPTIONS',
            'PYTHONPATH', 'NMS_RETAIN_TEMPORARY', 'SDKROOT', 'MACOSX_DEPLOYMENT_TARGET', 'DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH')}
        write_json(directory / 'request.json', record)
        started = time.monotonic()
        process = None
        problem = None
        code = None
        cleanup = None
        try:
            with (directory / 'stdout').open('wb') as stdout, (directory / 'stderr').open('wb') as stderr:
                process = subprocess.Popen(list(map(str, args)), cwd=kwargs.get('cwd'), env=env,
                    stdin=subprocess.PIPE if kwargs.get('input') is not None else subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, start_new_session=True)
                write_json(directory / 'process.json', {'pid': process.pid, 'pgid': process.pid})
                payload = kwargs.get('input')
                if isinstance(payload, str):
                    payload = payload.encode(kwargs.get('encoding') or 'utf-8')
                process.communicate(input=payload, timeout=min(
                    kwargs.get('timeout') or self.command_seconds, self.command_seconds))
                code = process.returncode
        except BaseException as error:
            problem = error
        finally:
            if process is not None:
                cleanup = stop_group(process)
                code = process.returncode
            write_json(directory / 'status.json', {'returncode': code,
                'seconds': time.monotonic() - started, 'cleanup': cleanup,
                'exception': None if problem is None else repr(problem),
                'timed_out': isinstance(problem, (subprocess.TimeoutExpired, TimeoutError))})
            write_json(directory / 'products-after.json', self.products())
            after = self.inputs()
            write_json(directory / 'inputs-after.json', after)
            write_json(directory / 'input-equality.json', {'equal': before == after})
        if problem is not None:
            self.retention_failures.append(str(directory) + ': ' + repr(problem))
            raise problem
        if cleanup is not None and not cleanup['confirmed']:
            self.retention_failures.append(str(directory) + ': unconfirmed cleanup')
            raise RuntimeError('I could not confirm bounded command cleanup')
        if before != after:
            self.retention_failures.append(str(directory) + ': immutable input drift')
            raise RuntimeError('I detected source/provider/tool changes; I stop this gate')
        stdout = (directory / 'stdout').read_bytes()
        stderr = (directory / 'stderr').read_bytes()
        if kwargs.get('text') or kwargs.get('universal_newlines') or kwargs.get('encoding'):
            encoding = kwargs.get('encoding') or 'utf-8'
            errors = kwargs.get('errors') or 'strict'
            stdout, stderr = stdout.decode(encoding, errors), stderr.decode(encoding, errors)
        result = subprocess.CompletedProcess(args, code, stdout, stderr)
        if kwargs.get('check'):
            result.check_returncode()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=PHASES)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tools', required=True, type=Path,
                        help='JSON: executables {name: absolute path}, libraries [absolute paths], native_clang_flags [strings]')
    parser.add_argument('--command-seconds', type=int, default=120)
    parser.add_argument('--phase-seconds', type=int, default=3600)
    options = parser.parse_args()
    if min(options.command_seconds, options.phase_seconds) <= 0:
        parser.error('I require positive bounds')
    output = options.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('I retain evidence outside the checkout')
    output.mkdir(parents=True, exist_ok=False)
    selection = json.loads(options.tools.read_text())
    native_flags = selection.get('native_clang_flags', [])
    if not isinstance(native_flags, list) or any(
            not isinstance(flag, str) or not flag or '\0' in flag for flag in native_flags):
        raise ValueError('I require native_clang_flags to be an array of nonempty NUL-free strings')
    executables = selection['executables']
    for name in REQUIRED:
        path = Path(executables[name])
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError('I require an absolute executable for ' + name)
    if Path(executables['python3']).resolve() != Path(sys.executable).resolve():
        raise ValueError('I require the selected Python to launch this runner')
    for name, value in executables.items():
        path = Path(value)
        if Path(name).name != name or not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError('I require an absolute executable for ' + name)
    libraries = list(map(Path, selection['libraries']))
    if not libraries or any(not p.is_absolute() or not p.is_file() for p in libraries):
        raise ValueError('I require explicit existing sanitizer/runtime library inventories')
    for name in PROVIDERS:
        if not (ROOT / name).is_file():
            raise ValueError('I require a prepared provider: ' + name)
    # I retain these read-only setup commands too; no compiler or gate runs here.
    def git_metadata(name, arguments):
        command = ['git', '-C', str(ROOT), *arguments]
        with (output / (name + '.stdout')).open('wb') as stdout, (output / (name + '.stderr')).open('wb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            error = None
            try:
                child.wait(timeout=30)
            except BaseException as caught:
                error = caught
            finally:
                cleanup = stop_group(child)
                write_json(output / (name + '.json'), {'args': command, 'returncode': child.returncode, 'cleanup': cleanup,
                           'exception': repr(error) if error else None})
            if error:
                raise error
            if not cleanup['confirmed']:
                raise RuntimeError('I could not confirm bounded metadata cleanup')
            if child.returncode:
                raise RuntimeError('I could not retain ' + name)
        return (output / (name + '.stdout')).read_bytes()
    tracked = git_metadata('tracked-paths', ['ls-files', '-z'])
    git_metadata('head', ['rev-parse', 'HEAD'])
    git_metadata('worktree-status', ['status', '--porcelain=v1'])
    tracked_names = [os.fsdecode(p) for p in tracked.split(b'\0') if p]
    # Documentation archives are not compiler/fixture inputs. All other tracked
    # paths remain covered, plus the exact acceptance contract and roadmap.
    sources = [ROOT / p for p in tracked_names if not p.startswith('docs/') or
               p in ('docs/MANAGED_STRING_FINAL_ACCEPTANCE.md', 'docs/ROADMAP.md')]
    included_names = {str(p.relative_to(ROOT)) for p in sources}
    write_json(output / 'source-scope.json', {'included': sorted(included_names),
        'excluded_documentation': [p for p in tracked_names if p not in included_names],
        'policy': 'all tracked non-docs paths plus acceptance contract and roadmap'})
    aliases = output / 'tool-bin'
    aliases.mkdir()
    for name, path in executables.items():
        if Path(name).name != name:
            raise ValueError('I require simple tool names')
        (aliases / name).symlink_to(Path(path).resolve())
    child_hook = output / 'python-retention'
    child_hook.mkdir()
    (child_hook / 'sitecustomize.py').write_text('''# I archive Python CLI intermediates before their real cleanup.
import json
import os
from pathlib import Path
import shutil
import tempfile
_original = tempfile.TemporaryDirectory
class _Retained(_original):
    def cleanup(self):
        source = Path(self.name)
        if source.exists():
            target = Path(tempfile.mkdtemp(prefix='cli-archive-', dir=os.environ['NMS_RETAIN_TEMPORARY']))
            for path in source.rglob('*'):
                if path.is_file():
                    destination = target / 'files' / path.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
            (target / 'origin.json').write_text(json.dumps({'original': str(source), 'pid': os.getpid()}))
        super().cleanup()
tempfile.TemporaryDirectory = _Retained
''')
    os.environ.update(PYTHONPATH=str(child_hook) + os.pathsep + str(ROOT),
        NMS_RETAIN_TEMPORARY=str(output / 'temporary'), PATH=str(aliases) + os.pathsep + os.environ['PATH'],
        CC=shlex.join([executables['clang'], *native_flags]),
        NMS_NATIVE_CLANG_FLAGS=shlex.join(native_flags), NMS_WASM_CC='clang',
        NMS_RUNTIME_CLANG='clang', NMS_RUNTIME_OPT='opt', NMS_CORE_LEAK_CHECK='1',
        NANOLANG_LLVM_VM=str(ROOT / 'bin/nano_vm'),
        NANOLANG_LLVM_C=str(ROOT / 'bin/nvm2c'),
        NANOLANG_LLVM_TRANSLATOR=str(ROOT / 'bin/nvm2llvm'),
        NANO_NVM2LLVM=str(ROOT / 'bin/nvm2llvm'), NANO_LLC='llc', NANO_WASM_LD='wasm-ld',
        NMS_EMITTED_OPTIMIZATION='O2' if options.phase == 'O2' else 'none',
        ASAN_OPTIONS='detect_leaks=1:abort_on_error=1', LSAN_OPTIONS='',
        UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    retention = Retention(output, [Path(p) for p in executables.values()] + libraries +
                          list(aliases.iterdir()) + [child_hook / 'sitecustomize.py'],
                          sources, options.command_seconds)
    write_json(output / 'selection.json', selection)
    write_json(output / 'host.json', {'uname': list(platform.uname()), 'python': sys.version,
               'phase': options.phase, 'command_seconds': options.command_seconds,
               'phase_seconds': options.phase_seconds,
               'input_maps': 'phase endpoints freshly hashed; command maps stat-keyed cache',
               'cache_key': 'resolved path, device, inode, size, mtime_ns, ctime_ns'})
    before = retention.inputs(fresh=True)
    write_json(output / 'inputs-before.json', before)
    retention.install()
    result = None
    failure = None
    started = time.monotonic()
    def deadline(signum, frame):
        raise TimeoutError('I reached the qualification phase deadline')
    previous_term = signal.signal(signal.SIGTERM, deadline)
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(options.phase_seconds)
    try:
        for name in REQUIRED:
            retention.run([str(aliases / name), '--version'], capture_output=True, text=True, timeout=30, check=True)
        with (output / 'unittest.log').open('w') as stream, \
             (output / 'phase-stdout').open('w') as stdout, \
             (output / 'phase-stderr').open('w') as stderr, \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.' + name for name in PHASES[options.phase]])
            def test_ids(node):
                if isinstance(node, unittest.TestSuite):
                    return [name for item in node for name in test_ids(item)]
                return [node.id()]
            write_json(output / 'selected-tests.json', test_ids(suite))
            result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(suite)
    except BaseException:
        failure = traceback.format_exc()
        (output / 'exception.txt').write_text(failure)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        signal.signal(signal.SIGTERM, previous_term)
        retention.restore()
        after = retention.inputs(fresh=True)
        write_json(output / 'inputs-after.json', after)
        write_json(output / 'products-final.json', retention.products(fresh=True))
        passed = bool(result and result.wasSuccessful() and not result.skipped and
                      not result.expectedFailures and not failure and not retention.retention_failures and before == after)
        write_json(output / 'status.json', {'passed': passed, 'seconds': time.monotonic() - started,
            'tests_run': result.testsRun if result else 0, 'exception': failure,
            'failures': len(result.failures) if result else None,
            'errors': len(result.errors) if result else None,
            'skipped': result.skipped if result else [], 'inputs_equal': before == after,
            'commands': retention.counter, 'retention_failures': retention.retention_failures})
        reports = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in output.rglob('*') if p.is_file() and
                   p.parts[len(output.parts)] not in ('objects', 'temporary', 'tool-bin')}
        write_json(output / 'report-sha256.json', reports)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
