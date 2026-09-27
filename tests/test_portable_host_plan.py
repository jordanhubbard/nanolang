"""I retain bounded declaration-query gates; I never execute their bytecode."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'read_query': 'src/nanoisa/portable_host_plan.c',
    'read_decode': 'src/nanovm/vm_decode.c',
    'read_stack': 'src/nanoisa/verifier.c',
    'read_types': 'src/nanoisa/verifier_types.c',
}


class PortableHostPlan(unittest.TestCase):
    def retain(self, path):
        path = Path(path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        target = self.store / digest
        if not target.exists():
            shutil.copyfile(path, target)
        return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
                'archive': str(target)}

    def inventory(self):
        return {str(p): self.retain(p) for p in self.inputs}

    def write(self, name, value):
        (self.artifacts / name).write_text(json.dumps(value, indent=2) + '\n')

    def command(self, argv):
        number = self.index
        self.index += 1
        argv = list(map(str, argv))
        status = {'argv': argv, 'cwd': str(ROOT), 'timeout': False,
                  'cleanup_complete': True, 'returncode': None}
        log = self.artifacts / f'{number:02d}.log'
        self.write(f'{number:02d}-command.json', status)
        try:
            with log.open('wb') as output:
                process = subprocess.Popen(argv, cwd=ROOT, env=self.env,
                                           stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    status['returncode'] = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    status['timeout'] = True
                    for sig in (signal.SIGTERM, signal.SIGKILL):
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            continue
                    status['returncode'] = 124
                    status['child_returncode'] = process.poll()
                    status['cleanup_complete'] = process.poll() is not None
                # A compiler may leave descendants after its own exit.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    status['cleanup_complete'] = False
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except OSError as error:
            status['error'] = repr(error)
            status['cleanup_complete'] = False
        finally:
            self.write(f'{number:02d}-status.json', status)
            products = {p.name: self.retain(p) for p in sorted(self.work.iterdir()) if p.is_file()}
            self.write(f'{number:02d}-products.json', products)
        self.assertEqual(status['returncode'], 0, (status, log.read_text(errors='replace')))
        self.assertTrue(status['cleanup_complete'], status)
        return log.read_text(errors='replace')

    def test_owned_declarations_and_allocation_boundaries(self):
        requested = os.environ.get('PORTABLE_READ_ARTIFACTS')
        self.artifacts = Path(requested).resolve() if requested else Path(tempfile.mkdtemp(prefix='nano-portable-read-'))
        if requested:
            self.artifacts.mkdir(parents=True, exist_ok=False)
        self.store = self.artifacts / 'objects'
        self.store.mkdir()
        self.work = self.artifacts / 'products'
        self.work.mkdir()
        self.index = 0
        compiler = shlex.split(os.environ.get('PORTABLE_READ_CC', 'cc'))
        self.assertTrue(compiler)
        executable = shutil.which(compiler[0])
        self.assertIsNotNone(executable)
        compiler[0] = str(Path(executable).resolve())
        flags = shlex.split(os.environ.get('PORTABLE_READ_CFLAGS', ''))
        common = [*compiler, '-std=c11', '-D_DEFAULT_SOURCE', '-O1', '-g',
                  '-Wall', '-Wextra', '-Werror', '-Isrc/nanoisa', *flags]
        providers = [str((ROOT / p).resolve()) for p in shlex.split(os.environ['PORTABLE_READ_OBJECTS'])]
        self.assertTrue(providers)
        ldflags = shlex.split(os.environ.get('PORTABLE_READ_LDFLAGS', '-lm -lcrypto'))
        self.env = dict(os.environ, ASAN_OPTIONS=asan_options("halt_on_error=1"),
                        UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', LSAN_OPTIONS='')
        self.inputs = sorted(set(
            [p.resolve() for p in (ROOT / 'src').rglob('*') if p.is_file() and p.suffix in ('.c', '.h', '.inc')]
            + [ROOT / 'Makefile.gnu', ROOT / 'tests/nanoisa/test_portable_host_plan.c',
               Path(__file__).resolve(), Path(compiler[0]), Path(sys.executable).resolve()]
            + [Path(p) for p in providers]), key=str)
        self.write('configuration.json', {'compiler': compiler, 'flags': flags,
                   'providers': providers, 'ldflags': ldflags,
                   'instrumented_translation_units': SOURCES,
                   'sanitizer_environment': {k: self.env[k] for k in ('ASAN_OPTIONS', 'UBSAN_OPTIONS', 'LSAN_OPTIONS')},
                   'limits': 'I hash actual C/header/include sources and named providers/compiler/Python at phase endpoints; I do not claim all transitive system tools.'})
        print(f'I retain query evidence at {self.artifacts}', flush=True)
        before = self.inventory()
        self.write('inputs-before.json', before)
        try:
            for instrument in (True, False):
                mode = 'instrumented' if instrument else 'linked'
                objects = []
                for prefix, source in SOURCES.items():
                    obj = self.work / f'{mode}-{prefix}.o'
                    hooks = [f'-D{name}={prefix}_{name}' for name in ('malloc', 'calloc', 'realloc', 'free')] if instrument else []
                    self.command([*common, *hooks, '-c', source, '-o', obj])
                    objects.append(obj)
                binary = self.work / mode
                self.command([*common, *([] if instrument else ['-DREAD_LINKED']),
                              'tests/nanoisa/test_portable_host_plan.c', *objects,
                              *providers, *ldflags, '-o', binary])
                output = self.command([binary])
                self.assertIn('private portable read-text checks; no bytecode or host execution.', output)
                print(output, end='', flush=True)
        finally:
            after = self.inventory()
            self.write('inputs-after.json', after)
            self.write('phase-summary.json', {'commands': self.index, 'inputs_equal': before == after})
            self.assertEqual(before, after, 'I require equal source/provider/compiler bytes at the phase endpoints.')


if __name__ == '__main__':
    unittest.main()
