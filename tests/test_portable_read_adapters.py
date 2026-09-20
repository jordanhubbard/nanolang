"""I retain private real-reader gates; no NanoISA profile is selected."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['src/nanoisa/portable_read_host.c',
           'src/nanoisa/portable_read_managed.c', 'src/nanoisa/managed_strings.c']


class PortableReadAdapters(unittest.TestCase):
    def dump(self, name, value):
        (self.artifacts / name).write_text(json.dumps(value, indent=2) + '\n')

    def retain(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        target = self.store / digest
        if not target.exists():
            target.write_bytes(data)
        return {'path': str(path), 'sha256': digest, 'bytes': len(data),
                'archive': str(target)}

    def products(self):
        return {str(p.relative_to(self.work)): self.retain(p)
                for p in sorted(self.work.rglob('*')) if p.is_file()}

    def inventory(self):
        return {str(p): self.retain(p) for p in self.inputs}

    def command(self, argv):
        index = self.index
        self.index += 1
        argv = list(map(str, argv))
        status = {'argv': argv, 'cwd': str(ROOT), 'returncode': None,
                  'timeout': False, 'leader_reaped': False,
                  'group_disappeared': False, 'signals': [], 'errors': []}
        self.dump(f'{index:03d}-command.json', status)
        self.dump(f'{index:03d}-products-before.json', self.products())
        process = None
        log = self.artifacts / f'{index:03d}.log'
        with log.open('wb') as output:
            try:
                process = subprocess.Popen(argv, cwd=ROOT, env=self.env,
                                           stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    status['timeout'] = True
            except Exception as error:
                status['errors'].append(repr(error))
            finally:
                if process is not None:
                    def exists():
                        try:
                            os.killpg(process.pid, 0)
                            return True
                        except ProcessLookupError:
                            return False
                        except OSError as error:
                            status['errors'].append(repr(error))
                            return True
                    for sig in (signal.SIGTERM, signal.SIGKILL):
                        if not exists():
                            break
                        try:
                            os.killpg(process.pid, sig)
                            status['signals'].append(sig.name)
                        except ProcessLookupError:
                            pass
                        except OSError as error:
                            status['errors'].append(repr(error))
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            process.poll()
                            if not exists():
                                break
                            time.sleep(0.05)
                    status['returncode'] = process.poll()
                    status['leader_reaped'] = process.returncode is not None
                    status['group_disappeared'] = not exists()
                self.dump(f'{index:03d}-status.json', status)
                self.dump(f'{index:03d}-products-after.json', self.products())
        self.assertFalse(status['errors'], status)
        self.assertFalse(status['timeout'], status)
        self.assertTrue(status['leader_reaped'], status)
        self.assertTrue(status['group_disappeared'], status)
        self.assertFalse(status['signals'], status)
        self.assertEqual(status['returncode'], 0, (status, log.read_text(errors='replace')))
        return log.read_text(errors='replace')

    def test_private_c_and_typed_llvm_boundaries(self):
        requested = os.environ.get('PORTABLE_ADAPTER_ARTIFACTS')
        self.artifacts = Path(requested).resolve() if requested else Path(tempfile.mkdtemp(prefix='nano-read-adapter-'))
        if requested:
            self.artifacts.mkdir(parents=True, exist_ok=False)
        self.store = self.artifacts / 'objects'
        self.work = self.artifacts / 'products'
        self.store.mkdir()
        self.work.mkdir()
        self.index = 0
        cc = shlex.split(os.environ.get('PORTABLE_ADAPTER_CC', 'cc'))
        llvm = shlex.split(os.environ.get('PORTABLE_ADAPTER_CLANG', 'clang'))
        for compiler in (cc, llvm):
            self.assertTrue(compiler)
            found = shutil.which(compiler[0])
            self.assertIsNotNone(found)
            compiler[0] = str(Path(found).resolve())
        flags = shlex.split(os.environ.get('PORTABLE_ADAPTER_CFLAGS', ''))
        llvm_flags = shlex.split(os.environ.get('PORTABLE_ADAPTER_LLVM_FLAGS', ''))
        extras = json.loads(os.environ.get('PORTABLE_ADAPTER_EXTRA_TOOLS', '{}'))
        self.assertIsInstance(extras, dict)
        self.assertTrue(all(isinstance(v, str) and Path(v).is_file() for v in extras.values()))
        self.env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                        UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', LSAN_OPTIONS='')
        self.inputs = sorted(set(
            [p.resolve() for p in (ROOT / 'src').rglob('*') if p.is_file() and p.suffix in ('.c', '.h', '.inc')]
            + [ROOT / 'Makefile.gnu', ROOT / 'docs/NANOISA_PORTABLE_READ_TEXT_ADAPTERS.md',
               Path(__file__).resolve(), Path(sys.executable).resolve(), Path(cc[0]), Path(llvm[0])]
            + [p.resolve() for p in (ROOT / 'tests/nanoisa').glob('*portable_read*') if p.is_file()]
            + [Path(p).resolve() for p in extras.values()]), key=str)
        self.dump('configuration.json', {'cc': cc, 'flags': flags, 'clang': llvm,
                  'llvm_flags': llvm_flags, 'extra_tools': extras,
                  'rebuilt_sources': SOURCES, 'allocator_hooks': SOURCES,
                  'io_hooks': SOURCES[0], 'unhooked_comparison': 'All three TUs rebuilt without hooks.',
                  'environment': {k: self.env[k] for k in ('ASAN_OPTIONS', 'UBSAN_OPTIONS', 'LSAN_OPTIONS')},
                  'scope': 'Direct C and LLVM ABI forwarding only; no NanoISA emission or public profile. Inventoried phase endpoint bytes, not all transitive system tools.'})
        print(f'I retain private adapter evidence at {self.artifacts}', flush=True)
        before = self.inventory()
        self.dump('inputs-before.json', before)
        try:
            self.command([*cc, '--version'])
            self.command([*llvm, '--version'])
            # I retain the actual frontend target, including SDK normalization.
            probe = self.work / 'target-probe.c'
            probe_ir = self.work / 'target-probe.ll'
            probe.write_text('void portable_read_target_probe(void) {}\n')
            self.command([*llvm, *llvm_flags, '-std=c11', '-Wall', '-Wextra',
                          '-Werror', '-O0', '-S', '-emit-llvm', probe, '-o', probe_ir])
            declarations = [line for line in probe_ir.read_text().splitlines()
                            if line.startswith('target triple')]
            self.assertEqual(len(declarations), 1, declarations)
            self.assertRegex(declarations[0], r'^target triple = "[a-zA-Z0-9_.-]+"$')
            ir = self.work / 'typed-read.ll'
            ir.write_text(declarations[0] + '\n' +
                          (ROOT / 'tests/nanoisa/portable_read_link.ll').read_text())
            llvm_objects = {}
            for optimization in ('O0', 'O2'):
                obj = self.work / f'typed-{optimization}.o'
                self.command([*llvm, *llvm_flags, '-Werror', '-' + optimization,
                              '-c', ir, '-o', obj])
                llvm_objects[optimization] = obj
            common = [*cc, '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-O1', '-g',
                      '-Wall', '-Wextra', '-Werror', '-Isrc/nanoisa', *flags]
            for observed in (True, False):
                mode = 'observed' if observed else 'unhooked'
                objects = []
                for number, source in enumerate(SOURCES):
                    obj = self.work / f'{mode}-{number}.o'
                    hooks = ['-DREAD_ALLOC_HOOKS', '-include', 'tests/nanoisa/portable_read_hooks.h'] if observed else []
                    if observed and number == 0:
                        hooks += ['-DREAD_IO_HOOKS']
                    self.command([*common, *hooks, '-c', source, '-o', obj])
                    objects.append(obj)
                for route in ('C', 'O0', 'O2'):
                    name = mode + '-' + route
                    binary = self.work / name
                    fixture_flags = ['-DREAD_OBSERVED'] if observed else []
                    link = []
                    if route != 'C':
                        fixture_flags += ['-DREAD_LLVM']
                        link = [llvm_objects[route]]
                    self.command([*common, *fixture_flags, 'tests/nanoisa/test_portable_read_adapters.c',
                                  *objects, *link, '-lm', '-o', binary])
                    files = self.work / (name + '-files')
                    files.mkdir()
                    output = self.command([binary, files])
                    self.assertIn('private read-text adapter checks; no bytecode admission.', output)
                    if observed:
                        self.assertIn('4 measured call allocations', output)
                    print(output, end='', flush=True)
        finally:
            after = self.inventory()
            self.dump('inputs-after.json', after)
            self.dump('phase-summary.json', {'commands': self.index, 'inputs_equal': before == after})
            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
