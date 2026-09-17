#!/usr/bin/env python3
"""I compare two full compiler generations executed by NanoVM on one source pin."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(os.environ.get('NANOLANG_BOOTSTRAP_ROOT', Path(__file__).resolve().parents[1])).resolve()


class VMBytecodeBootstrap(unittest.TestCase):
    def test_full_compiler_fixed_point(self):
        budget = int(os.environ.get('NANOLANG_BOOTSTRAP_STAGE_TIMEOUT', '1800'))
        self.assertGreater(budget, 0)
        selected = os.environ.get('NANOLANG_BOOTSTRAP_EVIDENCE')
        evidence = Path(selected).resolve() if selected else Path(tempfile.mkdtemp(prefix='nanolang-vm-bootstrap-'))
        if selected:
            evidence.mkdir(parents=True, exist_ok=True)
            self.assertEqual(list(evidence.iterdir()), [], 'I require a fresh evidence directory.')
        print(f'I retain my bootstrap evidence at {evidence}', flush=True)
        env = os.environ.copy()
        env['NANO_AS_CAPTURE_HELPER'] = str(ROOT / 'bin/nano_as_capture.so')
        native_marker = evidence / 'unexpected-native-compiler'
        guarded_cc = evidence / 'guard-native-compiler'
        compiler_command = shlex.split(env.get('NANO_CC') or env.get('CC') or 'cc')
        self.assertTrue(compiler_command)
        guarded_cc.write_text(
            '#!' + sys.executable + '\nimport os, sys\nfrom pathlib import Path\n' +
            'if os.environ.get("NANOLANG_BOOTSTRAP_NO_CC") == "1":\n' +
            '    Path(' + repr(str(native_marker)) + ').write_text("I rejected a native compiler call.\\n")\n' +
            '    sys.exit(91)\n' +
            'command = ' + repr(compiler_command) + '\n' +
            'os.execvp(command[0], command + sys.argv[1:])\n')
        guarded_cc.chmod(0o755)
        env['CC'] = str(guarded_cc)
        env['NANO_CC'] = str(guarded_cc)
        env.pop('NANOLANG_BOOTSTRAP_NO_CC', None)
        manifest = {'root': str(ROOT), 'stages': {}, 'stage_timeout_seconds': budget,
                    'boundary': 'I execute VM-generation shadows as bytecode and reject native compiler calls. The product still contains its separate legacy C backend.'}

        def save():
            (evidence / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

        def git(*args):
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        def run(label, argv):
            started = time.monotonic()
            with (evidence / f'{label}.log').open('wb') as log:
                process = subprocess.Popen([str(a) for a in argv], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=budget)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    manifest['stages'][label] = {'argv': [str(a) for a in argv], 'timeout': True,
                                                 'elapsed_seconds': time.monotonic() - started}
                    save()
                    self.fail(f'I exceeded the diagnostic stage budget: {label}; see {evidence}.')
            manifest['stages'][label] = {'argv': [str(a) for a in argv], 'exit_code': code,
                                         'elapsed_seconds': time.monotonic() - started}
            save()
            self.assertEqual(code, 0, f'{label}: see {evidence / (label + ".log")}')

        def imports(path, label):
            dump = subprocess.check_output([ROOT / 'bin/nanoisa', 'dump', path], cwd=ROOT, text=True)
            (evidence / f'{label}.nasm').write_text(dump)
            libraries = sorted({shlex.split(line)[1] for line in dump.splitlines()
                                if line.startswith('.import ') and shlex.split(line)[1]})
            self.assertTrue(all(Path(p).is_absolute() for p in libraries))
            return {p: digest(p) for p in libraries}

        self.assertEqual(git('status', '--porcelain'), '', 'I require a clean pinned compiler source.')
        manifest['source_commit'] = git('rev-parse', 'HEAD')
        manifest['helper_sha256'] = digest(env['NANO_AS_CAPTURE_HELPER'])
        manifest['labels'] = {'seed': 'C-seed NanoVirt output, a different lowering implementation',
                              'stage1': 'VM execution of seed compiling the same source',
                              'stage2': 'VM execution of stage1 compiling the same source',
                              'comparison': 'raw stage1 versus stage2; no normalization'}
        save()
        source = 'src_nano/nanoc_v06.nano'
        seed, first, second = [evidence / f'{name}.nvm' for name in ('seed', 'stage1', 'stage2')]
        run('seed', [ROOT / 'bin/nano_virt', source, '--emit-nvm', '--strip-debug', '-o', seed])
        run('seed-verify', [ROOT / 'bin/nano_vm', '--verify-only', seed])
        hosts = imports(seed, 'seed')
        manifest['host_libraries'] = hosts
        save()
        # I retain the seed's compiler identity and immutable artifact cache.
        # Changing CC here would request new host-library builds before shadows.
        env['NANOLANG_BOOTSTRAP_NO_CC'] = '1'
        env['NANO_VM'] = str(ROOT / 'bin/nano_vm')
        for label, compiler, output in [('stage1', seed, first), ('stage2', first, second)]:
            run(label, [ROOT / 'bin/nano_vm', compiler, '--', source, '--emit-nvm', '-o', output])
            run(label + '-verify', [ROOT / 'bin/nano_vm', '--verify-only', output])
            self.assertEqual(imports(output, label), hosts, 'I preserve the exact pinned host closure.')
            manifest['stages'][label]['sha256'] = digest(output)
            manifest['stages'][label]['bytes'] = output.stat().st_size
            save()
        self.assertEqual(first.read_bytes(), second.read_bytes(), 'My compiler generations must match raw bytes.')
        manifest['raw_stage1_stage2_equal'] = True
        save()
        hello = evidence / 'hello.nano'
        hello.write_text('fn main() -> int { assert (== (+ 19 23) 42) return 0 }\n'
                         'shadow main { assert (== (main) 0) }\n')
        product = evidence / 'hello.nvm'
        run('hello-compile', [ROOT / 'bin/nano_vm', second, '--', hello, '--emit-nvm', '-o', product])
        run('hello-verify', [ROOT / 'bin/nano_vm', '--verify-only', product])
        run('hello-execute', [ROOT / 'bin/nano_vm', product])
        self.assertEqual({p: digest(p) for p in hosts}, hosts)
        self.assertEqual(digest(env['NANO_AS_CAPTURE_HELPER']), manifest['helper_sha256'])
        self.assertEqual(git('rev-parse', 'HEAD'), manifest['source_commit'])
        self.assertEqual(git('status', '--porcelain'), '')
        self.assertFalse(native_marker.exists(), 'I invoked a native compiler during VM generations.')
        manifest['vm_generations_native_compiler_calls'] = 0
        manifest['complete'] = True
        save()


if __name__ == '__main__':
    unittest.main(verbosity=2)
