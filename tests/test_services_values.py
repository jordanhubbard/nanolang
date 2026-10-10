"""I exercise mixed private service lifetimes with real hosts and fault hooks."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ServicesValues(unittest.TestCase):
    def test_real_lifetimes_and_faults(self):
        artifacts = Path(tempfile.mkdtemp(prefix='nano-services-values-'))
        print(f'I retain mixed value evidence at {artifacts}', flush=True)
        compiler = shlex.split(os.environ.get('CC', 'cc'))
        flags = ['-std=c11', '-D_DEFAULT_SOURCE', '-Wall', '-Wextra', '-Werror',
                 '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1', UBSAN_OPTIONS='halt_on_error=1')
        commands = []
        def run(args):
            completed = subprocess.run(args, cwd=ROOT, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            commands.append({'argv': args, 'returncode': completed.returncode, 'output': completed.stdout})
            (artifacts / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
            self.assertEqual(completed.returncode, 0, completed.stdout)
            return completed.stdout
        sources = ['nsi_services_values', 'nsi_file_values', 'nsi_file',
                   'nsi_socket_values', 'nsi_socket', 'nsi_cap']
        for instrument in (False, True):
            objects = []
            for source in sources:
                obj = str(artifacts / f'{source}-{instrument}.o')
                hooks = (['-include', 'tests/services_values_alloc.h',
                          '-Dcalloc=services_test_calloc', '-Dfree=services_test_free',
                          '-Dclose=services_test_close', '-Dfclose=services_test_fclose']
                         if instrument else [])
                run([*compiler, *flags, *hooks, '-c', f'src/{source}.c', '-o', obj])
                objects.append(obj)
            exe = str(artifacts / f'values-{instrument}')
            run([*compiler, *flags, *(['-DSERVICES_ALLOC_TEST'] if instrument else []),
                 'tests/test_nsi_services_values.c', *objects, '-o', exe])
            output = run([exe]);self.assertIn('PASS', output);print(output.strip(), flush=True)
