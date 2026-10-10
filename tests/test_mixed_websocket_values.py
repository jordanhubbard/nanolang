"""I qualify mixed WebSocket value ownership and the actual transport separately."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from tests import test_websocket_dispatch as dispatch

ROOT = Path(__file__).resolve().parents[1]
BASE = ['nsi_services_values', 'nsi_websocket_values', 'nsi_file_values',
        'nsi_file', 'nsi_socket_values', 'nsi_socket', 'nsi_cap']

class MixedWebSocketValues(unittest.TestCase):
    server = dispatch.WebSocketDispatch.server
    bad_reply = False

    @classmethod
    def setUpClass(cls):
        cls.artifacts = Path(tempfile.mkdtemp(prefix='nano-mixed-websocket-values-'))
        print(f'I retain mixed WebSocket value evidence at {cls.artifacts}', flush=True)
        cls.compiler = shlex.split(os.environ.get('CC', 'cc'))
        cls.flags = ['-std=c11', '-D_DEFAULT_SOURCE', '-Wall', '-Wextra', '-Werror',
                     '-g', '-O1', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        cls.commands = []

    def command(self, name, args):
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=90,
            env=dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                     UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',
                     NANOLANG_RESOLVER=str(ROOT / 'bin/nano-resolver')))
        self.commands.append({'name': name, 'argv': args, 'returncode': result.returncode,
                              'stdout': result.stdout, 'stderr': result.stderr})
        (self.artifacts / 'commands.json').write_text(json.dumps(self.commands, indent=2)+'\n')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result.stdout

    def test_controlled_transport_lifetimes_and_faults(self):
        for instrument in (False, True):
            objects = []
            for source in BASE:
                output = self.artifacts / f'{source}-{instrument}.o'
                hooks = (['-include', 'tests/mixed_websocket_values_hooks.h',
                          '-Dcalloc=mixed_values_calloc', '-Dfree=mixed_values_free',
                          '-Dfclose=mixed_values_fclose'] if instrument else [])
                self.command(output.stem, [*self.compiler, *self.flags, *hooks,
                    '-c', f'src/{source}.c', '-o', str(output)])
                objects.append(str(output))
            exe = self.artifacts / f'controlled-{instrument}'
            self.command(exe.name+'-build', [*self.compiler, *self.flags,
                *(['-DMIXED_VALUES_INSTRUMENT'] if instrument else []),
                'tests/test_nsi_mixed_websocket_values.c', *objects, '-o', str(exe)])
            print(self.command(exe.name, [str(exe)]).strip(), flush=True)

    def real_binary(self):
        exe = self.artifacts / 'real'
        if not exe.exists():
            sources = [f'src/{s}.c' for s in BASE + ['nsi_websocket_transport',
                       'nsi_websocket_protocol', 'nsi_socket_resolver', 'utf8']]
            crypto = shlex.split(subprocess.check_output(['pkg-config', '--cflags', '--libs', 'libcrypto'], text=True))
            self.command('real-build', [*self.compiler, *self.flags,
                'tests/mixed_websocket_values_probe.c', *sources, *crypto, '-o', str(exe)])
        return exe

    def test_real_denied_network(self):
        self.command('real-denied', [str(self.real_binary()), 'ws://127.0.0.1:1', 'denied'])

    def test_installed_denied_network(self):
        prefix = self.artifacts / 'installed'
        self.command('install', ['make', '-f', 'Makefile.gnu', 'install-services-public-runtime',
                     f'PREFIX={prefix}', f'CC={shlex.join(self.compiler)}'])
        relocated = self.artifacts / 'relocated'
        shutil.move(prefix, relocated)
        source = self.artifacts / 'installed-probe.c'
        source.write_text((ROOT / 'tests/mixed_websocket_values_probe.c').read_text().replace(
            '#include "../src/nsi_services_values.h"', '#include <nanolang/services/nsi_services_values.h>'))
        crypto = shlex.split(subprocess.check_output(['pkg-config', '--libs', 'libcrypto'], text=True))
        exe = self.artifacts / 'installed-probe'
        self.command('installed-build', [*self.compiler, *self.flags, '-I'+str(relocated / 'include'),
            str(source), str(relocated / 'lib/libnano_services_runtime.a'), *crypto, '-o', str(exe)])
        self.command('installed-denied', [str(exe), 'ws://127.0.0.1:1', 'denied'])

    def test_real_peer_lifetimes(self):
        exe = self.real_binary()
        port, messages = self.server()
        for host, mode in [('127.0.0.1', 'normal'), ('localhost', 'invalid-close'),
                           ('127.0.0.1', 'borrowed'), ('127.0.0.1', 'unhandled')]:
            self.command(mode, [str(exe), f'ws://{host}:{port}?mode=test', mode])
        self.assertEqual(messages, [b'a\0b'] * 10)
