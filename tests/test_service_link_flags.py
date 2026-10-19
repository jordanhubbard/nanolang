"""I retain instrumentation link flags when publishing relocated service runtimes."""
import os
from pathlib import Path
import shlex
import shutil
import sys
import unittest
from tests import test_websocket_service_drivers as websocket

ROOT = websocket.ROOT


class ServiceLinkFlags(unittest.TestCase):
    setUp = websocket.WebSocketServiceDrivers.setUp
    run_command = websocket.WebSocketServiceDrivers.run_command

    def test_cleanup_accepts_only_regular_nonexecutable_coverage_metadata(self):
        command = [sys.executable, '-c', 'pass']
        for extension in ('gcno', 'gcda'):
            (self.work / ('.nano-publish.A1b2C3-program.' + extension)).write_bytes(b'metadata')
        self.run_command(command)
        controls = [('.nano-file.A1b2C3', 'directory'),
                    ('.nano-publish.A1b2C3', 'file'),
                    ('.nano-publish.A1b2C3-other.gcno', 'file'),
                    ('.nano-file.A1b2C3-program.gcno', 'file'),
                    ('.nano-publish.D4e5F6-program.gcno', 'directory'),
                    ('.nano-publish.D4e5F6-program.gcda', 'symlink'),
                    ('.nano-publish.D4e5F6-program.gcno', 'executable')]
        for name, kind in controls:
            with self.subTest(name=name, kind=kind):
                path = self.work / name
                if kind == 'directory':
                    path.mkdir()
                elif kind == 'symlink':
                    path.symlink_to(self.work / '.nano-publish.A1b2C3-program.gcda')
                else:
                    path.write_bytes(b'leftover')
                    if kind == 'executable':
                        path.chmod(0o700)
                try:
                    with self.assertRaises(AssertionError):
                        self.run_command(command)
                finally:
                    if kind == 'directory':
                        path.rmdir()
                    else:
                        path.unlink()
        self.run_command(command)

    def test_relocated_instrumented_runtime_and_override_precedence(self):
        self.source.write_text(websocket.DECL + websocket.MESSAGE + websocket.MAIN)
        prefix = self.work / 'installed'
        self.run_command(['make', '-f', 'Makefile.gnu', '-C', ROOT,
                          'PREFIX=' + str(prefix), 'install-websocket-public-runtime'])
        (prefix / 'bin').mkdir()
        for name in ('nanoc_c', 'nano_virt'):
            shutil.copy2(ROOT / 'bin' / name, prefix / 'bin' / name)
        moved = self.work / 'relocated'
        prefix.rename(moved)
        instrumented = self.work / 'instrumented'
        instrumented.mkdir()
        obj = instrumented / 'websocket_runtime.o'
        compiler = shlex.split(self.env['NANO_CC'])
        self.run_command([*compiler, '--coverage', '-std=c11', '-D_GNU_SOURCE',
                          '-I' + str(ROOT / 'src'), '-c',
                          ROOT / 'src/nanoisa/websocket_runtime.c', '-o', obj])
        # I replace a required member only in the private installed archive.
        self.run_command([*shlex.split(os.environ.get('AR', 'ar')), 'rcs',
                          moved / 'lib/libnano_websocket_runtime.a', obj])
        env = dict(self.env)
        # I retain other runtime requirements, including sanitizer link flags.
        inherited = env.get('NANO_LDFLAGS', env.get('LDFLAGS', ''))
        coverage_flags = inherited + ' --coverage'
        for variable in ('NANO_ROOT', 'NANOLANG_ROOT', 'NANO_LDFLAGS'):
            env.pop(variable, None)
        env['LDFLAGS'] = coverage_flags
        for name in ('nanoc_c', 'nano_virt'):
            with self.subTest(driver=name):
                native = self.work / (name + '.installed')
                command = [moved / 'bin' / name, self.source,
                           '--allow-websocket-connections', '-o', native]
                self.run_command(command, env=env)
                self.run_command([native, '--allow-websocket-connections'], 7, env=env)
                # I keep a nonempty explicit override ahead of ambient build flags.
                explicit = dict(env, NANO_LDFLAGS=coverage_flags,
                                LDFLAGS='-lnano_missing_link_flag_sentinel')
                self.run_command(command, env=explicit)
                self.run_command([native, '--allow-websocket-connections'], 7, env=explicit)
                # An explicit empty override must omit coverage and fail to link;
                # I also retain the previously published executable on failure.
                previous = native.read_bytes()
                failed = self.run_command(command, 1, env=dict(env, NANO_LDFLAGS=''))
                self.assertRegex(failed.stderr, r'(?i)(gcov|coverage|profile)')
                self.assertEqual(native.read_bytes(), previous)
                self.run_command([native, '--allow-websocket-connections'], 7, env=env)


if __name__ == '__main__':
    unittest.main()
