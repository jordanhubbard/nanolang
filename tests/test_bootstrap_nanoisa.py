"""I refuse changed bootstrap inputs and artifacts before admitting a generation."""
from pathlib import Path
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from scripts.bootstrap_nanoisa import Bootstrap, digest, source_inputs


class BootstrapBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nano-bootstrap-boundary-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bootstrap = Bootstrap(self.root, 30)
        self.bootstrap.work = self.root / 'work'
        self.bootstrap.work.mkdir()
        for name in ('Makefile.gnu', 'scripts/bootstrap_nanoisa.py', 'tests/bootstrap_native_guard.py',
                     'src_nano/compiler.nano', 'modules/std/module.json', 'modules/std/host.c',
                     'bin/nano_virt', 'bin/nano_vm', 'bin/nanoisa', 'bin/nvm2c',
                     'bin/nano_aot_runtime.o', 'bin/nano_as_capture.so', 'host-library', 'bin/stage.nvm'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original bytes')
        host = self.root / 'host-library'
        self.bootstrap.manifest = {'sources': source_inputs(self.root),
                                   'tools': self.bootstrap.tool_inputs(),
                                   'hosts': {str(host): digest(host)},
                                   'artifacts': {'stage.nvm': digest(self.root / 'bin/stage.nvm')}}

    def test_source_tool_and_host_mutation_are_refused(self):
        self.bootstrap.check_inputs()
        for name in ('src_nano/compiler.nano', 'modules/std/host.c', 'bin/nano_vm', 'host-library'):
            path = self.root / name
            with self.subTest(input=name):
                path.write_text('changed bytes')
                with self.assertRaisesRegex(RuntimeError, 'unchanged'):
                    self.bootstrap.check_inputs()
                path.write_text('original bytes')
        self.bootstrap.check_inputs()

    def test_source_membership_changes_are_refused_but_host_cache_growth_is_allowed(self):
        cache = self.root / 'modules/std/.build/generated.c'
        cache.parent.mkdir()
        cache.write_text('host cache work')
        self.bootstrap.check_inputs()
        added = self.root / 'src_nano/new.nano'
        added.write_text('new source')
        with self.assertRaisesRegex(RuntimeError, 'source inputs'):
            self.bootstrap.check_inputs()
        added.unlink()
        (self.root / 'src_nano/compiler.nano').unlink()
        with self.assertRaisesRegex(RuntimeError, 'source inputs'):
            self.bootstrap.check_inputs()

    def test_native_codegen_marker_and_changed_artifact_are_refused(self):
        self.bootstrap.check_artifacts(['stage.nvm'])
        (self.root / 'bin/stage.nvm').write_text('replacement')
        with self.assertRaisesRegex(RuntimeError, 'recorded bootstrap artifact'):
            self.bootstrap.check_artifacts(['stage.nvm'])
        (self.bootstrap.work / 'unexpected-native-compiler').touch()
        with self.assertRaisesRegex(RuntimeError, 'native code generation'):
            self.bootstrap.check_inputs()

    def test_native_link_uses_effective_flags_for_covered_runtime(self):
        cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
        runtime = self.root / 'runtime.c'
        runtime.write_text('int answer(void) { return 42; }\n')
        covered = self.root / 'bin/nano_aot_runtime.o'
        flags = '-fprofile-arcs -ftest-coverage'
        subprocess.run([*cc, *shlex.split(flags), '-c', str(runtime), '-o', str(covered)],
                       check=True, capture_output=True, text=True, timeout=30)
        for module in ('compiler_support', 'nanoisa', 'std', 'file_source_inputs', 'file_source_catalog'):
            metadata = self.root / 'modules' / module / 'module.json'
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps({}))
        configurations = ({'LDFLAGS': flags},
                          {'LDFLAGS': '-lmissing_outer_link_flag', 'NANO_LDFLAGS': flags})
        for configuration in configurations:
            with self.subTest(configuration=configuration):
                self.bootstrap.env.pop('NANO_LDFLAGS', None)
                self.bootstrap.env.update(configuration)
                self.bootstrap.env['NANO_CC'] = shlex.join(cc)
                self.bootstrap.env['NANO_CFLAGS'] = '-O0'
                # I capture the real Stage1 configuration before VM generation.
                with patch.object(self.bootstrap, 'run', side_effect=InterruptedError('seed boundary')):
                    with self.assertRaisesRegex(InterruptedError, 'seed boundary'):
                        self.bootstrap.stage1()
                run = self.bootstrap.run

                def link_boundary(label, argv, env=None):
                    if label == 'probe-translate':
                        (self.bootstrap.work / 'probe.c').write_text(
                            'int answer(void); int main(void) { return answer() != 42; }\n')
                    elif label == 'probe-hello-compile':
                        raise InterruptedError('native boundary')
                    else:
                        try:
                            run(label, argv, env)
                        except RuntimeError as error:
                            log = Path(self.bootstrap.manifest['steps'][-1]['log'])
                            self.fail(str(error) + '\n' + log.read_text())

                # I isolate translation/smoke setup, but use the actual native
                # command, host compiler, covered runtime object and executable.
                with patch.object(self.bootstrap, 'run', side_effect=link_boundary):
                    with self.assertRaisesRegex(InterruptedError, 'native boundary'):
                        self.bootstrap.native('probe', self.root / 'bin/stage.nvm')
                subprocess.run([str(self.bootstrap.work / 'probe')], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
