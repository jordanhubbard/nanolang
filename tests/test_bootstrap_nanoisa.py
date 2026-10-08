"""I refuse changed bootstrap inputs and artifacts before admitting a generation."""
from pathlib import Path
import tempfile
import unittest
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


if __name__ == '__main__':
    unittest.main()
