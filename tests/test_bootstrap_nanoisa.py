"""I refuse changed bootstrap inputs and artifacts before admitting a generation."""
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts.bootstrap_nanoisa import Bootstrap, HOST_MODULES, digest, source_inputs


class BootstrapBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nano-bootstrap-boundary-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bootstrap = Bootstrap(self.root, 30)
        self.bootstrap.work = self.root / 'work'
        self.bootstrap.work.mkdir()
        for name in ('Makefile.gnu', 'scripts/bootstrap_nanoisa.py', 'tests/bootstrap_native_guard.py',
                     'src_nano/compiler.nano', 'src/engine.inc', 'modules/std/module.json', 'modules/std/host.c',
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
        for name in ('src_nano/compiler.nano', 'src/engine.inc', 'modules/std/host.c', 'bin/nano_vm', 'host-library'):
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
        for module in HOST_MODULES:
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

    def test_file_product_host_closure_is_pinned_and_generated_c_is_refused(self):
        cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or 'cc')
        self.bootstrap.env['NANO_CC'] = shlex.join(cc)
        guard_source = Path(__file__).with_name('bootstrap_native_guard.py')
        (self.root / 'tests/bootstrap_native_guard.py').write_text(guard_source.read_text())
        for name in HOST_MODULES:
            directory = self.root / 'modules' / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'module.json').write_text(json.dumps({'c_sources': [name + '.c']}))
            (directory / (name + '.c')).write_text('int answer(void) { return 42; }\n')
        library = self.root / 'modules/file_product/.build/.nano-gen-pinned/libfile_product.dylib'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'host artifact')

        def seed_boundary(label, argv, env=None):
            if label == 'seed':
                Path(argv[-1]).write_bytes(b'seed boundary fixture')
            else:
                self.assertEqual(label, 'seed-verify')

        with patch.object(self.bootstrap, 'run', side_effect=seed_boundary), \
             patch.object(self.bootstrap, 'imports', return_value={str(library): digest(library)}), \
             patch.object(self.bootstrap, 'generation', side_effect=InterruptedError('generation boundary')):
            with self.assertRaisesRegex(InterruptedError, 'generation boundary'):
                self.bootstrap.stage1()
        self.assertEqual(self.bootstrap.manifest['hosts'], {str(library): digest(library)})
        retained = json.loads((self.bootstrap.work / 'retained-host-inputs.json').read_text())
        self.assertEqual(retained[str(library.parent)], ['file_product.o'])
        self.assertIn('__snapshot_0_0.i', retained[str(library.parent.parent)])
        environment = {**self.bootstrap.env, 'NANOLANG_BOOTSTRAP_NO_CC': '1'}
        wrapper = self.bootstrap.work / 'guard-native-compiler'
        host = self.root / 'modules/file_product/file_product.c'
        result = subprocess.run([wrapper, host, '-c', '-o', self.bootstrap.work / 'host.o'],
                                env=environment, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        product = self.bootstrap.work / 'program.c'
        product.write_text(host.read_text())
        result = subprocess.run([wrapper, product, '-c', '-o', self.bootstrap.work / 'product.o'],
                                env=environment, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 91, result.stderr)
        self.assertFalse((self.bootstrap.work / 'product.o').exists())

    def test_shadow_deadline_reaches_children_and_is_pinned_by_receipt(self):
        with patch.dict(os.environ, {}, clear=True):
            bootstrap = Bootstrap(self.root, 30)
        self.assertEqual(bootstrap.shadow_timeout, 30)
        bootstrap.work = self.bootstrap.work
        bootstrap.manifest = {**self.bootstrap.manifest, 'version': 1,
                              'root': str(self.root), 'work': str(bootstrap.work),
                              'shadow_timeout_seconds': 30, 'steps': []}
        bootstrap.run('deadline', [sys.executable, '-c',
                      'import os; print(os.environ["NANO_SHADOW_TIMEOUT_SECONDS"])'],
                      bootstrap.guarded_environment())
        self.assertEqual(Path(bootstrap.manifest['steps'][0]['log']).read_text(), '30\n')
        bootstrap.publish_receipt()
        bootstrap.load()
        changed = Bootstrap(self.root, 30, shadow_timeout=31)
        with self.assertRaisesRegex(RuntimeError, 'recorded bootstrap shadow deadline'):
            changed.load()

    def test_shadow_deadline_overrides_are_bounded(self):
        with patch.dict(os.environ, {'NANO_SHADOW_TIMEOUT_SECONDS': '17'}):
            self.assertEqual(Bootstrap(self.root, 30).shadow_timeout, 17)
            self.assertEqual(Bootstrap(self.root, 30, shadow_timeout=23).shadow_timeout, 23)
        for value in ('0', '301', '-1', '', '+30', ' 30', '٣٠', '9' * 5000):
            with self.subTest(value=value), patch.dict(os.environ, {'NANO_SHADOW_TIMEOUT_SECONDS': value}):
                with self.assertRaisesRegex(ValueError, '1 to 300'):
                    Bootstrap(self.root, 30)


if __name__ == '__main__':
    unittest.main()
