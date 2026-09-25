"""I preserve complete sanitizer selection and refuse incomplete evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('partition', ROOT / 'scripts/ci_sanitizer_partitions.py')
partition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(partition)


class SanitizerPartitions(unittest.TestCase):
    def inventory(self):
        return [*partition.DEDICATED, *(target for _, target in partition.PHASES), 'test-units-tail',
                *(f'test-control-{index}' for index in range(40))]

    def test_exact_union_and_fixed_dedicated_workers(self):
        original = self.inventory()
        value = partition.plan('head', original)
        requested = [target for worker in value['workers'] for target in worker['targets']]
        self.assertCountEqual(requested, original)
        self.assertEqual(len(requested), len(set(requested)))
        self.assertEqual(value['workers'][0]['targets'], ['test-forth-session'])
        self.assertEqual(value['workers'][1]['targets'], ['test-nanoisa-src-nano'])
        self.assertEqual(value['workers'][2]['targets'], ['test-scalar-reconstruction'])
        self.assertEqual(len(value['workers']), 40)
        self.assertEqual([(worker['id'], worker['targets'][0]) for worker in value['workers'][3:11]],
                         list(partition.PHASES))
        self.assertEqual(value['workers'][-1], {'id': 'negative', 'targets': [], 'native_bootstrap': False})

    def test_new_target_is_included_and_changes_digest(self):
        original = self.inventory()
        first = partition.plan('head', original)
        added = partition.plan('head', original + ['test-added-required-control'])
        self.assertNotEqual(first['inventory_sha256'], added['inventory_sha256'])
        self.assertEqual(sum(w['targets'].count('test-added-required-control') for w in added['workers']), 1)

    def test_database_refuses_missing_tail_duplicates_and_syntax(self):
        good = 'test-units: ' + ' '.join(t for t in self.inventory() if t != 'test-units-tail') + '\n\t+@$(MAKE) test-units-tail'
        self.assertCountEqual(partition.parse_database(good), self.inventory())
        self.assertCountEqual(partition.parse_database(good + '\n\t\n'), self.inventory())
        bad = ['', good + '\n' + good, good + ' test-units-tail',
               good.replace('test-units-tail', ''), good + ' | test-extra',
               good + ' $(DYNAMIC)', good + ' ; echo unsafe', good + ' test_control',
               good.split('\n')[0], good + '\n\t@echo extra',
               good.replace('$(MAKE)', 'make'), good.replace('test-units: ', 'test-units: test-units-tail ')]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                partition.parse_database(text)

    def test_manifest_rejects_reassignment_and_wrong_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            good = partition.plan('head', self.inventory())
            partition.save(path, good)
            self.assertEqual(partition.checked_plan(path), good)
            changed = json.loads(json.dumps(good))
            changed['workers'][0]['targets'].append('test-units-tail')
            partition.save(path, changed)
            with self.assertRaises(ValueError):
                partition.checked_plan(path)
            changed = dict(good, inventory_sha256='wrong')
            partition.save(path, changed)
            with self.assertRaises(ValueError):
                partition.checked_plan(path)

    def test_commands_preserve_flags_and_whole_original_targets(self):
        value = partition.plan('head', self.inventory())
        self.assertEqual(partition.command_for(value['workers'][0], 'sanitize'), ['make', 'sanitize'])
        self.assertEqual(partition.command_for(value['workers'][0], 'bootstrap'),
                         ['make', 'build', *partition.BOOTSTRAP_FLAGS,
                          *partition.BOOTSTRAP_DRIVER_FLAGS, *partition.BOOTSTRAP_TIMEOUT_FLAGS,
                          *partition.FLAGS])
        self.assertEqual(partition.command_for(value['workers'][0], 'bootstrap1'),
                         ['make', 'bootstrap1', *partition.FLAGS])
        self.assertEqual(partition.command_for(value['workers'][0], 'bootstrap1-driver'),
                         ['make', 'bootstrap1-driver', *partition.BOOTSTRAP_FLAGS, *partition.FLAGS])
        self.assertEqual(partition.command_for(value['workers'][0], 'bootstrap1-companions'),
                         ['make', 'file-companion-plan', *partition.FLAGS])
        self.assertEqual(partition.command_for(value['workers'][0], 'bootstrap2-nvm'),
                         ['bin/nanoc_stage1', '--root-shadows-only', 'src_nano/nanoc_v06.nano',
                          '--emit-nvm', '-o', partition.STAGE2_NVM])
        self.assertEqual(partition.command_for(value['workers'][0], 'bootstrap2-c'),
                         ['bin/nvm2c', partition.STAGE2_NVM, '-o', partition.STAGE2_C])
        native_object = partition.command_for(value['workers'][0], 'bootstrap2-object')
        self.assertEqual(native_object[0], partition.NATIVE_CC)
        self.assertIn(partition.STAGE2_C, native_object)
        self.assertEqual(native_object[-3:], ['-c', '-o', partition.STAGE2_OBJECT])
        native = partition.command_for(value['workers'][0], 'bootstrap2-native')
        self.assertEqual(native[0], partition.NATIVE_CC)
        self.assertIn(partition.STAGE2_OBJECT, native)
        self.assertIn('bin/nano_aot_runtime.o', native)
        for provider in partition.STAGE2_PROVIDER_OBJECTS:
            self.assertIn(provider, native)
        self.assertEqual(native[-2:], ['-o', 'bin/nanoc_stage2'])
        self.assertEqual(partition.command_for(value['workers'][0], 'bootstrap2-smoke'),
                         ['bin/nanoc_stage2', 'examples/language/nl_hello.nano',
                          '-o', 'build/sanitizer-stage2/hello'])
        self.assertEqual(partition.command_for(value['workers'][0], 'bootstrap3'),
                         ['make', 'bootstrap3', *partition.BOOTSTRAP_FLAGS,
                          *partition.BOOTSTRAP_DRIVER_FLAGS, *partition.BOOTSTRAP_TIMEOUT_FLAGS,
                          *partition.FLAGS])
        for worker in value['workers'][:-1]:
            self.assertEqual(partition.command_for(worker, 'tests'), ['make', *worker['targets'], *partition.FLAGS])
        self.assertEqual(partition.command_for(value['workers'][-1], 'tests'), ['bash', 'tests/run_negative_tests.sh'])
        with self.assertRaises(ValueError):
            partition.command_for(value['workers'][0], 'providers')

    def test_aggregate_requires_all_exact_successful_workers(self):
        value = partition.plan('head', self.inventory())
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for worker in value['workers']:
                path = Path(tmp) / worker['id'] / 'result.json'
                partition.save(path, {'worker': worker['id'], 'targets': worker['targets'],
                    'head': value['head'], 'inventory_sha256': value['inventory_sha256'],
                    'native_bootstrap': worker['native_bootstrap'], 'instrumentation_verified': True, 'success': True})
                paths.append(path)
            self.assertTrue(partition.aggregate(value, tmp)['success'])
            original = paths[-1].read_text()
            for field, replacement in [('success', False), ('head', 'different'),
                                       ('inventory_sha256', 'different'), ('targets', ['test-extra']),
                                       ('instrumentation_verified', False), ('native_bootstrap', True)]:
                data = json.loads(original); data[field] = replacement; partition.save(paths[-1], data)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    partition.aggregate(value, tmp)
            paths[-1].write_text(original)
            duplicate = Path(tmp) / 'duplicate' / 'result.json'
            duplicate.parent.mkdir(); duplicate.write_text(original)
            with self.assertRaises(ValueError):
                partition.aggregate(value, tmp)
            duplicate.unlink(); paths[-1].unlink()
            with self.assertRaises(ValueError):
                partition.aggregate(value, tmp)

    def test_actual_make_inventory_contains_tail(self):
        result = subprocess.run(['make', '-qp', 'test-units', *partition.FLAGS],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertIn(result.returncode, (0, 1), result.stderr)
        targets = partition.parse_database(result.stdout)
        native = partition.native_bootstrap_consumers(result.stdout, targets)
        self.assertIn('test-scalar-reconstruction', native)
        self.assertIn('test-selfhost-byte-array-identity', native)
        self.assertNotIn('test-forth-session', native)
        planned = partition.plan('head', targets, native)
        self.assertCountEqual([target for worker in planned['workers'] for target in worker['targets']], targets)
        self.assertIn('test-units-tail', targets)
        self.assertIn('test-verify-all-programs', targets)
        self.assertIn('test-nanovm', targets)

    def test_bootstrap_graph_follows_shared_and_order_only_edges_and_cycles(self):
        text = ('test-a: first | shared\nfirst: second\nsecond: first shared\n'
                'shared: .bootstrap2.built\n.bootstrap2.built: seed\n'
                'test-b: ordinary\nordinary: file.c\ntest-c: shared\n')
        self.assertEqual(partition.native_bootstrap_consumers(text, ['test-a', 'test-b', 'test-c']),
                         ['test-a', 'test-c'])
        for bad in ('test-a: $(UNRESOLVED)\n', 'test-a: first\ntest-a: second\n',
                    'test-a: first ; command\n', ''):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                partition.native_bootstrap_consumers(bad, ['test-a'])
        self.assertEqual(partition.native_bootstrap_consumers(
            'test-a: CFLAGS += -g\ntest-a: bootstrap3\n', ['test-a']), ['test-a'])

    def test_bootstrap_roles_and_flags_are_manifest_obligations(self):
        targets = self.inventory()
        native = ['test-control-0']
        value = partition.plan('head', targets, native)
        workers = [w for w in value['workers'] if w['native_bootstrap']]
        self.assertEqual(len(workers), 1)
        self.assertEqual(partition.command_for(workers[0], 'bootstrap'),
                         ['make', 'bootstrap3', *partition.BOOTSTRAP_FLAGS,
                          *partition.BOOTSTRAP_DRIVER_FLAGS, *partition.BOOTSTRAP_TIMEOUT_FLAGS,
                          *partition.FLAGS])
        self.assertEqual(partition.BOOTSTRAP_TIMEOUT_FLAGS, ['BOOTSTRAP2_TIMEOUT=7200'])
        self.assertEqual(value['native_cflags'], partition.NATIVE_CFLAGS)
        self.assertEqual(value['native_ldflags'], partition.NATIVE_LDFLAGS)
        self.assertEqual(value['stage2_native_cflags'], partition.STAGE2_NATIVE_CFLAGS)
        self.assertEqual(value['native_cc'], partition.NATIVE_CC)
        self.assertEqual(value['schema'], 5)
        self.assertEqual(partition.generated_native_environment({}), {
            'NANO_CFLAGS': partition.NATIVE_CFLAGS,
            'NANO_LDFLAGS': partition.NATIVE_LDFLAGS,
            'NANO_CC': partition.NATIVE_CC,
        })
        for name in ('NANO_CFLAGS', 'NANO_LDFLAGS', 'NANO_CC'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                partition.generated_native_environment({name: 'different'})
        for bad in (['absent'], [native[0], native[0]]):
            with self.assertRaises(ValueError):
                partition.plan('head', targets, bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            value['workers'][0]['native_bootstrap'] = True
            partition.save(path, value)
            with self.assertRaises(ValueError):
                partition.checked_plan(path)

    def test_symbol_families_require_both_actual_sanitizers(self):
        self.assertEqual(partition.sanitizer_symbols(' U __asan_init\n U __ubsan_handle_add_overflow\n'),
                         {'asan': True, 'ubsan': True})
        self.assertEqual(partition.sanitizer_symbols(' T ordinary_main\n'), {'asan': False, 'ubsan': False})
        self.assertEqual(partition.sanitizer_symbols(' U __asan_report_load8\n'), {'asan': True, 'ubsan': False})

    def test_build_bundle_is_head_bound_checksummed_and_safe(self):
        value = partition.plan('head', self.inventory())
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as restored_tmp:
            source = Path(source_tmp)
            (source / 'bin').mkdir()
            (source / 'bin/nanoc_c').write_bytes(b'instrumented compiler')
            providers = {}
            for name in ('compiler_support', 'file_companion', 'file_source_catalog', 'nanoisa', 'std'):
                provider = source / f'modules/{name}/.build/.nano-gen-test/lib{name}.so'
                provider.parent.mkdir(parents=True)
                provider.write_bytes(('exact ' + name + ' provider').encode())
                providers[name] = provider
            (source / '.stage1.built').write_text('')
            output = source / 'bundle'
            try:
                os.chdir(source)
                with mock.patch.object(partition, 'current_head', return_value='head'):
                    metadata = partition.bundle_create(output, value, 'base')
                archive, description = output / 'base.tar.gz', output / 'base.json'
                self.assertEqual(metadata['archive_sha256'], partition.file_hash(archive))
                os.chdir(restored_tmp)
                with mock.patch.object(partition, 'current_head', return_value='head'):
                    restored = partition.bundle_restore(Path(restored_tmp) / 'report', value, 'base',
                                                        archive, description)
                self.assertTrue(restored['success'])
                self.assertEqual(Path('bin/nanoc_c').read_bytes(), b'instrumented compiler')
                for name, provider in providers.items():
                    restored_provider = Path(restored_tmp) / provider.relative_to(source)
                    self.assertEqual(restored_provider.read_bytes(),
                                     ('exact ' + name + ' provider').encode())
                os.chdir(source)
                (source / 'bin/nanoc_stage1').write_bytes(b'instrumented stage 1')
                (source / 'bin/nanoc_stage1_driver').write_bytes(b'ordinary stage 1 producer')
                for name in partition.BUNDLE_PRODUCTS['stage1']:
                    path = source / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        path.write_bytes(('required ' + name).encode())
                with mock.patch.object(partition, 'current_head', return_value='head'):
                    stage1 = partition.bundle_create(output, value, 'stage1')
                self.assertTrue({'bin/nanoc_c', 'bin/nanoc_stage1',
                                 'bin/nanoc_stage1_driver'}.issubset(stage1['products']))
                os.chdir(restored_tmp)
                tampered = source / 'tampered.tar.gz'
                tampered.write_bytes(archive.read_bytes() + b'modified')
                tampered_data = json.loads(description.read_text())
                tampered_data['archive'] = tampered.name
                tampered_description = source / 'tampered.json'
                partition.save(tampered_description, tampered_data)
                with mock.patch.object(partition, 'current_head', return_value='head'), self.assertRaises(ValueError):
                    partition.bundle_restore(Path(restored_tmp) / 'report2', value, 'base', tampered,
                                             tampered_description)
                unsafe = source / 'unsafe.tar.gz'
                with tarfile.open(unsafe, 'w:gz') as stream:
                    info = tarfile.TarInfo('../outside')
                    info.size = 0
                    stream.addfile(info)
                data = json.loads(description.read_text())
                data['archive'] = unsafe.name
                data['archive_sha256'] = partition.file_hash(unsafe)
                unsafe_description = source / 'unsafe.json'
                partition.save(unsafe_description, data)
                with mock.patch.object(partition, 'current_head', return_value='head'), self.assertRaises(ValueError):
                    partition.bundle_restore(Path(restored_tmp) / 'report3', value, 'base', unsafe,
                                             unsafe_description)
            finally:
                os.chdir(previous)

    def test_instrumentation_refuses_missing_products_tool_failure_and_plain_stage(self):
        worker = {'id': 'unit', 'native_bootstrap': True}
        good = b' U __asan_init\n U __ubsan_handle_add_overflow\n'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch.object(partition.Path, 'is_file', return_value=False):
                with self.assertRaises(ValueError):
                    partition.instrumented_products(worker, output)
            for status, symbols in ((1, good), (0, b' T main\n'), (0, b' U __asan_init\n')):
                completed = subprocess.CompletedProcess(['nm'], status, symbols, b'')
                with mock.patch.object(partition.Path, 'is_file', return_value=True), \
                     mock.patch.object(partition, 'file_hash', return_value='digest'), \
                     mock.patch.object(partition.subprocess, 'run', return_value=completed):
                    with self.assertRaises(ValueError):
                        partition.instrumented_products(worker, output)
            completed = subprocess.CompletedProcess(['nm'], 0, good, b'')
            with mock.patch.object(partition.Path, 'is_file', return_value=True), \
                 mock.patch.object(partition, 'file_hash', return_value='digest'), \
                 mock.patch.object(partition.subprocess, 'run', return_value=completed):
                result = partition.instrumented_products(worker, output)
            self.assertEqual(result['native_cc'], partition.NATIVE_CC)
            self.assertEqual(result['native_ldflags'], partition.NATIVE_LDFLAGS)
            self.assertEqual(result['stage2_native_cflags'], partition.STAGE2_NATIVE_CFLAGS)
            self.assertEqual(set(result['products']), {'bin/nanoc_c', 'bin/nanoc_stage1', 'bin/nanoc_stage2'})
            prepared = {'products': {p: 'digest' for p in result['products']}}
            self.assertTrue(partition.instrumentation_stable(worker, output, prepared, prepared))
            changed = {'products': dict(prepared['products'], **{'bin/nanoc_stage2': 'other'})}
            self.assertFalse(partition.instrumentation_stable(worker, output, prepared, changed))

    def test_parallel_make_tail_waits_for_every_prerequisite(self):
        makefile = (ROOT / 'Makefile.gnu').read_text()
        marker = '\t+@$(MAKE) test-units-tail'
        self.assertEqual(makefile.count(marker), 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Actual parallel Make, with different completion delays; the recursive
            # recipe is copied from production and observes both completed files.
            (root / 'Makefile').write_text(
                '.PHONY: test-units first second test-units-tail\n'
                'test-units: first second\n' + marker + '\n'
                'first:\n\t@sleep 0.1; touch first.done\n'
                'second:\n\t@sleep 0.2; touch second.done\n'
                'test-units-tail:\n\t@test -f first.done && test -f second.done\n'
                '\t@printf "both prerequisites finished\\n" > tail.done\n')
            result = subprocess.run(['make', '-j4', 'test-units'], cwd=root,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / 'tail.done').read_text(), 'both prerequisites finished\n')

    def test_workflow_requires_aggregate_and_preserves_limits(self):
        import yaml
        workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
        self.assertEqual(workflow['defaults']['run']['shell'], 'bash')
        jobs = workflow['jobs']
        worker = jobs['sanitizer-workers']
        self.assertNotIn('REPORT', worker['env'])
        self.assertNotIn('PLAN', worker['env'])
        self.assertEqual(worker['steps'][0]['run'],
                         'echo "REPORT=$RUNNER_TEMP/sanitizer-${{ matrix.id }}" >> "$GITHUB_ENV"\n'
                         'echo "PLAN=$RUNNER_TEMP/sanitizer-plan/plan.json" >> "$GITHUB_ENV"\n'
                         'echo "BUNDLE=$RUNNER_TEMP/sanitizer-bootstrap" >> "$GITHUB_ENV"\n')
        workers = jobs['sanitizer-workers']
        self.assertEqual(workers['timeout-minutes'], "${{ (matrix.id == 'source' && 75) || (matrix.id == 'scalar' && 45) || 30 }}")
        self.assertFalse(workers['strategy']['fail-fast'])
        self.assertEqual(workers['strategy']['max-parallel'], 8)
        tests = next(step for step in workers['steps'] if step.get('id') == 'tests')
        self.assertEqual(tests['timeout-minutes'], "${{ (matrix.id == 'source' && 45) || (matrix.id == 'scalar' && 35) || 20 }}")
        self.assertEqual(workers['env']['ASAN_OPTIONS'], 'detect_leaks=0')
        self.assertEqual(workers['env']['NANO_SHADOW_TIMEOUT_SECONDS'], '300')
        for job in ('sanitizer-base', 'sanitizer-stage1', 'sanitizer-stage2-nvm',
                    'sanitizer-stage2-c', 'sanitizer-stage2-object', 'sanitizer-bootstrap',
                    'sanitizer-providers'):
            self.assertEqual(jobs[job]['env']['NANO_SHADOW_TIMEOUT_SECONDS'], '300')
        restore = next(step for step in workers['steps'] if step.get('id') == 'restore')
        self.assertIn(' restore --manifest ', restore['run'])
        self.assertNotIn('NANOLANG_COMPILER', workers['env'])
        instrumentation = next(step for step in workers['steps'] if step.get('id') == 'instrumentation')
        self.assertIn(' instrumentation --manifest ', instrumentation['run'])
        self.assertLess(workers['steps'].index(instrumentation), workers['steps'].index(tests))
        self.assertEqual(jobs['sanitizer-workers']['needs'], ['sanitizer-plan', 'sanitizer-providers'])
        self.assertEqual(jobs['sanitizers']['needs'],
                         ['sanitizer-plan', 'sanitizer-base', 'sanitizer-stage1',
                          'sanitizer-stage2-nvm', 'sanitizer-stage2-c', 'sanitizer-stage2-object',
                          'sanitizer-bootstrap', 'sanitizer-providers', 'sanitizer-workers'])
        self.assertEqual(jobs['sanitizer-stage1']['needs'], ['sanitizer-plan', 'sanitizer-base'])
        stage1_steps = jobs['sanitizer-stage1']['steps']
        driver = next(step for step in stage1_steps if step.get('name') == 'Build ordinary Stage 1 producer')
        companions = next(step for step in stage1_steps
                          if step.get('name') == 'Build instrumented Stage 2 companion providers')
        publish_stage1 = next(step for step in stage1_steps if step.get('name') == 'Publish exact Stage 1 bundle')
        self.assertIn('--phase bootstrap1-driver', driver['run'])
        self.assertIn('--phase bootstrap1-companions', companions['run'])
        self.assertLess(stage1_steps.index(driver), stage1_steps.index(publish_stage1))
        self.assertLess(stage1_steps.index(companions), stage1_steps.index(publish_stage1))
        self.assertEqual(jobs['sanitizer-stage2-nvm']['needs'], ['sanitizer-plan', 'sanitizer-stage1'])
        self.assertEqual(jobs['sanitizer-stage2-c']['needs'], ['sanitizer-plan', 'sanitizer-stage2-nvm'])
        self.assertEqual(jobs['sanitizer-stage2-object']['needs'], ['sanitizer-plan', 'sanitizer-stage2-c'])
        self.assertEqual(jobs['sanitizer-bootstrap']['needs'], ['sanitizer-plan', 'sanitizer-stage2-object'])
        self.assertEqual(jobs['sanitizer-bootstrap']['timeout-minutes'], 150)
        nvm = next(step for step in jobs['sanitizer-stage2-nvm']['steps']
                   if step.get('name') == 'Build verified Stage 2 NanoISA')
        translated = next(step for step in jobs['sanitizer-stage2-c']['steps']
                          if step.get('name') == 'Translate verified Stage 2 NanoISA to C')
        native_object = next(step for step in jobs['sanitizer-stage2-object']['steps']
                             if step.get('name') == 'Compile instrumented Stage 2 object')
        native = next(step for step in jobs['sanitizer-bootstrap']['steps']
                      if step.get('name') == 'Compile instrumented native Stage 2')
        self.assertIn('--phase bootstrap2-nvm', nvm['run'])
        self.assertIn('--phase bootstrap2-c', translated['run'])
        self.assertIn('--phase bootstrap2-object', native_object['run'])
        self.assertIn('--phase bootstrap2-native', native['run'])
        verification = next(step for step in jobs['sanitizer-bootstrap']['steps']
                            if step.get('name') == 'Verify Stage 2 and complete bootstrap')
        self.assertIn('--phase bootstrap2-smoke', verification['run'])
        self.assertIn('touch .bootstrap1.built .bootstrap2.built', verification['run'])
        self.assertLess(verification['run'].index('--phase bootstrap2-smoke'),
                        verification['run'].index('touch .bootstrap1.built .bootstrap2.built'))
        self.assertLess(verification['run'].index('touch .bootstrap1.built .bootstrap2.built'),
                        verification['run'].index('--phase bootstrap3'))
        self.assertNotIn('\n          bin/nanoc_stage2 ', verification['run'])
        self.assertIn('set -o pipefail', verification['run'])
        self.assertIn('bootstrap_pid=$!', verification['run'])
        self.assertIn('sleep 60', verification['run'])
        self.assertIn('wait "$bootstrap_pid"', verification['run'])
        self.assertIn('150-minute job lease', verification['run'])
        self.assertIn('clang', partition.NATIVE_CC)
        self.assertIn('-lffi', partition.command_for({'id': 'scalar'}, 'bootstrap2-native'))
        self.assertEqual(jobs['sanitizer-providers']['needs'], ['sanitizer-plan', 'sanitizer-bootstrap'])
        self.assertEqual(jobs['sanitizers']['if'], 'always()')
        aggregate = next(step for step in jobs['sanitizers']['steps'] if 'run' in step)
        self.assertIn('test "$WORKER_RESULT" = success', aggregate['run'])
        self.assertIn('test "$PLAN_RESULT" = success', aggregate['run'])
        self.assertIn('test "$BASE_RESULT" = success', aggregate['run'])
        self.assertIn('test "$STAGE1_RESULT" = success', aggregate['run'])
        self.assertIn('test "$STAGE2_NVM_RESULT" = success', aggregate['run'])
        self.assertIn('test "$STAGE2_C_RESULT" = success', aggregate['run'])
        self.assertIn('test "$STAGE2_OBJECT_RESULT" = success', aggregate['run'])
        self.assertIn('test "$PROVIDERS_RESULT" = success', aggregate['run'])
        self.assertIn('test "$BOOTSTRAP_RESULT" = success', aggregate['run'])
        platform_runs = [step.get('run', '') for step in jobs['build-and-test']['steps']]
        self.assertNotIn('make test TEST_TIMEOUT=3600', platform_runs)
        coverage_tests = next(step for step in jobs['coverage']['steps'] if step.get('name') == 'Run tests')
        self.assertEqual(coverage_tests['run'], './tests/run_all_tests.sh')
        self.assertEqual(jobs['coverage']['env']['COVERAGE_THRESHOLD'], '25.0')
        for step in workers['steps']:
            self.assertNotIn('continue-on-error', step)


if __name__ == '__main__':
    unittest.main()
