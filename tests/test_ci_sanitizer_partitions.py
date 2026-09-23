"""I preserve complete sanitizer selection and refuse incomplete evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('partition', ROOT / 'scripts/ci_sanitizer_partitions.py')
partition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(partition)


class SanitizerPartitions(unittest.TestCase):
    def inventory(self):
        return [*partition.DEDICATED, 'test-units-tail',
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
        self.assertEqual(len(value['workers']), 18)
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
                         ['make', 'build', *partition.FLAGS])
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
        self.assertEqual(partition.command_for(workers[0], 'bootstrap'), ['make', 'bootstrap3', *partition.FLAGS])
        self.assertEqual(value['native_cflags'], partition.NATIVE_CFLAGS)
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
        jobs = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())['jobs']
        worker = jobs['sanitizer-workers']
        self.assertNotIn('REPORT', worker['env'])
        self.assertNotIn('PLAN', worker['env'])
        self.assertEqual(worker['steps'][0]['run'],
                         'echo "REPORT=$RUNNER_TEMP/sanitizer-${{ matrix.id }}" >> "$GITHUB_ENV"\n'
                         'echo "PLAN=$RUNNER_TEMP/sanitizer-plan/plan.json" >> "$GITHUB_ENV"\n')
        workers = jobs['sanitizer-workers']
        self.assertEqual(workers['timeout-minutes'], "${{ (matrix.id == 'source' && 75) || (matrix.id == 'scalar' && 45) || 30 }}")
        self.assertFalse(workers['strategy']['fail-fast'])
        self.assertEqual(workers['strategy']['max-parallel'], 4)
        tests = next(step for step in workers['steps'] if step.get('id') == 'tests')
        self.assertEqual(tests['timeout-minutes'], "${{ (matrix.id == 'source' && 45) || (matrix.id == 'scalar' && 35) || 20 }}")
        self.assertEqual(workers['env']['ASAN_OPTIONS'], 'detect_leaks=0')
        self.assertEqual(workers['env']['NANO_SHADOW_TIMEOUT_SECONDS'], '60')
        self.assertNotIn('NANOLANG_COMPILER', workers['env'])
        instrumentation = next(step for step in workers['steps'] if step.get('id') == 'instrumentation')
        self.assertIn(' instrumentation --manifest ', instrumentation['run'])
        self.assertLess(workers['steps'].index(instrumentation), workers['steps'].index(tests))
        self.assertEqual(jobs['sanitizers']['needs'], ['sanitizer-plan', 'sanitizer-workers'])
        self.assertEqual(jobs['sanitizers']['if'], 'always()')
        aggregate = next(step for step in jobs['sanitizers']['steps'] if 'run' in step)
        self.assertIn('test "$WORKER_RESULT" = success', aggregate['run'])
        self.assertIn('test "$PLAN_RESULT" = success', aggregate['run'])
        for step in workers['steps']:
            self.assertNotIn('continue-on-error', step)


if __name__ == '__main__':
    unittest.main()
