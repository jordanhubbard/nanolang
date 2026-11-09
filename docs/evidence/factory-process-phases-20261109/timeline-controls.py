"""I check retained actual sightings and refuse tampered audit inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('process_timeline_audit', ROOT/'analyze-processes.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ProcessTimelineAudit(unittest.TestCase):
    def setUp(self):
        self.snapshots = [json.loads(line) for line in (ROOT/'run.processes.jsonl').read_text().splitlines()]
        self.witness = json.loads((ROOT/'native-control.json').read_text())
        self.process = self.witness['command']

    def test_actual_native_witness_and_observer_totals(self):
        result = audit.timeline(self.snapshots, self.process)
        self.assertEqual(result['root_pid'], self.witness['parent_pid'])
        self.assertEqual(result['snapshot_count'], 3)
        self.assertEqual(result['snapshots_without_root'], 0)
        self.assertEqual({row['pid'] for row in result['processes']},
                         {self.witness['parent_pid'], self.witness['native_child_pid']})
        self.assertNotIn(self.witness['excluded_live_sibling_pid'],
                         {row['pid'] for row in result['processes']})
        self.assertGreater(result['observation_window_seconds'], 0)
        self.assertLess(result['observation_window_seconds'], result['factory_wall_seconds'])
        self.assertEqual(result['snapshot_wall_seconds'], self.process['process_observation']['snapshot_wall_seconds'])

    def test_injected_unowned_sibling_refuses(self):
        snapshots = copy.deepcopy(self.snapshots)
        row = dict(snapshots[0]['processes'][0])
        row['pid'] = self.witness['excluded_live_sibling_pid']
        snapshots[0]['processes'].append(row)
        with self.assertRaisesRegex(AssertionError, 'outside the observed root tree'):
            audit.timeline(snapshots, self.process)

    def test_order_identity_and_failed_observation_refuse(self):
        variants = []
        snapshots = copy.deepcopy(self.snapshots)
        snapshots.reverse()
        variants.append(snapshots)
        snapshots = copy.deepcopy(self.snapshots)
        snapshots[0]['started_monotonic'] = float('nan')
        variants.append(snapshots)
        snapshots = copy.deepcopy(self.snapshots)
        snapshots[0]['root_pid'] += 1
        variants.append(snapshots)
        snapshots = copy.deepcopy(self.snapshots)
        snapshots[0]['observation_failed'] = True
        variants.append(snapshots)
        snapshots = copy.deepcopy(self.snapshots)
        snapshots[-1]['processes'][0]['started'] = 'I retain a different root start identity.'
        variants.append(snapshots)
        for i, snapshots in enumerate(variants):
            with self.subTest(tampered_input=i), self.assertRaises(AssertionError):
                audit.timeline(snapshots, self.process)

    def test_missing_snapshot_or_changed_observer_total_refuses(self):
        with self.assertRaises(AssertionError):
            audit.timeline(self.snapshots[:-1], self.process)
        process = copy.deepcopy(self.process)
        process['process_observation']['snapshot_wall_seconds'] += 1
        with self.assertRaises(AssertionError):
            audit.timeline(self.snapshots, process)
        process = copy.deepcopy(self.process)
        process['wall_seconds'] = .1
        with self.assertRaises(AssertionError):
            audit.timeline(self.snapshots, process)


if __name__ == '__main__':
    unittest.main(verbosity=2)
