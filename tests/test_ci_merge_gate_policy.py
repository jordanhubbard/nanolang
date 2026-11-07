"""I preserve unrelated admin policy and refuse missing or non-successful check evidence."""
import copy
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from scripts import ci_merge_gate_policy as gates

ROOT = Path(__file__).resolve().parents[1]


class MergeGatePolicy(unittest.TestCase):
    def setUp(self):
        self.observed = json.loads((ROOT / 'docs/evidence/merge-check-policy-20261107/observed-ruleset.json').read_text())
        self.policy = json.loads((ROOT / '.github/merge-gates.json').read_text())

    def test_export_changes_only_additional_status_contexts(self):
        original = copy.deepcopy(self.observed)
        original_checks = gates.status_rule(self.observed)['parameters']['required_status_checks']
        original_checks[0]['integration_id'] = 15368
        before = copy.deepcopy(self.observed)
        payload = gates.proposed_payload(self.observed, self.policy)
        self.assertEqual(self.observed, before)
        checks = gates.status_rule(payload)['parameters']['required_status_checks']
        self.assertEqual(checks[:len(original_checks)], original_checks)
        self.assertEqual(set(gates.contexts(payload)), set(self.policy['required_contexts']))
        for key in gates.PAYLOAD_KEYS:
            if key != 'rules':
                self.assertEqual(payload[key], original[key])
        expected_rules = copy.deepcopy(before['rules'])
        for rule in expected_rules:
            if rule['type'] == 'required_status_checks':
                rule['parameters']['required_status_checks'] = checks
        self.assertEqual(payload['rules'], expected_rules)

    def test_additional_owner_checks_are_preserved(self):
        checks = gates.status_rule(self.observed)['parameters']['required_status_checks']
        checks.append({'context': 'Owner additional check', 'integration_id': 123})
        payload = gates.proposed_payload(self.observed, self.policy)
        self.assertIn(checks[-1], gates.status_rule(payload)['parameters']['required_status_checks'])
        self.assertEqual(gates.audit(self.observed, self.policy)['additional_observed_contexts'], ['Owner additional check'])

    def test_wrong_repository_or_duplicate_policy_refuses(self):
        for field, value in [('source', 'another/repository'), ('id', 1), ('target', 'tag')]:
            with self.subTest(field=field):
                wrong = dict(self.observed, **{field: value})
                with self.assertRaises(ValueError):
                    gates.proposed_payload(wrong, self.policy)
        self.policy['required_contexts'].append(self.policy['required_contexts'][0])
        with self.assertRaises(ValueError):
            gates.proposed_payload(self.observed, self.policy)

    def test_missing_checks_and_latest_non_success_never_become_admission(self):
        report = gates.audit(self.observed, self.policy, {'check_runs': []})
        self.assertEqual(len(report['missing_required_contexts']), 14)
        self.assertFalse(report['all_declared_head_checks_succeeded'])
        complete = {'check_runs': [{'id': number, 'name': name, 'status': 'completed', 'conclusion': 'success'}
                                  for number, name in enumerate(self.policy['required_contexts'])]}
        self.assertTrue(gates.audit(self.observed, self.policy, complete)['all_declared_head_checks_succeeded'])
        partial = dict(complete, total_count=100)
        self.assertFalse(gates.audit(self.observed, self.policy, partial)['all_declared_head_checks_succeeded'])
        pages = [{'total_count': 25, 'check_runs': complete['check_runs'][:10]},
                 {'total_count': 25, 'check_runs': complete['check_runs'][10:]}]
        self.assertTrue(gates.audit(self.observed, self.policy, pages)['all_declared_head_checks_succeeded'])
        for conclusion in ['failure', 'skipped', 'neutral', None]:
            checks = copy.deepcopy(complete)
            checks['check_runs'].append(dict(checks['check_runs'][0], id=1000, conclusion=conclusion))
            self.assertFalse(gates.audit(self.observed, self.policy, checks)['all_declared_head_checks_succeeded'])

    def test_cli_exports_locally_and_reports_live_policy_drift(self):
        source = ROOT / 'docs/evidence/merge-check-policy-20261107/observed-ruleset.json'
        before = source.read_bytes()
        with tempfile.TemporaryDirectory(prefix='nano-ruleset-review-') as folder:
            payload = Path(folder) / 'proposed.json'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/ci_merge_gate_policy.py'),
                                     '--observed-ruleset', str(source), '--payload', str(payload), '--check'],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(json.loads(result.stdout)['ruleset_matches_declared_minimum'])
            self.assertEqual(len(gates.contexts(json.loads(payload.read_text()))), 25)
        self.assertEqual(source.read_bytes(), before)

    def test_qualification_contexts_exist_for_both_native_and_prose_paths(self):
        text = (ROOT / '.github/workflows/retirement-qualification.yml').read_text()
        self.assertNotIn('\n    paths:', text.split('\njobs:', 1)[0])
        contexts = []
        for job in ['qualification-documentation', 'retirement']:
            block = text.split('  ' + job + ':\n', 1)[1].split('\n  retirement:', 1)[0]
            name = re.search(r'^    name: (.+)$', block, re.M).group(1)
            axes = {key: [value.strip() for value in re.search(r'^        ' + key + r': \[(.+)\]$', block, re.M).group(1).split(',')]
                    for key in ['os', 'corpus']}
            contexts.append({name.replace('${{ matrix.os }}', platform).replace('${{ matrix.corpus }}', corpus)
                             for platform, corpus in itertools.product(axes['os'], axes['corpus'])})
        declared = {name for name in self.policy['required_contexts'] if name.startswith('Qualification (')}
        self.assertEqual(len(declared), 8)
        self.assertEqual(contexts, [declared, declared])


if __name__ == '__main__':
    unittest.main()
