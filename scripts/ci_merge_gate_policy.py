#!/usr/bin/env python3
"""I audit declared merge contexts and export a reviewable ruleset; I never apply it."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_KEYS = ('name', 'target', 'enforcement', 'bypass_actors', 'conditions', 'rules')


def status_rule(ruleset):
    rules = [rule for rule in ruleset['rules'] if rule['type'] == 'required_status_checks']
    if len(rules) != 1:
        raise ValueError('I require exactly one existing required-status rule.')
    return rules[0]


def contexts(ruleset):
    return [row['context'] for row in status_rule(ruleset)['parameters']['required_status_checks']]


def proposed_payload(observed, policy):
    if policy.get('schema') != 'nanolang/merge-gate-policy@1':
        raise ValueError('I require my reviewed merge-gate policy schema.')
    if (observed.get('id') != policy['ruleset_id']
            or observed.get('source') != policy['repository'] or observed.get('target') != 'branch'):
        raise ValueError('I require the exact declared repository branch ruleset.')
    required = policy['required_contexts']
    if (not required or any(not isinstance(name, str) or not name.strip() for name in required)
            or len(required) != len(set(required))):
        raise ValueError('I require a nonempty unique declared context list.')
    payload = {key: copy.deepcopy(observed[key]) for key in PAYLOAD_KEYS}
    existing = contexts(payload)
    if len(existing) != len(set(existing)):
        raise ValueError('I require unique existing check contexts before review.')
    checks = status_rule(payload)['parameters']['required_status_checks']
    checks.extend({'context': name} for name in required if name not in existing)
    return payload


def audit(observed, policy, checks=None):
    proposed_payload(observed, policy)
    declared, actual = set(policy['required_contexts']), set(contexts(observed))
    report = {'schema': 'nanolang/merge-gate-policy-audit@1',
              'ruleset_id': policy['ruleset_id'], 'repository': policy['repository'],
              'declared_context_count': len(declared), 'observed_context_count': len(actual),
              'missing_required_contexts': sorted(declared - actual),
              'additional_observed_contexts': sorted(actual - declared),
              'ruleset_matches_declared_minimum': declared <= actual,
              'owner_approval_required_for_application': True,
              'blocking_policy_prerequisites': policy['blocking_policy_prerequisites'],
              'native_lifecycle_qualified': False}
    if checks is not None:
        by_name = {}
        pages = checks if isinstance(checks, list) else [checks]
        rows = [check for page in pages for check in page['check_runs']]
        expected = max((page.get('total_count', len(page['check_runs'])) for page in pages), default=0)
        complete = len({check['id'] for check in rows}) >= expected
        report['head_observation_complete'] = complete
        for check in rows:
            if check['name'] not in by_name or check.get('id', 0) > by_name[check['name']].get('id', 0):
                by_name[check['name']] = check
        report['head_missing_contexts'] = sorted(declared - by_name.keys())
        report['head_contexts_without_completed_success'] = [
            {'name': name, 'status': by_name[name]['status'], 'conclusion': by_name[name]['conclusion']}
            for name in sorted(declared & by_name.keys())
            if by_name[name]['status'] != 'completed' or by_name[name]['conclusion'] != 'success']
        report['all_declared_head_checks_succeeded'] = not (
            not complete or report['head_missing_contexts'] or report['head_contexts_without_completed_success'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=ROOT / '.github/merge-gates.json')
    parser.add_argument('--observed-ruleset', type=Path, required=True)
    parser.add_argument('--observed-checks', type=Path)
    parser.add_argument('--payload', type=Path, help='I write a local review artifact, without a network mutation.')
    parser.add_argument('--check', action='store_true', help='I fail if the live ruleset lacks declared checks.')
    parser.add_argument('--require-head-success', action='store_true',
                        help='I also require completed success for every declared context; this is not full release admission.')
    args = parser.parse_args()
    if args.require_head_success and not args.observed_checks:
        parser.error('I require actual check-run observations for --require-head-success.')
    policy = json.loads(args.policy.read_text())
    observed = json.loads(args.observed_ruleset.read_text())
    checks = json.loads(args.observed_checks.read_text()) if args.observed_checks else None
    report = audit(observed, policy, checks)
    if args.payload:
        args.payload.write_text(json.dumps(proposed_payload(observed, policy), indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return int((args.check and not report['ruleset_matches_declared_minimum'])
               or (args.require_head_success and not report['all_declared_head_checks_succeeded']))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('I could not audit my merge policy: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
