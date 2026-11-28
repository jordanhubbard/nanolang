#!/usr/bin/env python3
"""I compare existing complete-shadow prefixes without starting another run."""
from pathlib import Path
import gzip
import hashlib
import json
import statistics

BASE = Path(__file__).resolve().parent
manifest = json.loads((BASE / 'inputs.json').read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
def read(row):
    path = (BASE / row['artifact']).resolve()
    assert path.is_relative_to(BASE.parents[2] / 'docs/evidence')
    compressed = path.read_bytes()
    assert sha(compressed) == row['compressed_sha256']
    raw = gzip.decompress(compressed)
    assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
    return raw

def event_identity(event):
    return event['source'], event['line'], event['name']

runs = {}
for dataset in manifest['datasets']:
    qualification = json.loads(read(dataset['qualification']))
    assert qualification['before'] == qualification['after']
    assert qualification['timeout'] == 600 and qualification['shadow_timeout'] == 'default10'
    events = []
    for line in read(dataset['trace']).splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    starts = [e for e in events if e.get('event') == 'shadow-start']
    completions = [e for e in events if e.get('event') == 'shadow-complete']
    assert len(starts) == qualification['starts'] and len(completions) == qualification['completions']
    assert all(e['failures'] == 0 and e['elapsed_seconds'] >= 0 for e in completions)
    assert list(map(event_identity, completions)) == list(map(event_identity, starts[:len(completions)]))
    status = next(e for e in events if e.get('phase') == 'supervisor-status')
    dispatch = next(e for e in events if e.get('phase') == 'dispatch-start')
    assert status['deadline_seconds'] == dispatch['value'] == 10
    duration = ((status['monotonic_seconds'] - dispatch['monotonic_seconds']) +
                (status['monotonic_nanoseconds'] - dispatch['monotonic_nanoseconds']) / 1e9)
    if dataset['kind'] == 'passing':
        assert qualification['exit'] == 0 and len(completions) == len(starts) == 1306
        assert status['completed'] and status['exit_code'] == 0 and not status['parent_timeout']
    else:
        assert qualification['exit'] == 1 and len(starts) == len(completions) + 1
        assert status['signal'] == 14 and not status['completed'] and status['parent_timeout']
    runs[dataset['mode'], dataset['kind']] = qualification, completions, duration
assert len(runs) == 6
comparisons = []
for mode in ['ordinary', 'private']:
    baseline, reference, _ = runs[mode, 'passing']
    for kind in ['unbound-failure', 'cpu7-failure']:
        qualification, actual, dispatch_duration = runs[mode, kind]
        assert list(map(event_identity, actual)) == list(map(event_identity, reference[:len(actual)]))
        nano_sources = {p: h for p, h in baseline['before'].items() if p.endswith('.nano')}
        assert len(nano_sources) == 233
        assert nano_sources == {p: h for p, h in qualification['before'].items() if p.endswith('.nano')}
        prior_seconds = sum(e['elapsed_seconds'] for e in reference[:len(actual)])
        actual_seconds = sum(e['elapsed_seconds'] for e in actual)
        ratios = [b['elapsed_seconds'] / a['elapsed_seconds'] for a, b in zip(reference, actual)
                  if a['elapsed_seconds'] >= 0.005]
        comparisons.append(dict(mode=mode, kind=kind, matching_completed_prefix=len(actual),
                                nano_inputs=233, prior_prefix_seconds=prior_seconds,
                                observed_prefix_seconds=actual_seconds,
                                prefix_elapsed_ratio=actual_seconds / prior_seconds,
                                median_ratio_prior_at_least_5ms=statistics.median(ratios),
                                observed_dispatch_to_status_seconds=dispatch_duration,
                                completed_shadow_fraction=actual_seconds / dispatch_duration))
assert manifest['new_native_attempts'] == 0
assert not any(manifest[k] for k in ['ordinary_full_source', 'private_full_source', 'whole_release'])
result = dict(result='PASS', meaning='Existing trace comparison, not renewed qualification',
              native_attempts=0, comparisons=comparisons, historical_cause_established=False,
              current_source_acceptance=False, whole_release=False)
print(json.dumps(result, indent=2))
