#!/usr/bin/env python3
"""I qualify complete current-source metadata without claiming a compiler bootstrap."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import ci_bootstrap_checkpoint as checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.test_output/checkpoint-metadata-qualification'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    names = checkpoint.tracked(ROOT)
    selected = sorted(set(names) | checkpoint.visible_source_directories(ROOT))
    report = {'commit': checkpoint.command(['git', 'rev-parse', 'HEAD']),
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'tracked_files': len(names), 'all_records': len(selected),
              'wire_bound': checkpoint.MAX_MANIFEST_BYTES,
              'scope': 'I qualify actual complete tracked-tree/directory metadata and source identity. '
                       'Native compiler products, a bootstrap receipt and full checkpoint transfer/consumers remain separate.'}
    save('result.json', report)
    records = {}
    for index, name in enumerate(selected):
        records[name] = checkpoint.record(ROOT / name)
        if index % 50000 == 0:
            save('progress.json', {'phase': 'record', 'completed': index, 'total': len(selected),
                                  'utc': datetime.now(timezone.utc).isoformat()})
            print('I recorded', index, 'of', len(selected), 'actual metadata entries.', flush=True)
    checkpoint.validate_links(records, ROOT)
    manifest = {'version': 1, 'root': str(ROOT), 'commit': report['commit'],
                'tracked': {name: records[name]['sha256'] for name in names}, 'files': records}
    observations = []
    original = json.dumps
    def observe(value, *args, **kwargs):
        encoded = original(value, *args, **kwargs)
        observations.append({'version': value['version'], 'bytes': len(encoded.encode())})
        return encoded
    try:
        checkpoint.json.dumps = observe
        encoded = checkpoint.encode_manifest(manifest)
    finally:
        checkpoint.json.dumps = original
    report['wire_observations'] = observations
    report['wire_sha256'] = hashlib.sha256(encoded).hexdigest()
    report['wire_bytes'] = len(encoded)
    (OUT / 'actual-source-metadata.json').write_bytes(encoded)
    decoded = checkpoint.decode_manifest(encoded)
    report['lossless_round_trip'] = decoded == manifest
    report['expected_large_format_selected'] = (len(observations) == 2 and observations[0]['version'] == 2 and
        observations[0]['bytes'] > report['wire_bound'] and observations[1]['version'] == 3 and
        observations[1]['bytes'] <= report['wire_bound'])
    report['all_source_metadata_unchanged'] = True
    for index, name in enumerate(selected):
        if checkpoint.record(ROOT / name) != records[name]:
            report['all_source_metadata_unchanged'] = False
            report.setdefault('changed_paths', []).append(name)
        if index % 50000 == 0:
            save('progress.json', {'phase': 'after-guard', 'completed': index, 'total': len(selected),
                                  'utc': datetime.now(timezone.utc).isoformat()})
    report['git_unchanged'] = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=ROOT).returncode == 0
    report['duration_seconds'] = time.monotonic() - started
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['success'] = all(report[key] for key in ('lossless_round_trip', 'expected_large_format_selected',
                                                  'all_source_metadata_unchanged', 'git_unchanged'))
    save('result.json', report)
    print(json.dumps(report, indent=2))
    if not report['success']:
        raise RuntimeError('I retain the incomplete or failed actual source metadata qualification.')


if __name__ == '__main__':
    main()
