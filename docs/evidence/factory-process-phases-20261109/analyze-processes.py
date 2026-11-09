"""I audit owned process sightings without claiming exact CPU or lifetime attribution."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile


def timeline(snapshots, process):
    assert type(process['wall_seconds']) in (int, float) and math.isfinite(process['wall_seconds']) and process['wall_seconds'] >= 0
    observation = process['process_observation']
    assert snapshots and len(snapshots) == observation['snapshots']
    identities = {}
    root_pid, root_start, previous = None, None, None
    observer_seconds, gaps, missing_root = 0., [], 0
    start = snapshots[0]['started_monotonic']
    for index, snapshot in enumerate(snapshots):
        assert not snapshot.get('observation_failed')
        before, after = snapshot['started_monotonic'], snapshot['finished_monotonic']
        assert all(type(value) in (int, float) and math.isfinite(value) for value in (before, after))
        assert 0 <= before <= after
        if previous is not None:
            assert previous <= before
            gaps.append(before-previous)
        previous = after
        observer_seconds += after-before
        if root_pid is None:
            root_pid = snapshot['root_pid']
        assert type(root_pid) is int and root_pid > 0 and snapshot['root_pid'] == root_pid
        rows = snapshot['processes']
        assert len({row['pid'] for row in rows}) == len(rows)
        by_pid = {row['pid']: row for row in rows}
        assert snapshot['root_seen'] == (root_pid in by_pid)
        if root_pid in by_pid:
            identity = by_pid[root_pid]['started']
            if root_start is None:
                root_start = identity
            assert identity == root_start
        else:
            missing_root += 1
        selected = {root_pid} if root_pid in by_pid else set()
        while True:
            expanded = selected | {row['pid'] for row in rows if row['ppid'] in selected}
            if expanded == selected:
                break
            selected = expanded
        assert selected == set(by_pid), 'I refuse a process outside the observed root tree.'
        for row in rows:
            assert all(type(row[key]) is int and row[key] > 0 for key in ('pid', 'ppid'))
            assert all(isinstance(row[key], str) and row[key] for key in ('started', 'status', 'command_name'))
            key = (row['pid'], row['started'], row['command_name'], row['ppid'])
            if key not in identities:
                identities[key] = {**row, 'sightings': 0, 'snapshot_indices': [],
                    'first_snapshot_bounds_seconds': [before-start, after-start],
                    'last_snapshot_bounds_seconds': [before-start, after-start],
                    'status_counts': collections.Counter()}
            item = identities[key]
            item['sightings'] += 1
            item['snapshot_indices'].append(index)
            item['last_snapshot_bounds_seconds'] = [before-start, after-start]
            item['status_counts'][row['status']] += 1
    assert root_start is not None
    assert abs(observer_seconds-observation['snapshot_wall_seconds']) < 1e-6
    window = snapshots[-1]['finished_monotonic']-start
    assert window <= process['wall_seconds']+1e-6
    items = sorted(identities.values(), key=lambda item: (item['first_snapshot_bounds_seconds'], item['pid']))
    return {'scope': 'I retain observed process presence and snapshot bounds. Gaps, ps collection races, exec transitions and process reparenting limit phase attribution. ps comm is not independently resolved image identity; no span is exact CPU or tool execution time.',
            'root_pid': root_pid, 'root_start_identity': root_start,
            'snapshot_count': len(snapshots), 'snapshots_without_root': missing_root,
            'observation_window_seconds': window, 'factory_wall_seconds': process['wall_seconds'],
            'snapshot_wall_seconds': observer_seconds,
            'gap_seconds': {'count': len(gaps), 'median': statistics.median(gaps) if gaps else None,
                            'max': max(gaps) if gaps else None},
            'processes': items}


def main():
    archive, destination = sys.argv[1:]
    destination = Path(destination)
    assert destination.is_dir(), 'I first require the complete comparison artifact audit.'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        prefix = 'cseed-source-comparison/'
        summary = json.loads(z.read(prefix+'summary.json'))
        assert summary['comparison_valid'] and summary['all_planned_samples_finished']
        artifact = json.loads((destination/'artifact.json').read_text())
        assert artifact['zip_sha256'] == hashlib.sha256(Path(archive).read_bytes()).hexdigest()
        assert summary == json.loads((destination/'summary.json').read_text())
        analyses = []
        for sample in summary['samples']:
            name = f"samples/{sample['index']:02}-{sample['variant']}.processes.jsonl"
            raw = z.read(prefix+name)
            assert Path(sample['process']['process_observation']['path']).name == Path(name).name
            snapshots = [json.loads(line) for line in raw.decode().splitlines()]
            analyses.append({'index': sample['index'], 'variant': sample['variant'],
                'creation_succeeded': sample['creation_succeeded'],
                'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
                'timeline': timeline(snapshots, sample['process'])})
        (destination/'process-timelines.json').write_text(json.dumps(analyses, indent=2)+'\n')
        print(json.dumps([{'index': a['index'], 'variant': a['variant'],
            'creation_succeeded': a['creation_succeeded'],
            'snapshots': a['timeline']['snapshot_count'],
            'observed_identities': len(a['timeline']['processes']),
            'snapshot_wall_seconds': a['timeline']['snapshot_wall_seconds'],
            'max_gap_seconds': a['timeline']['gap_seconds']['max']} for a in analyses], indent=2))


if __name__ == '__main__':
    main()
