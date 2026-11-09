"""I summarize observed parent/name relationships without asserting exact phase times."""
import collections
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
timelines = json.loads((root/'process-timelines.json').read_text())
summary = json.loads((root/'summary.json').read_text())
result = []
for attempt in timelines:
    trace = attempt['timeline']
    provider = summary['plan']['worktrees'][attempt['variant']]+'/bin/nanoc_c'
    mains = [row for row in trace['processes']
             if row['ppid'] == trace['root_pid'] and row['command_name'] == provider]
    assert len(mains) == 1
    main = mains[0]
    children = [row for row in trace['processes']
                if row['ppid'] == main['pid'] and row['command_name'] == provider]
    assert len(children) == 1
    child = children[0]
    first, last = child['snapshot_indices'][0], child['snapshot_indices'][-1]
    before = [row for row in trace['processes'] if Path(row['command_name']).name == 'clang'
              and any(i < first for i in row['snapshot_indices'])]
    after = [row for row in trace['processes'] if Path(row['command_name']).name == 'clang'
             and any(i > last for i in row['snapshot_indices'])]
    counts = collections.Counter(Path(row['command_name']).name for row in trace['processes'])
    result.append({'index': attempt['index'], 'variant': attempt['variant'],
        'factory_wall_seconds': trace['factory_wall_seconds'],
        'snapshot_wall_seconds': trace['snapshot_wall_seconds'],
        'snapshot_count': trace['snapshot_count'],
        'max_snapshot_gap_seconds': trace['gap_seconds']['max'],
        'c_provider_pid': main['pid'], 'same_command_child_pid': child['pid'],
        'same_command_child_first_sighting_bounds_seconds': child['first_snapshot_bounds_seconds'],
        'same_command_child_last_sighting_bounds_seconds': child['last_snapshot_bounds_seconds'],
        'clang_command_identities_observed_before_child': len(before),
        'clang_command_identities_observed_after_child': len(after),
        'command_identity_counts': dict(counts.most_common()),
        'scope': 'I identify one same-ps-comm child below the C-provider process, consistent with the original shadow fork. I infer its phase from that parent/name relationship and the completed shadow trace, not an independent executable-image or child-start timestamp receipt. Command sightings, snapshot spans and aggregate observer wall time are not exact CPU or execution durations. I do not subtract concurrent observer wall time from factory wall time.'})
(root/'phase-overview.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps([{'index': row['index'], 'variant': row['variant'],
    'same_command_child_first_sighting_bounds_seconds': row['same_command_child_first_sighting_bounds_seconds'],
    'clang_command_identities_observed_before_child': row['clang_command_identities_observed_before_child']}
    for row in result], indent=2))
