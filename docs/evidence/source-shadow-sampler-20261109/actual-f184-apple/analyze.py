"""I count sampled contexts without treating recursive totals as exclusive time."""
from pathlib import Path
import collections
import hashlib
import json
import re

root = Path(__file__).resolve().parent
path = root/'child.sample.txt'
text = path.read_text()
graph = text.split('Call graph:', 1)[1].split('Total number in stack', 1)[0]
stack = []
comparisons, allocations = collections.Counter(), collections.Counter()
sites = collections.Counter()
for line in graph.splitlines():
    match = re.match(r'^([ +!:|]*)(\d+) (.*?)  \(in ([^)]*)\)(.*)', line)
    if not match:
        continue
    prefix, count, name, library, location = match.groups()
    depth, count = len(prefix), int(count)
    while stack and stack[-1]['depth'] >= depth:
        stack.pop()
    parent = next((frame for frame in reversed(stack)
                   if frame['library'] == 'nanoc_c' and not frame['name'].startswith('DYLD-STUB')), None)
    is_comparison = 'strcmp' in name
    if is_comparison and not any(frame['comparison'] for frame in stack):
        comparisons[parent['name'] if parent else '<no C caller>'] += count
        sites[(parent['name']+parent['location']) if parent else '<no C caller>'] += count
    is_allocator = library == 'libsystem_malloc.dylib'
    if is_allocator and not any(frame['allocator'] for frame in stack):
        allocations[parent['name'] if parent else '<no C caller>'] += count
    stack.append({'depth': depth, 'name': name, 'library': library,
                  'location': location, 'comparison': is_comparison, 'allocator': is_allocator})
samples = int(re.search(r'(?m)^\s+(\d+) Thread_', graph).group(1))
assert samples == 602 and sum(comparisons.values()) == 135
assert comparisons['eval_expression'] == 76
assert sum(count for site, count in sites.items() if 'eval.c:5745' in site) == 35
assert sum(count for site, count in sites.items() if 'eval.c:5658' in site) == 32
observation = json.loads((root/'observation.json').read_text())
assert observation['sample_returncode'] == 0 and not observation['native_product_exists']
assert observation['shadow_trace']['starts'] == observation['shadow_trace']['completes'] == 1286
assert observation['shadow_trace']['failures'] == 0
(root/'analysis.json').write_text(json.dumps({
    'sample_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'thread_snapshots': samples,
    'string_comparison_snapshots': sum(comparisons.values()),
    'string_comparison_nearest_c_callers': dict(comparisons.most_common()),
    'string_comparison_c_call_sites': dict(sites.most_common()),
    'allocator_nearest_c_callers': dict(allocations.most_common()),
    'reported_physical_footprint': '3.4G',
    'method': 'I count each rootmost comparison/allocator frame once per call-graph branch and attribute it to the nearest non-stub C frame. I do not sum recursive inclusive totals.',
    'next_measurement': 'I measure exact-name first-byte filters at record-field access and byte contextualization before considering production adoption.',
    'limits': 'Snapshots are not call/allocation counts, a wall-time measurement, proof of a leak, or an optimization comparison. The sampled interval does not establish the cause of every hosted deadline. Native creation, lifecycle and release remain unqualified.',
}, indent=2)+'\n')
print('I verify 602 snapshots, 135 string-comparison contexts and the two 35/32-snapshot record-field sites.')
