#!/usr/bin/env python3
"""I verify bounded parser observations, not a full compiler qualification."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import re
import struct
import subprocess

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
m = json.loads((BASE / 'manifest.json').read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = {r['path']: r for r in m['files']}
assert len(rows) == len(m['files'])
verified = {}
images = set()
for row in rows.values():
    path = (BASE / row['artifact']).resolve()
    assert path.is_relative_to(REPO / 'docs/evidence')
    identity = row['compressed_sha256'], row['sha256'], row['bytes']
    if path in verified:
        assert verified[path] == identity
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == row['compressed_sha256']
    digest = hashlib.sha256(); size = 0; head = b''
    with gzip.open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if not head: head = block[:64]
            digest.update(block); size += len(block)
    assert digest.hexdigest() == row['sha256'] and size == row['bytes']
    if (head[:6] == b'\x7fELF\x02\x01' and len(head) == 64
            and int.from_bytes(head[16:18], 'little') in (2, 3)
            and int.from_bytes(head[18:20], 'little') == 183):
        images.add(row['sha256'])
    verified[path] = identity
assert {p.resolve() for p in (BASE / 'objects').glob('*.gz')} == {
    (BASE / row['artifact']).resolve() for row in rows.values() if row['artifact'].startswith('objects/')}
prior_path = BASE / m['base_manifest']['path']
assert sha(prior_path.read_bytes()) == m['base_manifest']['sha256']

def data(name):
    return gzip.decompress((BASE / rows[name]['artifact']).read_bytes())

def raw(name):
    return data('journal/parser-shadow-cost-20261124/' + name)

def js(name):
    return json.loads(raw(name + '.json'))

def provider(path):
    key = 'journal/' + path.split('/.test_output/', 1)[1] if '/.test_output/' in path else ''
    return data(key if key in rows else 'provider/' + path)

pin = '3331bdfaaf66176e7b1dc704285768a4692bf873'
assert m['source_pin'] == pin
proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
current = {}
for row in m['git_sources']:
    proc.stdin.write((pin + ':' + row['path'] + '\n').encode()); proc.stdin.flush()
    header = proc.stdout.readline().split(); assert header[1] == b'blob'
    value = proc.stdout.read(int(header[2])); assert proc.stdout.read(1) == b'\n'
    assert sha(value) == row['sha256'] == rows['source/' + row['path']]['sha256']
    assert len(value) == row['bytes']; current[row['path']] = row['sha256']
proc.stdin.close(); assert proc.wait() == 0 and len(current) == 3518
original = data('source/src_nano/parser.nano')
suffix = b'\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'
methods = re.findall(rb'^shadow\s+(\w+)', original, re.M)
assert len(methods) == 276 and len(set(methods)) == 276
results = []
for prefix in ['', 'allocation-observer/']:
    q = js(prefix + 'result')
    assert q['pin'] == pin and q['before'] == q['after'] and len(q['before']) == 3500
    for path, digest in q['before'].items():
        if path == 'docs/ROADMAP.md':
            # My two predeclared planning versions are not compiler source edits.
            assert sha(raw(prefix + 'frozen-ROADMAP.md')) == digest
        else:
            assert current[path] == digest, path
    assert q['parser_sha256'] == sha(original)
    copied = provider(q['work'] + '/parser_probe.nano')
    assert copied == original + suffix and sha(copied) == q['input_sha256']
    assert all(r['exit'] == 0 for r in q['build']) and len(q['build']) == 3
    for path, digest in {**q['providers'], **q['tools']}.items():
        assert sha(provider(path)) == digest
    assert q['original_shadow_declarations'] == 276
    assert [r['mode'] for r in q['runs']] == ['ordinary', 'private']
    for run in q['runs']:
        assert run['exit'] == 0 and not run['timed_out']
        assert run['outer_timeout'] == 600 and run['shadow_timeout'] == 'default10'
        assert run['starts'] == run['completions'] == 277 and run['completed_failures'] == 0
        assert '--root-shadows-only' in run['args'] and '--emit-nvm' not in run['args']
        assert sha(provider(run['args'][0])) == run['image_sha256']
        events = []
        for line in provider(run['work'] + '/stderr').splitlines():
            try: events.append(json.loads(line))
            except ValueError: pass
        starts = [e for e in events if e.get('event') == 'shadow-start']
        ends = [e for e in events if e.get('event') == 'shadow-complete']
        assert len(starts) == len(ends) == 277
        expected = [x.decode() for x in methods] + ['main']
        assert [e['name'] for e in starts] == [e['name'] for e in ends] == expected
        assert all(e['failures'] == 0 for e in ends)
        status = next(e for e in run['phases'] if e['phase'] == 'supervisor-status')
        assert status['completed'] and status['reaped'] and status['exit_code'] == 0
        assert not status['parent_timeout'] and not status['clock_failure'] and status['deadline_seconds'] == 10
        pid = status['child_pid']; sample_dir = run['work'] + '/samples/'
        samples = provider(sample_dir + 'pc-' + str(pid) + '.bin')
        assert len(samples) > 0 and len(samples) % 16 == 0
        provider(sample_dir + 'maps-' + str(pid) + '.txt')
        for row in run['samples']:
            value = provider(row['path']); assert len(value) == row['bytes'] and sha(value) == row['sha256']
        summary = dict(mode=run['mode'], observer='counter' if prefix else 'sampler',
                       samples=len(samples) // 16, methods=277)
        if prefix:
            metrics = json.loads(provider(sample_dir + 'resources-' + str(pid) + '.json'))
            assert metrics['pid'] == pid and metrics['status'] == metrics['counter_overflows'] == 0
            assert metrics['malloc_calls'] > 0 and metrics['minor_faults'] > 0
            if run['mode'] == 'private':
                assert metrics['gc_objects_before'] == metrics['gc_objects_after'] == 99
                assert metrics['gc_usage_before'] == metrics['gc_usage_after'] == 4059
            else:
                assert metrics['gc_objects_after'] - metrics['gc_objects_before'] == 223207
                assert metrics['gc_usage_after'] - metrics['gc_usage_before'] == 11114936
            summary['resources'] = metrics
        results.append(summary)

analysis = js('analysis'); resolution = js('resolution-commands')
assert all(row['exit'] == 0 for row in resolution)
for path, digest in {**analysis['tool_files'], **js('debug-files'), **js('extra-tools')}.items():
    assert sha(provider(path)) == digest
resolved = {}
for i, row in enumerate(resolution):
    assert row['args'][:4] == ['/usr/bin/addr2line', '-f', '-C', '-e']
    path = row['args'][4]; addresses = list(map(lambda x: int(x, 16), row['args'][5:]))
    output = raw('resolve-%02d.stdout' % i).decode().splitlines()
    assert len(output) == len(addresses) * 2
    for n, address in enumerate(addresses):
        result = output[n * 2], output[n * 2 + 1]
        assert (path, address) not in resolved or resolved[path, address] == result
        resolved[path, address] = result
for run, summary in zip(js('result')['runs'], analysis['runs']):
    pid = next(e['child_pid'] for e in run['phases'] if e['phase'] == 'supervisor-status')
    samples = provider(run['work'] + '/samples/pc-' + str(pid) + '.bin')
    frames = list(struct.iter_unpack('<QQ', samples)); counts = [Counter(), Counter()]
    maps = provider(run['work'] + '/samples/maps-' + str(pid) + '.txt').decode()
    regions = []
    for line in maps.splitlines():
        bits = line.split(maxsplit=5)
        if len(bits) == 6 and 'x' in bits[1] and bits[5].startswith('/'):
            start, end = [int(x, 16) for x in bits[0].split('-')]
            regions.append((start, end, int(bits[2], 16), bits[5]))
    assert [list(r) for r in regions] == summary['source_regions']
    for frame in frames:
        for i, pc in enumerate(frame):
            key = next(((p, pc - start + offset) for start, end, offset, p in regions if start <= pc < end), None)
            counts[i][resolved.get(key, ('unmapped', ''))[0]] += 1
    assert summary['samples'] == len(frames)
    for i, name in enumerate(['pc', 'lr']):
        assert summary[name] == [dict(symbol=k, count=v, fraction=v / len(frames)) for k, v in counts[i].most_common(25)]

for name, expected in [('first-emit-nvm-setup', b'only for checked File graphs'),
                       ('second-image-location-setup', b'runtime/nl_string.h: No such file')]:
    q = js(name + '/result'); assert len(q['runs']) == 1
    run = q['runs'][0]; assert run['exit'] == 1 and run['starts'] == run['completions'] == 0
    assert expected in provider(run['work'] + '/stderr')
    guards = js(name + '/post-refusal-guards'); assert all(guards.values())
products = js('native-product-controls'); assert len(products) == 4
for row in products:
    assert row['exit'] == 0 and sha(provider(row['args'][0])) == row['sha256']
assert {k for k, value in m['acceptance'].items() if value} == {'parser_subset_observed', 'allocation_span_observed'}
print(json.dumps(dict(result='PASS', source=pin, records=len(rows), git_inputs=len(current),
                      retained_native_images=len(images), observations=results,
                      full_source_acceptance=False, historical_cause_established=False,
                      default_ownership_enabled=False, whole_release=False), indent=2))
