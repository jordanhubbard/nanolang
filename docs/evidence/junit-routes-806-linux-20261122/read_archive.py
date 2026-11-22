#!/usr/bin/env python3
"""I verify retained bytes and recompute the original route census."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
m = json.loads((HERE / 'manifest.json').read_text())
assert sha((HERE / m['base_manifest']).read_bytes()) == m['base_manifest_sha256']
files = {}
for record in m['files']:
    compressed = (HERE / record['artifact']).read_bytes()
    assert len(compressed) == record['compressed_bytes']
    assert sha(compressed) == record['compressed_sha256']
    raw = gzip.decompress(compressed)
    assert len(raw) == record['bytes'] and sha(raw) == record['sha256']
    assert record['path'] not in files
    files[record['path']] = raw
assert len(files) == m['counts']['files']
assert sum(map(len, files.values())) == m['counts']['raw_bytes']
census = json.loads((HERE / 'census.json').read_text())['routes']
selections, refusals = [], []
for route, passes in zip(census, (237, 237, 215, 215), strict=True):
    cases = list(ET.fromstring(files[f"audit/routes/{route['name']}/results.xml"]).iter('testcase'))
    keys = [(c.get('classname'), c.get('name')) for c in cases]
    failures = [(c.get('classname'), c.get('name')) for c in cases if c.find('failure') is not None or c.find('error') is not None]
    assert len(cases) == route['total'] == 237
    assert len(cases) - len(failures) == route['passed'] == passes
    assert route['skipped'] == 0 and not any(c.find('skipped') is not None for c in cases)
    selections.append(keys)
    if failures:
        assert len(failures) == 22
        refusals.append(failures)
        assert b'Alarm clock' in files[f"audit/routes/{route['name']}/runner.log"]
assert all(keys == selections[0] for keys in selections)
assert refusals[0] == refusals[1]
controller = json.loads(files['post-controller/controller.json'])
assert controller['status'] == 'failed'
assert [s['exit'] for s in controller['steps']] == [0, 0, 1]
assert controller['source'] == m['source_commit']
for path, raw in files.items():
    if path.endswith('.tar.gz.json'):
        inventory = json.loads(raw)
        archive = files[path[:-5]]
        assert sha(archive) == inventory['archive_sha256']
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
            members = {t.name.rstrip('/'): t for t in tar.getmembers()}
            assert members.keys() == inventory['entries'].keys()
            for name, expected in inventory['entries'].items():
                member = members[name]
                if expected['kind'] == 'file':
                    data = tar.extractfile(member).read()
                    assert len(data) == expected['size'] and sha(data) == expected['sha256']
                else:
                    assert expected['kind'] == 'directory' and member.isdir()
assert len(m['failed_temporary']) == 2
assert all(entry['files'] == 0 for entry in m['failed_temporary'])
print(json.dumps({'status': 'PASS', 'files': len(files), 'routes': '237/237/215/215', 'stage_failures': 22, 'skips': 0}, indent=2))
