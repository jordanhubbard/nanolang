"""I independently read retained bytes without executing their native images."""
from pathlib import Path
import gzip
import hashlib
import json
import ast
import collections
import re

HERE = Path(__file__).resolve().parent

def normalized(calls,prefix):
 identities={};counts=collections.Counter()
 def replace(match):
  category=match.group(1);original=match.group(0)
  family='generation' if category in ('build','gen') else category
  key=(family,original.rsplit('-',1)[1])
  if key not in identities:counts[family]+=1;identities[key]=counts[family]
  return ('/tmp/nano-' if original.startswith('/tmp/') else '.nano-')+category+'-<'+str(identities[key])+'>'
 assert len({key for call in calls for arg in call for key in re.findall(r'v2-[0-9a-f]{64}',arg)})<=1
 return [[re.sub(r'/tmp/nano-(gcc-check)-[A-Za-z0-9]+',replace,re.sub(r'v2-[0-9a-f]{64}', 'v2-<cache-root>',re.sub(r'\.nano-(build|gen|link-source|link-query)-[A-Za-z0-9]+',replace,arg.replace(prefix,'$FIXTURE')))) for arg in call] for call in calls]

def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    rows = {item['path']: item for item in manifest['files']}
    checked = {}
    def read(name):
        entry = rows[name]
        if entry['artifact'] not in checked:
            packed = (HERE / entry['artifact']).read_bytes()
            raw = gzip.decompress(packed)
            assert hashlib.sha256(packed).hexdigest() == entry['compressed_sha256']
            assert len(packed) == entry['compressed_bytes']
            checked[entry['artifact']] = raw
        raw = checked[entry['artifact']]
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'] and len(raw) == entry['bytes']
        return raw
    for name in rows:
        read(name)
    prefix = 'hosted/capture-pair-profile/'
    result = json.loads(read(prefix + 'result.json'))
    assert result['completed'] and result['retained_images_unchanged'] and len(result['cases']) == 2
    inputs = commands = 0
    for case in result['cases']:
        base = prefix + case['name'] + '/'
        before = json.loads(read(base + 'observation/inputs-before.json'))
        assert before == json.loads(read(base + 'observation/inputs-after.json'))
        for name, expected in before.items():
            assert hashlib.sha256(read(base + 'root/' + name)).hexdigest() == expected
        assert case['native_closure_before'] == case['native_closure_after']
        assert case['source_inputs_unchanged'] and case['discovered'] == case['ran'] == 1
        assert case['failures'] == case['errors'] == 0 and case['skipped'] == []
        records = json.loads(read(base + 'observation/commands.json'))
        assert len(records) == 292
        assert sum(item['returncode'] != 0 for item in records) == 12
        assert all(item['returncode'] == 0 or item['argv'][1:2] == ['build'] for item in records)
        events = [json.loads(line) for line in read(base + 'observation/commands.jsonl').decode().splitlines()]
        assert len(events) == 584 and sum(item['event'] == 'complete' for item in events) == 292
        assert sum(item['argv'][1:2] == ['build'] for item in records) == 72
        inputs += len(before)
        commands += len(records)
    for group, images in result['images'].items():
        for name, image in images.items():
            assert hashlib.sha256(read('retained-native/' + group + '/' + name)).hexdigest() == image['sha256']
    old_case, new_case = result['cases']
    old_base, new_base = (prefix + item['name'] + '/' for item in (old_case, new_case))
    old_map = json.loads(read(old_base + 'observation/inputs-before.json'))
    new_map = json.loads(read(new_base + 'observation/inputs-before.json'))
    assert set(old_map) == set(new_map)
    assert [name for name in old_map if old_map[name] != new_map[name]] == ['tests/test_source_snapshots.py']
    helper = ast.parse(read(old_base + 'root/tests/assembler_search_fixture.py').decode())
    literals = {node.targets[0].id: ast.literal_eval(node.value) for node in helper.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)}
    old = read(old_base + 'root/tests/test_source_snapshots.py').decode()
    new = read(new_base + 'root/tests/test_source_snapshots.py').decode()
    assert old.count(literals['IMPORT']) == old.count(literals['LEGACY']) == 1
    assert new == old.replace(literals['IMPORT'], literals['IMPORT'] + literals['ADDED_IMPORT']).replace(literals['LEGACY'], literals['REPLACEMENT'])
    compiler_calls = 0
    for index in range(12):
        sides = []
        for case in result['cases']:
            fixture = case['fixture_paths'][index]
            name = prefix + case['name'] + '/observation/fixtures/' + Path(fixture).name + '/calls'
            calls = [json.loads(line) for line in read(name).decode().splitlines()]
            sides.append(normalized(calls, fixture))
        assert sides[0] == sides[1]
        compiler_calls += len(sides[0])
    assert compiler_calls == 870
    print(json.dumps({'logical_files': len(rows), 'unique_payloads': len(checked),
        'verified_source_and_native_inputs': inputs, 'observed_commands': commands, 'paired_compiler_calls_per_method': compiler_calls,
        'only_source_difference_is_declared_fixture_upgrade': True,
        'all_raw_and_compressed_hashes_and_lengths_match': True,
        'scope': 'I read retained bytes only; I do not execute ELF images or establish complete pair acceptance.'}, indent=2))

if __name__ == '__main__':
    main()
