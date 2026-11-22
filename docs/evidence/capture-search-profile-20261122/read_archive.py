"""I independently read retained bytes without executing their native images."""
from pathlib import Path
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent

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
    print(json.dumps({'logical_files': len(rows), 'unique_payloads': len(checked),
        'verified_source_and_native_inputs': inputs, 'observed_commands': commands,
        'all_raw_and_compressed_hashes_and_lengths_match': True,
        'scope': 'I read retained bytes only; I do not execute ELF images or establish complete pair acceptance.'}, indent=2))

if __name__ == '__main__':
    main()
