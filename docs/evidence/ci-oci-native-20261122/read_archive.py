"""I read actual native image transport evidence without executing its images."""
import argparse
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-archive', type=Path)
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    rows = {item['path']: item for item in manifest['files']}
    assert len(rows) == len(manifest['files'])
    cached = {}
    def read(name):
        item = rows[name]
        path = (HERE / item['artifact']).resolve(strict=True)
        assert path.is_relative_to(HERE.parent)
        if path not in cached:
            packed = path.read_bytes()
            assert len(packed) == item['compressed_bytes'] and sha(packed) == item['compressed_sha256']
            cached[path] = gzip.decompress(packed)
        raw = cached[path]
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
        return raw
    for name in rows:
        read(name)
    run = json.loads(read('observer/oci-native-45030266230-run.json'))
    assert run['status'] == 'completed' and run['conclusion'] == 'success'
    assert run['head_sha'] == 'a2ffa846ae20c4f932a97b9d75193a1be6c3b336'
    artifact = json.loads(read('observer/oci-native-45030266230-artifact.json'))
    assert artifact['id'] == 14757796289 and artifact['workflow_run']['id'] == 45030266230
    identity = json.loads(read('consumer/_temp/ci-loaded-image.json'))
    audit = json.loads(read('native-archive/independent-audit.json'))
    assert identity['archive_sha256'] == audit['archive_sha256'] == 'ba43ea56fe50245d4014d9c40a3d0de6e8382295f394bfd2b759ef683f0dce25'
    assert identity['image_id'] == audit['image_id'] == 'sha256:b34b7ee6db65698acc414ea6ff2f88e149fd6b0f1e3cf01438e713f551e9b31c'
    assert identity['config_id'] == audit['config_id'] == 'sha256:fd7b4195239ed23831bdd3f9462327e17f9e62c3ca69a49e1d9a0a9623dd465e'
    assert identity['platform'] == audit['platform'] == 'linux/amd64'
    for label, item in audit['metadata'].items():
        raw = read('native-archive/' + label + '.json')
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    archive_index = json.loads(read('native-archive/archive-index.json'))
    assert len(archive_index['manifests']) == 1 and archive_index['manifests'][0]['digest'] == identity['image_id']
    root_bytes = read('native-archive/image-index.json')
    assert 'sha256:' + sha(root_bytes) == identity['image_id']
    root = json.loads(root_bytes)
    selected = [d for d in root['manifests'] if d.get('platform') == {'os': 'linux', 'architecture': 'amd64'}]
    assert len(selected) == 1
    assert any(d.get('annotations', {}).get('vnd.docker.reference.type') == 'attestation-manifest' and
        d.get('platform') == {'os': 'unknown', 'architecture': 'unknown'} for d in root['manifests'])
    selected = selected[0]
    image_bytes = read('native-archive/platform-manifest.json')
    assert selected['size'] == len(image_bytes) and selected['digest'] == audit['platform_manifest'] == 'sha256:' + sha(image_bytes)
    image = json.loads(image_bytes)
    config = read('native-archive/config.json')
    assert image['config']['digest'] == identity['config_id'] == 'sha256:' + sha(config)
    assert image['config']['size'] == len(config)
    assert [(d['digest'], d['size']) for d in image['layers']] == [(d['digest'], d['bytes']) for d in audit['selected_image_layers']]
    assert all(d['matches'] for d in audit['selected_image_layers']) and len(image['layers']) == 10
    checkpoint_bytes = read('consumer/_temp/container-checkpoint-probe/checkpoint.tar.gz')
    with tarfile.open(fileobj=io.BytesIO(checkpoint_bytes)) as checkpoint:
        record = json.load(checkpoint.extractfile('checkpoint.json'))
        assert record['environment']['image'] == identity
        for name, item in record['files'].items():
            raw = checkpoint.extractfile('files/' + name).read()
            assert sha(raw) == item['sha256'] and len(raw) == item['size']
            assert raw == read('consumer/_temp/container-checkpoint-probe/' + name)
            assert raw[:6] == b'\x7fELF\x02\x01' and int.from_bytes(raw[18:20], 'little') == 62
    log = read('consumer/nanolang/nanolang/.test_output/ci-image-graph/smoke.log').decode()
    for marker in ('I passed real container transport, immutable-root, instrumented-provider and identity controls.',
        'I verified LLVM IR, linked Wasm and executed it in both engines inside the verified image.',
        'I compiled and executed sanitizer and coverage probes in the verified image.',
        'I restored exact providers in a second container despite a different outer runner label; changed flags still refuse identity.'):
        assert marker in log
    assert read('consumer/_temp/container-smoke-output') == b'container-output=retained\n'
    assert read('consumer/_temp/container-smoke-summary') == b'I retained the container summary.\n'
    assert read('consumer/_temp/container-checkpoint-probe/engine.wasm')[:4] == b'\0asm'
    assert 'Ran 7 tests' in read('producer/controls.log').decode()
    full_archive = False
    if args.native_archive:
        assert args.native_archive.stat().st_size == audit['archive_bytes']
        with args.native_archive.open('rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == identity['archive_sha256']
        with tarfile.open(args.native_archive, 'r:') as source:
            for label, item in audit['metadata'].items():
                assert source.extractfile(item['native_member']).read() == read('native-archive/' + label + '.json')
            for item in image['layers']:
                member = source.getmember('blobs/sha256/' + item['digest'].split(':')[1])
                assert member.isfile() and member.size == item['size']
                with source.extractfile(member) as raw:
                    assert 'sha256:' + hashlib.file_digest(raw, 'sha256').hexdigest() == item['digest']
        full_archive = True
    print(json.dumps({'logical_files': len(rows), 'all_retained_hashes_and_lengths_match': True,
        'actual_native_metadata_hash_edges_match': True, 'selected_image_layers': 10,
        'native_archive_rehashed_this_read': full_archive, 'native_cross_worker_smoke_passed': True,
        'scope': 'I execute data validation only. Native sanitizer/coverage/LLVM/Wasm probes are tiny smoke providers; '
                 'they are not a compiler bootstrap or full product-corpus qualification.'}, indent=2))


if __name__ == '__main__':
    main()
