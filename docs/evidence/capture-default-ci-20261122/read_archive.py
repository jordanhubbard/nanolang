"""I read actual hosted selection and guide evidence without executing native images."""
from pathlib import Path
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    rows = {item['path']: item for item in manifest['files']}
    assert len(rows) == len(manifest['files'])
    cache = {}
    def read(name):
        item = rows[name]
        path = (HERE / item['artifact']).resolve(strict=True)
        assert path.is_relative_to(HERE.parent)
        if path not in cache:
            packed = path.read_bytes()
            assert len(packed) == item['compressed_bytes']
            assert hashlib.sha256(packed).hexdigest() == item['compressed_sha256']
            cache[path] = gzip.decompress(packed)
        raw = cache[path]
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
        return raw
    for name in rows:
        read(name)
    prefix = 'native-selection/.test_output/capture-assembler-ci/'
    record = json.loads(read(prefix + 'result.json'))
    profile = json.loads(read(prefix + 'validated-profile.json'))
    assert record['settings_unchanged'] and record['environment_changed']
    assert record['compiler_before'] == record['compiler_after']
    assert record['compiler_before']['sha256'] == '864711037b900acdb7a449cf48d912cefa935b527fb60c767ccb956779e66136'
    assert hashlib.sha256(read(prefix + 'selected-as')).hexdigest() == record['selected_assembler']['sha256'] == profile['assembler_sha256'] == '7e9edb4b7fc0c7ce97fbcdd4ecb63e3eb1c1663e2452cd32ce0249110b253bf5'
    probe = read('native-selection/obj/test_module_generation_probe')
    assert probe[:6] == b'\x7fELF\x02\x01' and int.from_bytes(probe[18:20], 'little') == 62
    assert hashlib.sha256(probe).hexdigest() == record['probe_sha256'] == 'd4dfebb2bb7af578d8a1a594481696c539d5d1a33330d806e6ed609c76bb0a10'
    assert hashlib.sha256(read('executed-source/f645/src/module_builder.c')).hexdigest() == record['capture_source_sha256'] == '971b242622730d2581bd5d576bec8dc7ce48a05c67e4917765d4f347f346bb09'
    origin = record['profile_origin']
    assert origin['producer_run'] == 45009129801
    assert origin['manifest_sha256'] == hashlib.sha256((HERE.parents[2] / origin['manifest']).read_bytes()).hexdigest()
    identity = json.loads(read('native-selection/.test_output/capture-assembler-checks/identity.json'))
    assert identity['event_sha'] == 'ed31e880676deb7a7cae6325a090f726aa5ed447'
    run = json.loads(read('observer/ordinary-ci-45028658551-run.json'))
    assert run['status'] == 'completed' and run['conclusion'] == 'success'
    assert run['head_sha'] == 'f645082d2a52d5418bb914421a37cf9818bd9174'
    log = read('observer/ordinary-ci-45028658551.log').decode()
    assert identity['event_sha'] in log and 'userguide-check: 40 snippet(s) passed' in log
    assert 'NANO_SHADOW_TIMEOUT_SECONDS: 60' in log
    print(json.dumps({'files': len(rows), 'unique_payloads': len(cache),
        'all_hashes_and_lengths_match': True, 'native_setup_and_hosted_guide_pass': True,
        'compiler_settings_preserved': True, 'original_hosted_shadow_seconds': 60,
        'scope': 'I read evidence only. Ordinary ten-second qualification remains separately retained; full platform and release gates remain open.'}, indent=2))


if __name__ == '__main__':
    main()
