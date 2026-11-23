"""I independently read source, native bytes and original reports without ELF execution."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    checked = {}
    rows = {item['path']: item for item in manifest['files']}
    def read(name):
        item = rows[name]
        packed_path = (HERE / item['artifact']).resolve(strict=True)
        assert packed_path.is_relative_to(HERE.parent)
        if item['artifact'] not in checked:
            packed = packed_path.read_bytes()
            assert hashlib.sha256(packed).hexdigest() == item['compressed_sha256']
            assert len(packed) == item['compressed_bytes']
            checked[item['artifact']] = gzip.decompress(packed)
        raw = checked[item['artifact']]
        assert hashlib.sha256(raw).hexdigest() == item['sha256'] and len(raw) == item['bytes']
        return raw
    for name in rows:
        read(name)
    prefix = 'data/prior-fixed-clang/fixed-as246/'
    assert hashlib.sha256(read(prefix + 'binutils-2.46.0.tar.xz')).hexdigest() == 'd75a94f4d73e7a4086f7513e67e439e8fcdcbb726ffe63f4661744e6256b2cf2'
    for name in (prefix + 'bin/as', 'data/as246-candidate/test_module_generation_probe',
                 'data/as246-candidate/nano_as_capture.so'):
        image = read(name)
        assert image[:6] == b'\x7fELF\x02\x01' and int.from_bytes(image[18:20], 'little') == 62
    witnesses = json.loads(read(prefix + 'witness/results.json'))
    assert len(witnesses) == 8 and witnesses == json.loads(read(prefix + 'build.json'))['witnesses']
    for item in witnesses:
        name = prefix + 'witness/' + item['tool'] + '-' + Path(item['case']).stem + '.o'
        assert hashlib.sha256(read(name)).hexdigest() == item['object_sha256']
    old = json.loads(read('observer/previous-clang/tests.json'))
    current = json.loads(read('data/as246-candidate/tests.json'))
    assert old['skipped'] == current['skipped']
    assert json.loads(read('data/as246-candidate/response-tests.json'))['skipped'] == []
    with tempfile.TemporaryDirectory(prefix='nano-clang-archive-reader-') as temporary:
        root = Path(temporary)
        for name in rows:
            if not name.startswith('data/'):
                continue
            relative = Path(name.removeprefix('data/'))
            assert not relative.is_absolute() and '..' not in relative.parts
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(read(name))
        report = root / 'reader.json'
        result = subprocess.run([sys.executable, str(HERE / 'audit_clang.py'), str(root), str(report)],
                                capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace'))
        audit = json.loads(report.read_text())
    print(json.dumps({'logical_files': len(rows), 'unique_payloads': len(checked),
        'all_raw_and_compressed_hashes_and_lengths_match': True,
        'original_conditional_skip_tuples_unchanged': True,
        'native': audit,
        'scope': 'I execute a Python data reader only, never an ELF/compiler/GAS target. '
                 'The original native producer remains45009129801; this fresh consumer is45023493733.'}, indent=2))


if __name__ == '__main__':
    main()
