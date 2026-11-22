"""I validate retained native results and Git inputs without executing the native image."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = '6d8eeacce5e9dbdda7121cd0910d81d70c4772ce'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    packed = (HERE / 'observation.tar.gz').read_bytes()
    assert sha(packed) == manifest['archive_sha256'] and len(packed) == manifest['archive_bytes']
    files = {}
    links = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(packed), mode='r|gz') as source:
        for item in source:
            assert item.name not in files and item.name not in links
            if item.isfile():
                assert item.size < 64 * 1024**2
                total += item.size
                assert total < 512 * 1024**2
                files[item.name] = source.extractfile(item).read()
                assert len(files[item.name]) == item.size
            elif item.issym():
                links[item.name] = item.linkname
    def read(name):
        return files['observation/' + name]
    result = json.loads(read('result.json'))
    assert result['source_pin'] == manifest['source_pin'] == PIN
    assert result['all_source_hashes_unchanged'] and result['tools_unchanged']
    assert result['success'] is False and result['original_full_examples_or_release_accepted'] is False
    assert [r['returncode'] for r in result['reports']] == [0, 0, 2]
    assert [r['timeout'] for r in result['reports']] == [900, 120, 120]
    assert json.loads(read('tools-before.json')) == json.loads(read('tools-after.json'))
    image = read('nanoc_c')
    assert image[:6] == b'\x7fELF\x02\x01' and int.from_bytes(image[18:20], 'little') == 183
    assert sha(image) == 'c02766e9f133d2d9f7d044200bde6cff1632238ead8288157a0a139332cb0089'
    after_bytes = (HERE / 'post-product-observation.json').read_bytes()
    assert sha(after_bytes) == manifest['post_product_sha256']
    assert sha((HERE / 'observe-post-product.py').read_bytes()) == manifest['post_product_observer_sha256']
    after = json.loads(after_bytes)
    assert after['source_pin'] == PIN and after['product_stable_after_original_target']
    assert after['root_product'] == after['before_tests_product'] == sha(image)
    assert sha(files['bin/nano_as_capture.so']) == 'c168849e52c24918250d5117742bcd35479bd9b917d5129c283ed890e3c10894'
    log = read('imported-c-header-controls.stderr').decode()
    assert 'Ran 3 tests' in log and '\nOK\n' in log
    examples = read('original-escaped-examples.stdout').decode()
    assert examples.count('Compiling ') == 5 and 'ERROR: 3 example regression(s) failed' in examples
    assert 'static const int64_t SDL_INIT_VIDEO' in examples and 'SDL_WINDOW_SHOWN' in examples
    inputs = json.loads(read('source-inputs.json'))
    assert len(inputs) == result['inputs'] == 3634
    for name, expected in inputs.items():
        assert sha(gzip.decompress(read('source-inputs/' + expected + '.gz'))) == expected
    # I compare each retained source blob directly with the exact Git revision.
    paths = ['src', 'src_nano', 'tests', 'scripts', 'modules', 'std', 'stdlib', '.github',
             'GNUmakefile', 'Makefile.gnu']
    tree = subprocess.check_output(['git', 'ls-tree', '-rz', PIN, '--', *paths], cwd=ROOT)
    blobs = {}
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        assert kind == b'blob' and mode in (b'100644', b'100755')
        blobs[name.decode()] = oid.decode()
    assert set(blobs) == set(inputs)
    extra = {name.removeprefix('example-inputs-after/'): sha(raw)
             for name, raw in files.items() if name.startswith('example-inputs-after/')}
    for name in extra:
        blobs[name] = subprocess.check_output(['git', 'rev-parse', PIN + ':' + name], cwd=ROOT, text=True).strip()
    ordered = list(blobs)
    batch = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT,
        input=''.join(blobs[name] + '\n' for name in ordered).encode(), capture_output=True, timeout=60, check=True)
    stream = io.BytesIO(batch.stdout)
    for name in ordered:
        oid, kind, size = stream.readline().split()
        assert oid.decode() == blobs[name] and kind == b'blob'
        raw = stream.read(int(size))
        assert stream.read(1) == b'\n'
        assert sha(raw) == (inputs[name] if name in inputs else extra[name])
    assert stream.read() == b''
    hosted = (HERE / 'hosted-terminal.log.gz').read_bytes()
    assert sha(hosted) == manifest['hosted_packed_sha256']
    assert sha(gzip.decompress(hosted)) == manifest['hosted_log_sha256']
    assert sha((HERE / 'controller.py').read_bytes()) == manifest['controller_sha256']
    print(json.dumps({'regular_files': len(files), 'symlink_records': len(links),
        'guarded_Git_source_inputs': len(inputs), 'additional_Git_entry_sources_after_only': len(extra),
        'native_build_passed': True, 'native_header_methods_passed': 3,
        'original_escaped_examples_passed': 2, 'original_escaped_examples_failed': 3,
        'source_and_host_tool_guards_passed': True,
        'scope': 'I validate data and Git blobs only. I execute no ELF/compiler/GAS target. '
                 'The additional example entry sources are bound after execution; they were outside the initial guarded set.'}, indent=2))


if __name__ == '__main__':
    main()
