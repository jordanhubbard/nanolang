#!/usr/bin/env python3
"""I verify the retained compiler, changed sources, tool identities and terminals."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[2]
m = json.loads((here / 'manifest.json').read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
files = {}
for r in m['files']:
    compressed = (here / r['artifact']).read_bytes()
    assert sha(compressed) == r['compressed_sha256']
    data = gzip.decompress(compressed)
    assert len(data) == r['bytes'] and sha(data) == r['sha256']
    assert r['path'] not in files
    files[r['path']] = data
for name in ('src/transpiler.c', 'src/transpiler_iterative_v3_twopass.c', 'tests/test_imported_callback_typedefs.py'):
    expected = subprocess.check_output(['git', 'show', m['source_commit'] + ':' + name], cwd=root)
    assert files[name] == expected
assert sha(files['bin/nanoc_c']) == 'dde2cfef8d248a4a3feb16705818cb6168134239f0efb671a000ab93020bd236'
for name in ('1181-focused-final.log', '1181-focused-clang18.log'):
    text = files['logs/' + name].decode()
    assert 'Ran 5 tests' in text and text.rstrip().endswith('OK')
assert files['logs/1181-examples-r2.log'].decode().rstrip().endswith('Example regressions passed')
text = files['logs/1181-neighbors-r2.log'].decode()
assert 'Ran 39 tests' in text and text.rstrip().endswith('FAILED (failures=2)')
for name in ('test_captured_local_shadows_imported_alias', 'test_imported_closure_global_capture'):
    assert 'FAIL: ' + name in text
    assert 'FAIL: ' + name in files['logs/1181-baseline-closures.log'].decode()
assert json.loads(files['logs/frozen-guards.json'])['status'] == 'PASS'
assert '0 reserved names' in files['logs/1181-identifiers.log'].decode()
assert 'FAILED' not in files['logs/1181-contracts.log'].decode()
tools = json.loads(files['tools.json'])
assert tools['machine'] == 'aarch64'
print(json.dumps({'status': 'PASS', 'source_commit': m['source_commit'], 'records': len(files), 'focused_methods_per_toolchain': 5, 'original_examples': 5, 'legacy_neighbors_pass': 37, 'retained_baseline_closure_failures': 2}, indent=2))
