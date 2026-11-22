#!/usr/bin/env python3
"""I inspect retained binding evidence without executing native targets."""
from pathlib import Path
import gzip, hashlib, json
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
files = {}
for item in manifest['files']:
    packed = (root / item['artifact']).read_bytes()
    assert len(packed) == item['compressed_bytes']
    assert hashlib.sha256(packed).hexdigest() == item['compressed_sha256']
    raw = gzip.decompress(packed)
    assert len(raw) == item['bytes']
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    files[item['path']] = raw
selected = json.loads(files['before/.test_output/capture-assembler-ci/result.json'])
assert selected['compiler_before']['configured'] == 'clang'
assert selected['compiler_before']['sha256'] == '78e9dd2ccc9e88859b3a29109a44bfab00e3d6cb70f00f3daae063eed3d05251'
assert selected['probe_sha256'] == 'd4dfebb2bb7af578d8a1a594481696c539d5d1a33330d806e6ed609c76bb0a10'
commands = [line for line in files['before/.test_output/capture-assembler-checks/build.log'].splitlines() if line.startswith(b'cc ')]
assert len(commands) == 6
assert b'CC = cc\n' in files['source-before/Makefile.gnu']
assert b'"CC=${CC:-cc}"' in files['candidate/.github/actions/setup-capture-assembler/action.yml']
audit = json.loads(files['observer/shared-production-capture-helper-audit.json'])
assert audit['original_runner_bytes_identical'] and audit['all_original_debug_loop_AST_and_limits_identical']
assert b'Ran 31 tests' in files['observer/explicit-compiler-shared-helper-controls.log']
assert files['observer/explicit-compiler-shared-helper-controls.log'].rstrip().endswith(b'OK')
print(json.dumps({'files': len(files), 'all_hashes_and_lengths_match': True,
    'declared_clang_build_used_cc_commands': len(commands),
    'original_runner_and_debug_contract_preserved': True,
    'corrected_native_qualification': 'pending'}, indent=2))
