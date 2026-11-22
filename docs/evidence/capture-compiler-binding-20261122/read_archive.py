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

# I compare whole retained images without stripping or executing either image.
import struct
new = files['refused-current/capture-assembler-capture-Linux-X64-c34531bc751e/obj/test_module_generation_probe']
old = files['comparison/previous-clang-probe-66122']
assert hashlib.sha256(new).hexdigest() == 'c61fac6137641dfd5c0e3f2e6dd8ce0b098220e0b86b7e99274051f5533c5070'
assert hashlib.sha256(old).hexdigest() == '66122ae9d8401dda5457a0157a45b92e54f081385b394d0b7b0a5e5829d79f16'
def sections(raw):
    assert raw[:6] == b'\x7fELF\x02\x01'
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    entries = [struct.unpack_from('<IIQQQQIIQQ', raw, header[6] + i*header[11]) for i in range(header[12])]
    table = entries[header[13]]
    names = raw[table[4]:table[4]+table[5]]
    return {names[item[0]:].split(b'\0')[0].decode(): (item[1], item[5], raw[item[4]:item[4]+item[5]] if item[1] != 8 else b'') for item in entries}
a, b = sections(old), sections(new)
assert a.keys() == b.keys()
assert {name for name in a if a[name] != b[name]} == {'.debug_line', '.note.gnu.build-id'}
for pin, expected in (('3337d0e880', '32b8d7d4bc24e1587f4bd2badd738165'), ('3116f8f87d', '613d79ee8b132d4f30c5bb10f1d89582')):
    assert hashlib.md5(files['comparison/'+pin+'/src/nanolang.h']).hexdigest() == expected
build = files['refused-current/capture-assembler-capture-Linux-X64-c34531bc751e/.test_output/capture-assembler-checks/build.log']
assert len([line for line in build.splitlines() if line.startswith(b'clang ')]) == 6
print('I verify the actual corrected six-command Clang setup and retain its stale-image observer refusal; full native qualification is pending.')
