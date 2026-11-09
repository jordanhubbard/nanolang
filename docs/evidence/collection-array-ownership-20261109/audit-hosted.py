"""I audit uploaded snapshot products, selected inputs and actual operation outcomes."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def native(data, shared, mode, command):
    if data[:4] == b'\xcf\xfa\xed\xfe':
        assert struct.unpack_from('<I', data, 12)[0] == (6 if shared else 2)
        kind = 'Mach-O dylib' if shared else 'Mach-O executable'
    else:
        assert data[:4] == b'\x7fELF' and data[5] in (1, 2)
        value = struct.unpack_from('<H' if data[5] == 1 else '>H', data, 16)[0]
        assert value in ((3,) if shared else (2, 3))
        kind = 'ELF shared object' if shared else 'ELF executable/PIE'
    assert all(flag in command for flag in ('-Werror', '-UNDEBUG', '-pthread'))
    if mode == 'sanitized':
        assert '-fsanitize=address,undefined' in command
        assert b'__asan_init' in data
        if not any(Path(arg).name in ('origin.c', 'dependency.c') for arg in command):
            assert b'__ubsan_handle' in data
    if mode == 'threaded':
        assert '-fsanitize=thread' in command and b'__tsan_init' in data
    return kind


def audit(archive, pin, output, retained):
    assert not output.exists() and not retained.exists()
    source = subprocess.check_output(['git', 'show', pin+':tests/test_ffi_result_release.py'])
    literals = {node.value.args[0].value for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'write_text'
                and node.value.args and isinstance(node.value.args[0], ast.Constant)
                and isinstance(node.value.args[0].value, str)}
    inventory, texts, products, witnesses, to_retain = [], {}, [], [], {}
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        for info in z.infolist():
            assert not info.filename.startswith('/') and '..' not in Path(info.filename).parts
            if info.is_dir():
                continue
            h = hashlib.sha256()
            with z.open(info) as stream:
                for block in iter(lambda: stream.read(65536), b''):
                    h.update(block)
            inventory.append({'path': info.filename, 'bytes': info.file_size, 'sha256': h.hexdigest()})
            if Path(info.filename).suffix in ('.json', '.log', '.txt', '.stdout', '.stderr', '.sha256', '.nasm', '.c'):
                texts[info.filename] = z.read(info).decode(errors='replace')
        roots = sorted(p.rsplit('/', 1)[0]+'/' for p in texts if p.endswith('/operations.json')
                       and p.startswith(('collection-snapshots-', 'ffi-array-results-', 'ffi-result-release-')))
        assert len(roots) == 4
        assert sum(r.startswith('collection-snapshots-') for r in roots) == 1
        assert sum(r.startswith('ffi-array-results-') for r in roots) == 1
        assert sum(r.startswith('ffi-result-release-') for r in roots) == 2
        cli_count = 0
        for root in roots:
            ops = json.loads(texts[root+'operations.json'])
            inputs = json.loads(texts[root+'inputs-before-after.json'])
            for path, expected in inputs.items():
                if '/.test_output/' in path:
                    rel = path.split('/.test_output/', 1)[1]
                    actual = z.read(rel)
                    assert actual.decode() in literals, rel
                else:
                    rel = path.rsplit('/nanolang/', 1)[-1] if path.startswith('/') else path
                    actual = subprocess.check_output(['git', 'show', pin+':'+rel])
                assert digest(actual) == expected, path
            loader = root.startswith('ffi-result-release-')
            modes = ('plain', 'sanitized') if root.startswith('ffi-array-results-') else ('plain', 'sanitized', 'threaded')
            has_cli = loader and json.loads(texts[root+'scope.json'])['vm_cli_requested']
            labels = []
            for mode in modes:
                labels += [mode+'-'+x for x in ('dependency', 'origin', 'collections', 'driver', 'run')] if loader else ['build-'+mode, 'run-'+mode]
                if has_cli and mode == 'plain':
                    labels += ['cli-'+case+'-'+stage for case in ('copy', 'cross-image', 'input-alias') for stage in ('assemble', 'verify', 'execute')]
            assert [o['label'] for o in ops] == labels
            for op in ops:
                label = op['label']
                refusal = ('own image' if 'cross-image' in label else 'independent') if label.startswith('cli-') and label.endswith('-execute') and 'copy' not in label else None
                assert op['returncode'] == (1 if refusal else 0), label
                stderr = texts[root+label+'.stderr']
                if refusal:
                    assert refusal in stderr
                elif label.startswith(('run-', 'cli-')) or label.endswith('-run'):
                    assert stderr == '', label
                command = op.get('argv', op.get('command'))
                if label.startswith('build-') or (loader and label.split('-', 1)[0] in modes and not label.endswith('-run')):
                    absolute = command[-1]
                    rel = absolute.split('/.test_output/', 1)[1]
                    data = z.read(rel)
                    mode = label.split('-')[1] if label.startswith('build-') else label.split('-')[0]
                    if loader:
                        reports = json.loads(texts[root+'products.json'])
                        expected = next(p for p in reports if p['mode'] == mode)['sha256_before_after'][absolute]
                    else:
                        expected = json.loads(texts[root+mode+'.identity.json'])['sha256_before_after']
                    assert digest(data) == expected
                    shared = loader and not label.endswith('-driver')
                    kind = native(data, shared, mode, command)
                    products.append({'path': rel, 'bytes': len(data), 'sha256_before_after': expected, 'kind': kind, 'mode': mode,
                                     'instrumentation_symbols': {name: name.encode() in data for name in ('__asan_init', '__ubsan_handle', '__tsan_init')}})
                    to_retain[rel] = data
            if has_cli:
                cli_count += 1
                tools = json.loads(texts[root+'cli-tools-before-after.json'])
                assert len(tools) == 2 and {Path(p).name for p in tools} == {'nano_vm', 'nanoisa'}
                for case in ('copy', 'cross-image', 'input-alias'):
                    rel = root+'plain/'+case+'.nvm'
                    data = z.read(rel)
                    assert data[:4] == b'NVM\x02'
                    to_retain[rel] = data
            witnesses.append({'root': root, 'inputs': inputs, 'operations': ops, 'vm_cli': has_cli})
        assert cli_count == 1
        assert len(products) == 29 and sum('executable' in p['kind'] for p in products) == 11
    output.mkdir(parents=True)
    retained.mkdir(parents=True)
    for rel, data in to_retain.items():
        dest = retained/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(0o444)
    report = {'source_pin': pin, 'archive': {'bytes': archive.stat().st_size,
              'sha256': digest(archive.read_bytes()), 'entries': len(inventory), 'all_entries_crc_streamed': True},
              'products': products, 'witnesses': witnesses,
              'scope': 'I independently audit 11 native executables, 18 shared images and three NVM modules per platform. I match selected source and generated fixture inputs to immutable Git, check sanitizer symbols, command flags and actual success/refusal streams. The negative dependency cleanup is not called. VM/assembler hashes are reports; their binaries are absent from these archives. Header/compiler/SDK closure, source admission and release remain separate.'}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for name, value in (('texts', texts), ('inventory', inventory)):
        (output/(name+'.json.gz')).write_bytes(gzip.compress((json.dumps(value, sort_keys=True)+'\n').encode(), mtime=0))
    print(json.dumps({'pin': pin, 'native_products': len(products), 'entries': len(inventory)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('pin')
    p.add_argument('output', type=Path)
    p.add_argument('retained', type=Path)
    a = p.parse_args()
    audit(a.archive, a.pin, a.output, a.retained)
