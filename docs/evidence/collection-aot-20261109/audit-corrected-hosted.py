"""I independently audit retained AOT products without rerunning hosted programs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zipfile

PIN = 'b897da5d404d6b148d2734b3e4673ebcc8d81b3f'
CASES = ('both-dependency', 'companion-dependency', 'marker-dependency', 'bad-marker', 'missing-companion', 'missing-marker')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def audit(archive, output, retained):
    assert not output.exists() and not retained.exists()
    inventory, texts, products, saved = [], {}, [], {}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        for info in z.infolist():
            assert not info.filename.startswith('/') and '..' not in Path(info.filename).parts
            if info.is_dir():
                continue
            h = hashlib.sha256()
            with z.open(info) as stream:
                for block in iter(lambda: stream.read(65536), b''):
                    h.update(block)
            inventory.append({'path': info.filename, 'bytes': info.file_size, 'sha256': h.hexdigest()})
            if Path(info.filename).suffix in ('.json', '.txt', '.log', '.stdout', '.stderr', '.c', '.nasm', '.sha256'):
                texts[info.filename] = z.read(info).decode(errors='replace')
        roots = [p.removesuffix('products.json') for p in texts if p.startswith('collection-aot-') and p.endswith('/products.json')]
        assert len(roots) == 1
        root = roots[0]
        identity = json.loads(texts[root+'inputs-tools-before-after.json'])
        assert len(identity['inputs']) == 13 and len(identity['tools']) == 4
        for path, expected in identity['inputs'].items():
            assert digest(subprocess.check_output(['git', 'show', PIN+':'+path])) == expected, path
        uploaded_tool_scripts = []
        for path, expected in identity['tools'].items():
            matches = [n for n in names if Path(n).name == Path(path).name]
            if matches:
                assert len(matches) == 1
                relative = matches[0]
                data = z.read(relative)
                assert '/.test_output/' in path and relative == path.split('/.test_output/', 1)[1]
                assert digest(data) == expected and data.startswith(b'#!')
                saved[relative] = data
                texts[relative] = data.decode()
                uploaded_tool_scripts.append({'path':relative, 'bytes':len(data), 'sha256':expected})
        ops = json.loads(texts[root+'operations.json'])
        assert len(ops) == 238 and len({o['label'] for o in ops}) == 238
        refusals = {'test_companion_and_marker_origin_refusals-'+case+'-refuse'+opt for case in CASES for opt in ('-O0','-O2')}
        successes = {prefix+opt for prefix in ('test_actual_emitted_copy_failures-copy-control-run', 'test_real_collection_vm_and_native-all-run', 'test_real_json_and_filesystem_array_types-json-fs-run') for opt in ('-O0','-O2')}
        malformed = [o for o in ops if o['label'].startswith('test_all_malformed_signatures_preserve_output-') and o['label'].endswith('-refuse')]
        assert len(malformed) == 52
        malformed_labels = {o['label'] for o in malformed}
        observed = {o['label'] for o in ops}
        assert refusals | successes <= observed
        for op in ops:
            label = op['label']
            assert op['returncode'] == (-6 if label in refusals else 1 if label in malformed_labels else 0), label
            stdout, stderr = (z.read(root+label+suffix) for suffix in ('.stdout', '.stderr'))
            assert b'ERROR: AddressSanitizer' not in stderr and b'runtime error:' not in stderr, label
            if label in refusals:
                assert stdout == b'' and b'I stopped at a native invariant' in stderr
            elif label in malformed_labels:
                assert b'typed value adapter' in stderr
                target = root+Path(op['argv'][-1]).name
                assert z.read(target) == b'I retain the prior product.\n'
            elif label in successes:
                if 'copy-control-run' in label:
                    assert b'seven allocation denials' in stdout
                    assert stderr.count(b'I stopped at a native invariant') == 19
                else:
                    assert stdout == stderr == b''
            elif label.endswith('-all-vm'):
                assert stdout == stderr == b''
        reports = json.loads(texts[root+'products.json'])
        assert len(reports) == 34 and sum(p['shared'] for p in reports.values()) == 16
        for path, report in reports.items():
            relative = path.split('/.test_output/', 1)[1]
            assert relative.startswith(root)
            data = z.read(relative)
            assert digest(data) == report['sha256']
            builds = [o for o in ops if o['argv'][-2:] == ['-o', path]]
            assert len(builds) == 1
            command = builds[0]['argv']
            assert all(flag in command for flag in ('-Werror', '-UNDEBUG', '-pthread'))
            if data[:4] == b'\xcf\xfa\xed\xfe':
                assert struct.unpack_from('<I', data, 12)[0] == (6 if report['shared'] else 2)
                kind = 'Mach-O dylib' if report['shared'] else 'Mach-O executable'
            else:
                assert data[:4] == b'\x7fELF' and data[5] in (1,2)
                value = struct.unpack_from('<H' if data[5] == 1 else '>H', data, 16)[0]
                assert value in ((3,) if report['shared'] else (2,3))
                kind = 'ELF shared object' if report['shared'] else 'ELF executable/PIE'
            sanitized = '-fsanitize=address,undefined' in command
            if not report['shared']:
                assert sanitized and '-fno-sanitize-recover=all' in command
                assert b'__asan_init' in data and b'__ubsan_handle' in data
                execution = [o for o in ops if o['argv'] == [path]]
                assert len(execution) == 1 and execution[0]['label'] in refusals | successes
            elif sanitized:
                assert b'__asan_init' in data and b'__ubsan_handle' in data
            saved[relative] = data
            products.append({'path':relative, 'bytes':len(data), **report, 'kind':kind, 'sanitized':sanitized, 'build':builds[0]['label']})
        for p in names:
            if p.startswith(root) and p.endswith('.nvm'):
                data = z.read(p)
                assert data[:4] == b'NVM\x02'
                saved[p] = data
        assert not any(p.startswith('bin/') for p in names)
    output.mkdir(parents=True)
    retained.mkdir(parents=True)
    for p, data in saved.items():
        target = retained/p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        assert digest(target.read_bytes()) == digest(data)
    for name, value in (('inventory',inventory), ('texts',texts)):
        with gzip.open(output/(name+'.json.gz'), 'wt') as f:
            json.dump(value, f, sort_keys=True)
    report = {'source_pin':PIN, 'archive':{'bytes':archive.stat().st_size, 'sha256':digest(archive.read_bytes()), 'entries':len(inventory), 'all_entries_streamed_crc_checked':True}, 'root':root, 'source_inputs_git_matched':identity['inputs'], 'tools_reported_before_after':identity['tools'], 'reported_native_tool_bytes_uploaded':False, 'uploaded_tool_scripts':uploaded_tool_scripts, 'operations':ops, 'products':products, 'retained_nvm':sum(p.endswith('.nvm') for p in saved), 'successful_native_runs':6, 'expected_native_refusals':12, 'malformed_signature_refusals':52, 'sdk_and_header_closure_qualified':False, 'historical_deadline_cause_established':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(output), 'images':len(products), 'ops':len(ops), 'entries':len(inventory)}))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('archive',type=Path); p.add_argument('output',type=Path); p.add_argument('retained',type=Path)
    a=p.parse_args(); audit(a.archive,a.output,a.retained)
