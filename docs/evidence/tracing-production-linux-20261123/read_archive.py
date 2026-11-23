#!/usr/bin/env python3
"""I verify production tracing builds without claiming unwired CLI features."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'].startswith('829be95a26')
    for row in manifest['base_archives']:
        assert sha((HERE/row['path']).read_bytes()) == row['sha256']
    files, objects, descriptors = {}, {}, {}
    for row in manifest['files']:
        assert row['path'] not in files
        descriptor = tuple(row[k] for k in ('artifact','bytes','compressed_bytes','compressed_sha256'))
        h = row['sha256']
        if h not in objects:
            path = (HERE/row['artifact']).resolve(strict=True)
            assert path.is_relative_to((ROOT/'docs/evidence').resolve())
            packed = path.read_bytes()
            assert (len(packed),sha(packed)) == (row['compressed_bytes'],row['compressed_sha256'])
            raw = gzip.decompress(packed)
            assert (len(raw),sha(raw)) == (row['bytes'],h)
            objects[h], descriptors[h] = raw, descriptor
        assert descriptors[h] == descriptor
        files[row['path']] = objects[h]
    counts = manifest['counts']
    assert len(files) == counts['records']
    assert len(objects) == counts['unique_objects']
    assert sum(map(len,files.values())) == counts['logical_bytes']
    own = {Path(r['artifact']).name for r in manifest['files'] if Path(r['artifact']).parent == Path('objects')}
    assert own == {p.name for p in (HERE/'objects').iterdir()}
    paths = [r['path'] for r in manifest['git_sources']]
    tree = subprocess.check_output(['git','ls-tree','-rz',manifest['source_pin'],'--',*paths],cwd=ROOT)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            meta,p = item.split(b'\t',1)
            blobs[p.decode()] = meta.split()[2].decode()
    assert len(blobs) == len(paths) == counts['sources']
    for row in manifest['git_sources']:
        raw = files['source/'+row['path']]
        assert (len(raw),sha(raw)) == (row['bytes'],row['sha256'])
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob'] == blobs[row['path']]
    q = json.loads(files['journal/archive-qualification.json'])
    assert q['source_pin'] == manifest['source_pin']
    for key in ('standalone_trace_flag_acceptance','automatic_cli_shadow_acceptance','whole_record_union_leak_acceptance','deadline_changed'):
        assert q[key] is False
    assert q['baseline']['exit'] == 2
    failed = files['journal/production-baseline.stderr']
    assert b'undefined reference' in failed and b'g_tracing_config' in failed and b'trace_var_set' in failed
    qualified = q['qualified']
    assert len(qualified['commands']) == 35
    for row in qualified['commands']:
        assert row['exit'] == (2 if row['label']=='san-test-eval-callable-record-escape' else 0)
    for path,h in qualified['files'].items():
        assert sha(files['source/'+path]) == h
    for path,h in q['before']['files'].items():
        if path != 'Makefile.gnu':
            assert sha(files['source/'+path]) == h
    before = files['journal/before-Makefile.gnu']
    assert sha(before) == q['before']['files']['Makefile.gnu']
    line = next(l for l in before.splitlines(keepends=True) if l.startswith(b'COMMON_SOURCES = '))
    after = before.replace(line,line.rstrip(b'\n')+b' $(SRC_DIR)/tracing.c\n')
    line = next(l for l in after.splitlines(keepends=True) if b'tests/test_eval_assignment_tracing.c ' in l)
    after = after.replace(line,line.replace(b'$(filter-out $(OBJ_DIR)/eval.o',b'$(filter-out $(OBJ_DIR)/tracing.o $(OBJ_DIR)/eval.o'))
    assert after == files['source/Makefile.gnu']
    gaps = json.loads(files['journal/cli-gaps.json'])
    assert gaps['issue'] == 1209 and gaps['contract_accepted'] is False
    assert gaps['source_pin'] == manifest['source_pin']
    assert gaps['image_sha256'] == sha(files['products/normal/nano'])
    assert [r['exit'] for r in gaps['commands']] == [1,0]
    assert b'--trace-all' in files['journal/unwired-trace.stderr']
    assert b'assert false' in files['journal/shadow-refusal.nano']
    assert files['journal/unrun-shadow.stdout'] == b'UNEXPECTED_SHADOW_ACCEPT\n'
    profile = json.loads(files['journal/tool-profile.json'])
    assert profile['host'][4] == 'aarch64'
    assert all(profile[k]['exit'] == 0 for k in ('cc','as','ld'))
    assert '13.3.0' in profile['cc']['stdout']
    for mode in ('normal','san','default'):
        row = next(r for r in qualified['commands'] if r['label']==mode+'-product-build')
        flags = next(a for a in row['command'] if a.startswith('CFLAGS='))
        assert ('-DNANO_INTERPRETER' in flags) == (mode!='default')
        assert ('-fsanitize=address,undefined' in flags) == (mode=='san')
        assert files['journal/'+mode+'-product-smoke.stdout'] == b'TRACE_PRODUCTION_OK\n'
        assert files['journal/'+mode+'-product-smoke.stderr'] == b''
        assert b'Usage:' in files['journal/'+mode+'-product-help.stdout']
        assert b'nano (nanolang interpreter)' in files['journal/'+mode+'-product-version.stdout']
        text = files['journal/'+mode+'-test-eval-assignment-tracing.stdout']
        assert text.count(b' VAR_SET ') == 8 and text.count(b' FUNCTION_CALL runtime_kind:') == 14
        assert text.count(b'I preserve live assignment values with tracing enabled=') == 3
        assert b'I label all fourteen runtime kinds independently of declaration ordinals.' in text
        assert files['journal/'+mode+'-test-eval-assignment-tracing.stderr'] == b''
    for mode in ('normal','san'):
        text = files['journal/'+mode+'-test_eval-run.stdout']
        assert b'All eval tests passed!' in text and text.count(b'  Testing ') == 134
        text = files['journal/'+mode+'-test_typechecker-run.stdout']
        assert b'All typechecker tests passed!' in text and text.count(b'  Testing ') == 88
        assert b'991 allocations, 991 releases.' in files['journal/'+mode+'-test-eval-record-binding-ownership.stdout']
        assert files['journal/'+mode+'-test-eval-callable-expression-ownership.stderr'] == b''
        assert b'I retain 0 callable names across nine 64-call routes.' in files['journal/'+mode+'-test-eval-callable-lifetimes.stdout']
    report = files['journal/san-test-eval-callable-record-escape.stderr']
    assert b'3720 byte(s) leaked in 159 allocation(s)' in report
    assert b'heap-use-after-free' not in report
    base = json.loads((HERE/manifest['base_archives'][0]['path']).read_text())
    prior = {r['path']:r for r in base['git_sources']}
    for path in ('src/env.c','src/eval.c','src/tracing.c','src/tracing.h','src/nano_main.c','src/nanolang.h','src/nanoisa/nvm2c.c','tests/test_eval_assignment_tracing.c','tests/test_eval.c','tests/test_typechecker.c','tests/test_eval_record_binding_ownership.c','tests/test_eval_callable_expression_ownership.c','tests/test_eval_callable_record_escape.c','tests/test_eval_callable_lifetimes.c','tests/test_env_borrowed_values.c','tests/test_env_symbol_index.c'):
        assert sha(files['source/'+path]) == prior[path]['sha256']
    for row in base['files']:
        if row['path'].startswith('images/'):
            assert sha(files[row['path']]) == row['sha256']
    assert all(raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little')==183 for path,raw in files.items() if path.startswith('products/'))
    doc = files['source/docs/FEATURES.md']
    assert b'#1209' in doc and b'current CLI does not wire them yet' in doc
    for flag in (b'--trace-all',b'--trace-function=',b'--trace-var=',b'--trace-scope=',b'--trace-regex='):
        assert flag in doc
    changed = subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines()
    assert set(changed) == {'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CALLABLE_OWNERSHIP.md','docs/FEATURES.md'}
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; three production profiles; CLI contract remains open')

if __name__ == '__main__':
    main()
