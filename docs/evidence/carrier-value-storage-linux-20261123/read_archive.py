#!/usr/bin/env python3
"""I verify actual carrier storage without claiming source activation."""
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
    assert manifest['source_pin'] == '6e8054c87a7b8fe7be9c240eb825c8c3005b730f'
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
    assert manifest['acceptance'] == dict(storage_adapter=True,source_activation=False,typed_list_source_fix=False,full_source_lsan=False,whole_release=False)
    q = json.loads(files['journal/final-qualified.json'])
    assert len(q['commands'])==12
    assert all(r['exit']==0 for r in q['commands'])
    for p,h in q['files'].items():
        assert sha(files['source/'+p])==h
    expected = b'''I leave payload ownership with the caller through five actual creation/adoption allocation failures.
I retire actual nested record/union/tuple copies while preserving shared arrays and borrowed cyclic containers.
I preserve duplicate/multiple typed-list roots and explicit field facts, refuse integer guessing and never adopt a borrowed list.
I preserve nested union/record/STRING/callable/array payloads through an actual typed-list root.
I retire caller-created tuple storage containing an owned managed STRING.
I transfer complete actual graphs before caller teardown and retire STRING/FUNCTION arrays and a self-referential static array.
I retire packed/legacy/empty names and scalar metadata without owning referenced caller record elements.
I preserve all 129 actual carriers across 18 injected graph-walk failures, then recover and release all metadata.
I complete actual carrier adapter controls; interpreter source activation remains pending.
'''
    for mode in ('normal','optimized','san'):
        assert files['journal/'+mode+'-final-run.stdout']==expected
        assert files['journal/'+mode+'-matched-tuple-run.stdout']==b'I retire caller-created tuple storage containing an owned managed STRING.\n'
        for part in ('final-compile','final-link','final-run','matched-tuple-run'):
            assert files['journal/'+mode+'-'+part+'.stderr']==b''
        cmd=next(r['command'] for r in q['commands'] if r['label']==mode+'-final-compile')
        assert '-Werror' in cmd and '-std=c99' in cmd
        assert ('-O2' in cmd)==(mode=='optimized')
        assert ('-fsanitize=address,undefined' in cmd)==(mode=='san')
    assert b"env['ASAN_OPTIONS']='detect_leaks=1'" in files['journal/qualify-final.py']
    first=json.loads(files['journal/first-qualified.json'])
    assert len(first['commands'])==1 and first['commands'][0]['exit']==2
    report=files['journal/first-normal-test-eval-carrier-values.stderr']
    assert b'undefined reference' in report and b'g_argc' in report and b'g_argv' in report
    assert b'I retire actual' not in files['journal/first-normal-test-eval-carrier-values.stdout']
    for p,h in first['files'].items():
        path='journal/first-'+Path(p).name
        if path in files: assert sha(files[path])==h
    second=json.loads(files['journal/second-qualified.json'])
    assert len(second['commands'])==4 and all(r['exit']==0 for r in second['commands'])
    for mode in ('normal','san'):
        assert files['journal/second-'+mode+'-test-eval-carrier-values.stderr']==b''
        assert files['journal/second-'+mode+'-test-eval-carrier-heap.stderr']==b''
        text=files['journal/second-'+mode+'-test-eval-carrier-heap.stdout']
        assert b'I balance 67667 carrier payload allocations and releases.' in text
        assert b'through six allocation failures and retire all registry metadata.' in text
    assert second['files']['src/eval/carrier_heap.c']==q['files']['src/eval/carrier_heap.c']
    assert second['files']['src/eval/carrier_heap.h']==q['files']['src/eval/carrier_heap.h']
    old=json.loads(files['journal/old-tuple.json'])
    assert [r['exit'] for r in old['commands']]==[0,1]
    report=files['journal/old-tuple-run.stderr']
    assert b'AddressSanitizer: attempting free on address which was not malloc()-ed' in report
    assert b'free_tuple src/env.c' in report and b'destroy_value src/eval/carrier_values.c' in report
    assert files['journal/old-tuple-run.stdout']==b''
    assert b'case VAL_TUPLE: free_tuple(object); break;' in files['journal/second-carrier_values.c']
    assert b'case VAL_TUPLE: free_tuple(object); break;' not in files['source/src/eval/carrier_values.c']
    assert b'gc_get_header(value_key(ref))->type != GC_TYPE_ARRAY' in files['source/src/eval/carrier_values.c']
    assert b'(void **)' not in files['source/src/eval/carrier_values.c']
    # My matched managed-tuple witness remains byte-identical after other cases are added.
    def body(raw,name):
        start=raw.index(b'static void '+name+b'(void) {'); end=start+raw[start:].index(b'\n}\n')+3
        return raw[start:end]
    assert body(files['journal/fourth-test_eval_carrier_values.c'],b'managed_tuple_storage')==body(files['source/tests/test_eval_carrier_values.c'],b'managed_tuple_storage')
    identities=json.loads(files['journal/retained-inputs.json'])
    assert len(identities['products'])==15 and len(identities['providers'])==227
    for p,row in q['providers'].items(): assert identities['providers'][p]==row
    assert len(manifest['product_paths'])==8
    for path in manifest['product_paths']:
        raw=files['products/'+str(Path(path).relative_to('/tmp'))]
        assert raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little')==183
        assert (len(raw),sha(raw))==(identities['products'][path]['bytes'],identities['products'][path]['sha256'])
    tools=json.loads(files['journal/current-tools.json'])
    assert tools['host'][4]=='aarch64' and '13.3.0' in tools['cc']['stdout'] and '2.42' in tools['as']['stdout']
    assert tools['cc']['exit']==tools['as']['exit']==0
    for path,row in json.loads(files['journal/image-guards.json']).items():
        key='images/bin/nanoc_stage2' if path=='bin/nanoc' else 'images/'+path
        assert (len(files[key]),sha(files[key]))==(row['bytes'],row['sha256'])
    basepath=HERE/manifest['base_archives'][0]['path'];base=json.loads(basepath.read_text())
    prior={r['path']:r for r in base['git_sources']}
    allowed={'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md','src/eval/carrier_heap.c','src/eval/carrier_heap.h','tests/test_eval_carrier_heap.c'}
    for p,row in prior.items():
        if p not in allowed: assert sha(files['source/'+p])==row['sha256']
    make=files['source/Makefile.gnu']
    start=make.index(b'.PHONY: test-eval-carrier-values\n')
    end=make.index(b'.PHONY: test-eval-scalar-projection-ownership',start)
    assert sha(make[:start]+make[end:])==prior['Makefile.gnu']['sha256']
    checks=b'    CHECK(nl_carrier_heap_context(ctx.heap, value) == &ctx);\n    CHECK(nl_carrier_heap_context(ctx.heap, spare) == NULL);\n'
    current=files['source/tests/test_eval_carrier_heap.c'];assert current.count(checks)==1
    assert sha(current.replace(checks,b''))==prior['tests/test_eval_carrier_heap.c']['sha256']
    changed=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines()
    assert set(changed)==allowed|{'src/eval/carrier_values.c','src/eval/carrier_values.h','tests/test_eval_carrier_values.c','tests/support/carrier_value_allocations.h'}
    assert b'Actual evaluator/environment/GC paths remain unchanged.' in files['source/docs/EVAL_CARRIER_OWNERSHIP.md']
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; eight archived executable products; actual storage adapter only, source activation remains open')

if __name__ == '__main__':
    main()
