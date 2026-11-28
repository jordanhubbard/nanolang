#!/usr/bin/env python3
"""I verify private active source roots while complete default ownership remains open."""
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
    assert manifest['source_pin'] == '9e4f961f66abdc5f060dd58943e81617bcfecc0b'
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
    assert manifest['acceptance']==dict(private_active_roots=True,private_source_lsan=True,selected_bounded_carrier_counts=True,native_live_bytes_bounded=False,explicit_host_roots=True,automatic_external_gc_lifetimes=False,default_source_ownership=False,complete_bounded_heap=False,whole_release=False)
    def contains(name,text): assert text in files['journal/'+name], name
    def data(name):return json.loads(files['journal/'+name])
    q=data('final-qualified.json');assert len(q['commands'])==48
    failures={'default-scalar':2,'default-record':1,'default-api-replace':1,'default-source':1}
    for r in q['commands']:assert r['exit']==failures.get(r['label'],0),r['label']
    assert {r['label'] for r in q['commands'] if not r['detect_leaks']}=={'san-eval-run','san-checker-run'}
    for p,h in q['files'].items():assert sha(files['source/'+p])==h,p
    def shadows(name,names):
        raw=files['journal/'+name];events=[json.loads(l) for l in raw.splitlines() if l.startswith(b'{')]
        assert len(events)==len(names)*2
        for kind in ('shadow-start','shadow-complete'):
            found=[e['name'] for e in events if e['event']==kind]
            assert len(found)==len(names) and set(found)==set(names)
        assert all(e['failures']==0 for e in events)
        assert b'AddressSanitizer:' not in raw and b'LeakSanitizer:' not in raw
    record=['lifetime_target','record_factory','record_case','union_factory','union_case','main']
    scalar=['tuple_case','try_case','try_error_case','unwrap_case','unwrap_error_case','fallback_case','main']
    active=['root_item','root_sum','root_siblings','root_loop','root_for','root_list','root_box','root_choice','root_tuple','main']
    for profile in ('normal','san'):
        shadows('final-'+profile+'-fresh-active.stderr',active)
        contains('final-'+profile+'-fresh-active.stdout',b'start=0 after64=7 after128=7 peak=7 destroyed=3755.')
        for name,names in [('record',record),('record-growth',record),('scalar',scalar),('witness',['observe_list','main'])]:
            suffix='source' if name=='witness' else 'run'
            shadows('final-'+profile+'-'+name+'-'+suffix+'.stderr',names)
        contains('final-'+profile+'-record-growth-run.stdout',b'start=0 after64=1 after128=1 destroyed=401.')
        contains('final-'+profile+'-record-growth-run.stdout',b'registered=403 destroyed=402 transferred=1 live=0.')
        contains('final-'+profile+'-record-run.stdout',b'registered=19 destroyed=18 transferred=1 live=0.')
        contains('final-'+profile+'-scalar-run.stdout',b'registered=18 destroyed=16 transferred=2 live=0.')
        contains('final-'+profile+'-witness-source.stdout',b'registered=12 destroyed=12 transferred=0 live=0.')
        contains('final-'+profile+'-scalar-run.stderr',b'I currently borrow only fixed resource records with numeric or boolean fields')
        for name,expected in [('eval',134),('checker',88)]:
            raw=files['journal/final-'+profile+'-'+name+'-run.stdout'];assert raw.count(b'  Testing ')==expected
            assert b'All eval tests passed!' in raw if name=='eval' else b'All typechecker tests passed!' in raw
        assert files['journal/final-'+profile+'-metadata-run.stderr']==b''
        for mode in ('control','replace'):
            assert files['journal/final-'+profile+'-host-'+mode+'.stderr']==b''
            contains('final-'+profile+'-host-'+mode+'.stdout',b'pointer identity before replacement.')
            contains('final-'+profile+'-host-'+mode+'.stdout',b'finish explicit caller teardown.')
        contains('final-'+profile+'-host-replace.stdout',b'registered=3 destroyed=2 transferred=1 live=0.')
        contains('final-'+profile+'-test-eval-record-binding-ownership.stdout',b'991')
        contains('final-'+profile+'-test-eval-carrier-heap.stdout',b'67667')
        contains('final-'+profile+'-test-eval-carrier-values.stdout',b'18 injected graph-walk failures')
    for r in data('supplement.json')['commands']:assert r['exit']==0
    contains('tracking-only-growth-run.stdout',b'start=18 after64=210 after128=402 destroyed=0.')
    contains('tracking-only-growth-run.stdout',b'registered=403 destroyed=402 transferred=1 live=0.')
    shadows('tracking-only-growth-run.stderr',record)
    contains('final-default-scalar.stderr',b'1857 byte(s) leaked in 78 allocation(s)')
    contains('final-default-record.stderr',b'3600 byte(s) leaked in 153 allocation(s)')
    for name in ('api-replace','source'):
        contains('final-default-'+name+'.stderr',b'ERROR: AddressSanitizer: heap-use-after-free')
        contains('final-default-'+name+'.stderr',b'env_set_binding_impl')
    contains('witness-run.stderr',b'I cannot trace active interpreter roots.')
    contains('witness-diagnostic-run.stderr',b'root symbol items value-kind=12')
    contains('active-run.stderr',b'ERROR: AddressSanitizer: heap-use-after-free')
    contains('active-run.stderr',b'"name":"root_box"')
    contains('for-first.stderr',b'E034 EXPRESSION HAS NO EFFECT')
    contains('for-second.stderr',b'ERROR: AddressSanitizer: heap-use-after-free')
    contains('for-second.stderr',b'"name":"root_loop"')
    assert data('active-first.json')['commands'][-1]['exit']==1
    assert data('for-first.json')['exit']==data('for-second.json')['exit']==1
    first_source=files['journal/first-active-source.nano'].splitlines()
    events=[json.loads(l) for l in files['journal/active-run.stderr'].splitlines() if l.startswith(b'{')]
    for e in events:assert first_source[e['line']-1].startswith(('shadow '+e['name']+' ').encode())
    assert files['journal/for-second-source.nano']==files['source/tests/fixtures/carrier_active_roots.nano']
    for name,path in [('record','tests/test_eval_callable_record_escape.c'),('scalar','tests/test_eval_scalar_projection_sources.c')]:
        original=files['source/'+path].replace(b'int main(',b'int fixture_main(',1)
        original=original.replace(b'#include "test_eval.c"',b'#include "'+str(ROOT/'tests/test_eval.c').encode()+b'"')
        assert original in files['journal/'+name+'-driver.c']
    assert b'env_carrier_enable_collection(env, NULL, 0)' in files['journal/record-driver.c']
    assert b'ASSERT(env_carrier_collect(env));' in files['journal/host-driver.c']
    assert b'ASSERT(env_carrier_enable_collection(env,&root,1));' in files['journal/host-driver.c']
    base=json.loads((HERE/manifest['base_archives'][0]['path']).read_text());prior={r['path']:r for r in base['git_sources']}
    changed={'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md','src/eval.c','src/eval/carrier_runtime.c','src/eval/carrier_runtime.h','src/eval/carrier_values.c','src/eval/carrier_values.h','src/nanolang.h','src/typechecker.c','tests/fixtures/carrier_active_roots.nano','tests/test_eval_carrier_active_roots.c'}
    for p,r in prior.items():
        if p not in changed:assert sha(files['source/'+p])==r['sha256'],p
    bit=b'    bool checked_record_list; /* Checked result provenance for private interpreter roots. */\n'
    assert files['source/src/nanolang.h'].count(bit)==1
    assert sha(files['source/src/nanolang.h'].replace(bit,b''))==prior['src/nanolang.h']['sha256']
    checker=files['source/src/typechecker.c']
    for line in (b'    expr->checked_record_list = result == TYPE_LIST_GENERIC;\n',b'    stmt->checked_record_list = result == TYPE_LIST_GENERIC;\n'):
        assert checker.count(line)==1;checker=checker.replace(line,b'')
    assert sha(checker)==prior['src/typechecker.c']['sha256']
    for token in (b'env_carrier_enable(',b'env_carrier_enable_collection('):
        enablers={p for p,raw in files.items() if p.startswith('source/src/') and token in raw}
        assert enablers=={'source/src/eval/carrier_runtime.c','source/src/eval/carrier_runtime.h'}
    tools=data('current-tools.json');assert tools['host'][4]=='aarch64'
    assert '13.3.0' in tools['cc']['stdout'] and '2.42' in tools['as']['stdout']
    assert tools['cc']['exit']==tools['as']['exit']==0
    for p,r in data('image-guards.json').items():
        key='images/bin/nanoc_stage2' if p=='bin/nanoc' else 'images/'+p
        assert (len(files[key]),sha(files[key]))==(r['bytes'],r['sha256'])
    ids=data('retained-inputs.json');assert len(ids['products'])==56 and len(ids['objects'])==534
    assert len(manifest['product_paths'])==26
    for p in manifest['product_paths']:
        raw=files['products/'+str(Path(p).relative_to('/tmp'))]
        assert raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little')==183
        assert (len(raw),sha(raw))==(ids['products'][p]['bytes'],ids['products'][p]['sha256'])
    actual=subprocess.check_output(['git','diff','--name-only',base['source_pin'],manifest['source_pin'],'--',*paths],cwd=ROOT,text=True).splitlines();assert set(actual)==changed
    for name in ('native-allocation.json','native-allocation-gc.json','native-allocation-localize.json'):
        for r in data(name)['commands']:assert r['exit']==(0 if r['label'].endswith('link') else 94)
    assert all(r['exit']==1 for r in data('first-native-allocation.json')['commands'])
    contains('first-normal-allocation-link.stderr',b'redefinition')
    contains('normal-allocation-run.stdout',b'positive=2101904 before=260480 after64=300320 after128=337184 delta=36864 calls=128.')
    contains('san-allocation-run.stdout',b'positive=2097152 before=271873 after64=299497 after128=325225 delta=25728 calls=128.')
    contains('gc-allocation-run.stdout',b'first=53499/1049 middle=73287/1437 last=92871/1821.')
    contains('allocation-localize-run.stdout',b'I retain 384 live strings and 19584 managed bytes from the final64 source calls.')
    raw=files['journal/allocation-localize-run.stdout'].decode()
    pcs={}
    for line in raw.splitlines():
        if line.startswith('surviving-string '):
            key=tuple(line.split()[1:]);pcs[key]=pcs.get(key,0)+1
    sites=data('native-allocation-sites.json');assert len(sites)==3 and all(r['exit']==0 and r['count']==128 for r in sites)
    for r in sites:assert pcs[tuple(r['pcs'])]==r['count']
    assert sum(r['count'] for r in sites if 'eval_expression_impl' in r['stdout'] and 'eval_record_field_value' not in r['stdout'])==256
    assert sum(r['count'] for r in sites if 'eval_record_field_value' in r['stdout'])==128
    assert 384*51==19584 and 19584+384*16==25728
    for profile in ('normal','san'):shadows(profile+'-allocation-run.stderr',active)
    shadows('allocation-localize-run.stderr',active)
    original=files['source/tests/test_eval_carrier_active_roots.c'].replace(b'int main(void)',b'int fixture_main(void)')
    original=original.replace(b'#include "../src/',b'#include "'+str(ROOT).encode()+b'/src/').replace(b'#include "test_eval.c"',b'#include "'+str(ROOT/'tests/test_eval.c').encode()+b'"')
    assert original in files['journal/native-allocation-driver.c']
    assert b'after>before+4096)return 93' in files['journal/native-allocation-driver.c']
    assert b'last_bytes>middle_bytes+1024)return 94' in files['journal/native-allocation-driver.c']
    assert 'GNU addr2line' in tools['addr2line']['stdout'] and tools['addr2line']['exit']==0
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; selected carrier counts bounded; native-byte failure retained; default/complete ownership remains open')

if __name__=='__main__':main()
