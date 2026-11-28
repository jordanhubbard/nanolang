#!/usr/bin/env python3
"""I verify private real-source tracking while default and bounded heap gates remain open."""
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
    assert manifest['source_pin'] == '933083ea5f86f0b40252ff2423f4979267381d52'
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
    assert manifest['acceptance']==dict(private_source_tracking=True,private_record_union_lsan=True,private_scalar_source_lsan=True,generated_metadata=True,default_source_lsan=False,default_typed_list_acceptance=False,bounded_live_heap=False,whole_release=False)
    q=json.loads(files['journal/final-qualified.json']);assert len(q['commands'])==41
    failures={'san-fresh-default-scalar':2,'final-default-record':1,'final-default-api-replace':1,'final-default-source':1}
    for r in q['commands']:assert r['exit']==failures.get(r['label'],0)
    for p,h in q['files'].items():assert sha(files['source/'+p])==h
    def contains(name,text): assert text in files['journal/'+name]
    def shadows(name,names,refusal=False):
        raw=files['journal/'+name]
        events=[json.loads(line) for line in raw.splitlines() if line.startswith(b'{')]
        assert len(events)==2*len(names) and all(e['failures']==0 for e in events)
        for event in ('shadow-start','shadow-complete'):
            chosen=[e['name'] for e in events if e['event']==event]
            assert len(chosen)==len(names) and set(chosen)==set(names)
        if refusal:assert b'I currently borrow only fixed resource records with numeric or boolean fields' in raw
        assert b'AddressSanitizer:' not in raw and b'LeakSanitizer:' not in raw
    record_names=['lifetime_target','record_factory','record_case','union_factory','union_case','main']
    scalar_names=['tuple_case','try_case','try_error_case','unwrap_case','unwrap_error_case','fallback_case','main']
    for profile in ('normal','san'):
        contains(profile+'-eval-run.stdout',b'All eval tests passed!')
        assert files['journal/'+profile+'-eval-run.stdout'].count(b'  Testing ')==134
        contains(profile+'-checker-run.stdout',b'All typechecker tests passed!')
        assert files['journal/'+profile+'-checker-run.stdout'].count(b'  Testing ')==88
        contains(profile+'-metadata-run.stdout',b'four generated list signatures, duplicate admission, source independence and borrowed declarations across eight full metadata teardowns.')
        assert files['journal/'+profile+'-metadata-run.stderr']==b''
        contains(profile+'-record-run.stdout',b'registered=19 destroyed=18 transferred=1 live=0.')
        contains(profile+'-record-run.stdout',b'I preserve six independent nested-signature constructor values')
        contains(profile+'-scalar-run.stdout',b'registered=18 destroyed=16 transferred=2 live=0.')
        contains(profile+'-witness-source.stdout',b'registered=12 destroyed=12 transferred=0 live=0.')
        shadows(profile+'-record-run.stderr',record_names)
        shadows(profile+'-scalar-run.stderr',scalar_names,True)
        shadows(profile+'-witness-source.stderr',['observe_list','main'])
        for kind in ('control','replace'):
            assert files['journal/'+profile+'-witness-'+kind+'.stderr']==b''
            contains(profile+'-witness-'+kind+'.stdout',b'pointer identity before replacement.')
        contains(profile+'-test-eval-record-binding-ownership.stdout',b'991 allocations, 991 releases.')
        contains(profile+'-test-eval-carrier-heap.stdout',b'67667 carrier payload allocations and releases.')
        contains(profile+'-test-eval-carrier-values.stdout',b'18 injected graph-walk failures')
        contains(profile+'-test-eval-carrier-values.stdout',b'five actual creation/adoption allocation failures')
    for r in q['commands']:
        expected=not(r['label'] in ('san-eval-run','san-checker-run'))
        assert r['detect_leaks']==expected
    first=json.loads(files['journal/first-qualified.json'])
    assert [(r['label'],r['exit']) for r in first['commands']]==[('fresh-default-scalar',2),('witness-link',0),('witness-control',0),('witness-replace',0),('witness-source',1),('record-link',0),('record-run',1),('scalar-link',0),('scalar-run',1)]
    for name,summary in [('first-fresh-default-scalar.stderr',b'1937 byte(s) leaked in 82 allocation(s)'),('first-default-record.stderr',b'3720 byte(s) leaked in 159 allocation(s)'),('first-record-run.stderr',b'120 byte(s) leaked in 6 allocation(s)'),('first-scalar-run.stderr',b'80 byte(s) leaked in 4 allocation(s)'),('first-witness-source.stderr',b'379 byte(s) leaked in 14 allocation(s)'),('old-metadata-run.stderr',b'36632 byte(s) leaked in 1456 allocation(s)'),('san-fresh-default-scalar.stderr',b'1857 byte(s) leaked in 78 allocation(s)'),('final-default-record.stderr',b'3600 byte(s) leaked in 153 allocation(s)')]:contains(name,summary)
    for name in ('final-default-api-replace.stderr','final-default-source.stderr'):
        contains(name,b'heap-use-after-free');contains(name,b'env_set_binding_impl')
    contains('final-default-source.stderr',b'env_define_var_impl')
    growth=json.loads(files['journal/growth.json']);assert [r['exit'] for r in growth['commands']]==[0,0]
    contains('growth-run.stdout',b'start=18 after64=210 after128=402 destroyed=0.')
    contains('growth-run.stdout',b'registered=403 destroyed=402 transferred=1 live=0.')
    shadows('growth-run.stderr',record_names)
    # I regenerate each observer around the complete unchanged original fixture.
    prefix=b'''#include "'''+str(ROOT/'src/eval/carrier_runtime.h').encode()+b'''"
'''
    for name,path in [('record','tests/test_eval_callable_record_escape.c'),('scalar','tests/test_eval_scalar_projection_sources.c')]:
        assert files['journal/'+name+'-original.c']==files['source/'+path]
        body=files['source/'+path].replace(b'int main(',b'int fixture_main(',1)
        body=body.replace(b'#include "test_eval.c"',b'#include "'+str(ROOT/'tests/test_eval.c').encode()+b'"')
        assert body in files['journal/'+name+'-driver.c']
        assert files['journal/'+name+'-driver.c'].startswith(prefix)
    assert b'observed_enable=false' in files['journal/record-driver.c']
    assert b'env_carrier_enable(env)' in files['journal/record-driver.c']
    basepath=HERE/manifest['base_archives'][0]['path'];base=json.loads(basepath.read_text());prior={r['path']:r for r in base['git_sources']}
    changed={'src/env.c','src/eval.c','src/nanolang.h','src/eval/record_carrier_cleanup.inc','src/eval/carrier_values.c','src/eval/carrier_values.h','src/eval/carrier_runtime.c','src/eval/carrier_runtime.h','tests/test_env_generated_list_metadata.c','Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md'}
    for p,row in prior.items():
        if p not in changed:assert sha(files['source/'+p])==row['sha256']
    field=b'    struct NlEvalCarrierSession *carrier_session; /* Private, explicitly enabled qualification ownership. */\n'
    assert files['source/src/nanolang.h'].count(field)==1
    assert sha(files['source/src/nanolang.h'].replace(field,b''))==prior['src/nanolang.h']['sha256']
    enablers=[p for p,raw in files.items() if p.startswith('source/src/') and b'env_carrier_enable(' in raw]
    assert set(enablers)=={'source/src/eval/carrier_runtime.c','source/src/eval/carrier_runtime.h'}
    contains('final-default-api-replace.stdout',b'pointer identity before replacement.')
    tools=json.loads(files['journal/current-tools.json']);assert tools['host'][4]=='aarch64'
    assert '13.3.0' in tools['cc']['stdout'] and '2.42' in tools['as']['stdout']
    assert tools['cc']['exit']==tools['as']['exit']==0
    for p,row in json.loads(files['journal/image-guards.json']).items():
        key='images/bin/nanoc_stage2' if p=='bin/nanoc' else 'images/'+p
        assert (len(files[key]),sha(files[key]))==(row['bytes'],row['sha256'])
    identities=json.loads(files['journal/retained-inputs.json'])
    assert len(manifest['product_paths'])==15
    for p in manifest['product_paths']:
        raw=files['products/'+str(Path(p).relative_to('/tmp'))]
        assert raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little')==183
        assert (len(raw),sha(raw))==(identities['products'][p]['bytes'],identities['products'][p]['sha256'])
    actual=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines();assert set(actual)==changed
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; private source LSan clean; default/bounded-heap acceptance remains open')

if __name__ == '__main__':
    main()
