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
    assert manifest['source_pin'] == 'c4e2aa0500e918fb0304a0734d4cedb30e1c0af6'
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
    assert manifest['acceptance']==dict(private_string_scopes=True,private_source_alias_controls=True,checker_owned_loop_trees=True,nested_loop_elements=True,foreign_argument_refusals=True,actual_array_reclaim_counter=True,selected_asan_native_bytes=True,ordinary_native_byte_criterion=False,ordinary_sibling_safety=False,default_source_ownership=False,complete_bounded_heap=False,whole_release=False)
    from collections import Counter
    def data(p):return json.loads(files['journal/'+p])
    def contains(p,t):assert t in files['journal/'+p],p
    def events(p):return [json.loads(l) for l in files['journal/'+p].splitlines() if l.startswith(b'{')]
    def shadows(p,names):
        es=events(p);assert len(es)==2*len(names) and all(e['failures']==0 for e in es)
        for kind in ('shadow-start','shadow-complete'):assert Counter(e['name'] for e in es if e['event']==kind)==Counter(names)
        assert b'AddressSanitizer:' not in files['journal/'+p] and b'runtime error:' not in files['journal/'+p]
    q=data('final-source-qualified.json');assert len(q['commands'])==52
    fail={'normal-allocation-run':94,'default-scalar':2,'default-record':1,'default-api-replace':1,'default-source':1}
    for r in q['commands']:assert r['exit']==fail.get(r['label'],0),r['label']
    for p,h in q['files'].items():assert sha(files['source/'+p])==h,p
    assert {r['label'] for r in q['commands'] if not r['detect_leaks']}=={'san-eval-run','san-checker-run'}
    active=['root_item','root_sum','root_siblings','root_loop','root_for','root_list','root_box','root_choice','root_tuple','main']
    scoped=['string_scope_return','string_scope_bindings','string_scope_owner_rebind','string_scope_helpers','string_scope_loop','string_scope_change','string_scope_sibling','string_scope_pair','string_scope_argument','main']
    loops=['checker_loop_inferred','checker_loop_borrowed','checker_loop_control','main']
    record=['lifetime_target','record_factory','record_case','union_factory','union_case','main']
    scalar=['tuple_case','try_case','try_error_case','unwrap_case','unwrap_error_case','fallback_case','main']
    for profile in ('normal','san'):
        prefix='final-source-'+profile+'-'
        shadows(prefix+'fresh-active.stderr',active+scoped+loops*8)
        contains(prefix+'fresh-active.stdout',b'after64=7 after128=7 peak=7 destroyed=3755.')
        for row in (b'after batch1: bytes=93 objects=2.',b'after batch2: bytes=93 objects=2.',b'registered=20950 released=20949 exported=1 live=0.',b'mode=checker only.',b'mode=private runtime with all32 shadows.',b'two scratch allocation failures and both successful recovery calls.'):
            contains(prefix+'fresh-active.stdout',row)
        contains(prefix+'fresh-active.stderr',b'I require an exclusive borrowed owner for field mutation')
        for name,names in [('record',record),('record-growth',record),('scalar',scalar),('witness',['observe_list','main'])]:
            tail='source' if name=='witness' else 'run';shadows(prefix+name+'-'+tail+'.stderr',names)
        contains(prefix+'record-growth-run.stdout',b'after64=1 after128=1 destroyed=401.')
        contains(prefix+'record-growth-run.stdout',b'registered=403 destroyed=402 transferred=1 live=0.')
        contains(prefix+'record-run.stdout',b'registered=19 destroyed=18 transferred=1 live=0.')
        contains(prefix+'scalar-run.stdout',b'registered=18 destroyed=16 transferred=2 live=0.')
        contains(prefix+'witness-source.stdout',b'registered=12 destroyed=12 transferred=0 live=0.')
        contains(prefix+'scalar-run.stderr',b'I currently borrow only fixed resource records with numeric or boolean fields')
        for name,count in [('eval',134),('checker',88)]:
            raw=files['journal/'+prefix+name+'-run.stdout'];assert raw.count(b'  Testing ')==count
            assert (b'All eval tests passed!' if name=='eval' else b'All typechecker tests passed!') in raw
        assert files['journal/'+prefix+'metadata-run.stderr']==b''
        for mode in ('control','replace'):
            assert files['journal/'+prefix+'host-'+mode+'.stderr']==b''
            contains(prefix+'host-'+mode+'.stdout',b'pointer identity before replacement.')
            contains(prefix+'host-'+mode+'.stdout',b'finish explicit caller teardown.')
        contains(prefix+'test-eval-record-binding-ownership.stdout',b'991')
        contains(prefix+'test-eval-carrier-heap.stdout',b'67667')
        contains(prefix+'test-eval-carrier-values.stdout',b'18 injected graph-walk failures')
        shadows(profile+'-duplicate-run.stderr',['duplicate_loop_return','main'])
        contains(profile+'-duplicate-run.stdout',b'11/10/1/0.')
        raw=files['journal/final-'+profile+'-private-suite-run.stdout'];assert raw.count(b'  Testing ')==134 and b'All eval tests passed!' in raw
        assert b'runtime error:' not in files['journal/final-'+profile+'-private-suite-run.stderr']
    assert all(r['exit']==0 for r in data('final-private-suite.json')['commands'])
    assert all(r['exit']==0 for r in data('duplicate-qualified.json')['commands'])
    contains('final-source-san-allocation-run.stdout',b'positive=2097152 before=202094 after64=203990 after128=203990 delta=0 calls=128.')
    contains('final-source-normal-allocation-run.stdout',b'positive=2101904 before=160432 after64=163856 after128=165632 delta=1776 calls=128.')
    contains('complete-normal-allocation-run.stdout',b'delta=1584 calls=128.')
    contains('final-source-default-scalar.stderr',b'1857 byte(s) leaked in 78 allocation(s)')
    contains('final-source-default-record.stderr',b'3600 byte(s) leaked in 153 allocation(s)')
    for name in ('api-replace','source'):contains('final-source-default-'+name+'.stderr',b'ERROR: AddressSanitizer: heap-use-after-free')
    sup=data('supplement-qualified.json')['commands'];assert len(sup)==7
    sf={'sibling-final-default':1,'loop-final-default':1}
    for r in sup:assert r['exit']==sf.get(r['label'],0),r['label']
    for p in ('sibling-default.stderr','sibling-private.stderr','sibling-final-default.stderr'):contains(p,b'ERROR: AddressSanitizer: heap-use-after-free')
    shadows('sibling-final-private.stderr',['change_text','keep_left','main'])
    shadows('loop-final-default-semantics.stderr',loops*8)
    contains('loop-final-default.stderr',b'5080 byte(s) leaked in 176 allocation(s)')
    contains('strings-second-library.stderr',b'13 byte(s) leaked in 2 allocation(s)')
    contains('strings-second-full.stderr',b'I require an exclusive borrowed owner for field mutation')
    contains('second-strings-run.stderr',b'352 byte(s) leaked in 2 allocation(s)')
    contains('private-suite-diagnostic.stderr',b'test_eval_foreign_native_call_api')
    contains('private-suite-diagnostic.stderr',b'load of null pointer')
    contains('repair-san-private-suite-run.stdout',b'FAILED: env_static_array_reclaim_count() - before >= 5')
    contains('pre-nested-qualified.json',b'"exit": 2')
    contains('final-normal-fresh-active.stderr',b'for-in requires a list, array, or range expression')
    contains('fault-retained-replay.stderr',b'I fail STRING buffer check value.type==VAL_VOID')
    gap=data('curation-gap.json');assert gap['original_recorded_exit']==2 and gap['replay_exit']==1
    assert 'not the original output' in gap['limitation']
    assert sha(files['journal/'+gap['source_snapshot']])==gap['source_sha256']
    contains('accepted-normal-fresh-active.stdout',b'got 2, expected 1')
    for row in (b'value=before',b'value=after'):contains('global-roots-run.stdout',row)
    assert data('leaf-neighbors.json')['exit']==0
    assert files['journal/leaf-neighbors.stdout'].count(b'All tests passed!')>=1
    for token in (b'ERROR: AddressSanitizer',b'runtime error:'):assert token not in files['journal/leaf-neighbors.stderr']
    # I retain full original fixtures and assertion bodies in the source observers.
    for name,path in [('record','tests/test_eval_callable_record_escape.c'),('scalar','tests/test_eval_scalar_projection_sources.c')]:
        original=files['source/'+path].replace(b'int main(',b'int fixture_main(',1)
        original=original.replace(b'#include "test_eval.c"',b'#include "'+str(ROOT/'tests/test_eval.c').encode()+b'"')
        assert original in files['journal/'+name+'-driver.c']
    native=files['source/tests/test_eval_carrier_active_roots.c'].replace(b'int main(void)',b'int fixture_main(void)')
    native=native.replace(b'#include "../src/',b'#include "'+str(ROOT).encode()+b'/src/').replace(b'#include "test_eval.c"',b'#include "'+str(ROOT/'tests/test_eval.c').encode()+b'"')
    assert native in files['journal/native-allocation-driver.c']
    assert b'last_bytes>middle_bytes+1024)return 94' in files['journal/native-allocation-driver.c']
    assert b'after>before+4096)return 93' in files['journal/native-allocation-driver.c']
    assert b'#include "'+str(ROOT/'tests/test_eval.c').encode()+b'"' in files['journal/private-suite.c']
    base=json.loads((HERE/manifest['base_archives'][0]['path']).read_text());prior={r['path']:r for r in base['git_sources']}
    changed={'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md','src/env.c','src/eval.c','src/eval/carrier_runtime.c','src/eval/carrier_runtime.h','src/eval/carrier_values.c','src/eval/carrier_values.h','src/eval/eval_string.c','src/nanolang.h','src/typechecker.c','tests/test_eval_carrier_string_scopes.c','tests/test_checker_loop_metadata.c','tests/test_eval_string_buffer_failures.c','tests/fixtures/carrier_string_scopes.nano','tests/fixtures/carrier_string_field_invalid.nano','tests/fixtures/checker_loop_metadata.nano'}
    for p,r in prior.items():
        if p not in changed:assert sha(files['source/'+p])==r['sha256'],p
    h=files['source/src/nanolang.h'];decl=b'TypeInfo *env_own_checker_type_info(Environment *env, TypeInfo *info);\n'
    assert h.count(decl)==1 and sha(h.replace(decl,b''))==prior['src/nanolang.h']['sha256']
    checker=files['source/src/typechecker.c'];new=b'                iter_info = env_own_checker_type_info(tc->env,\n                    copy_generic_argument_info(range_expr, tc->env, 0));'
    assert checker.count(new)==1
    assert sha(checker.replace(new,b'                iter_info = copy_generic_argument_info(range_expr, tc->env, 0);'))==prior['src/typechecker.c']['sha256']
    make=files['source/Makefile.gnu'];a=make.index(b'.PHONY: test-eval-string-buffer-failures\n');b=make.index(b'.PHONY: test-eval-carrier-active-roots\n',a)
    assert sha(make[:a]+make[b:])==prior['Makefile.gnu']['sha256']
    lib=files['source/src/eval/eval_string.c'];new=b'    Value value = create_string(result);\n    free(result);\n    return value;'
    assert lib.count(new)==2;lib=lib.replace(new,b'    return create_string(result);').replace(b'    if (!result) return create_void();\n',b'')
    assert sha(lib)==prior['src/eval/eval_string.c']['sha256']
    for token in (b'env_carrier_enable(',b'env_carrier_enable_collection('):
        enablers={p for p,raw in files.items() if p.startswith('source/src/') and token in raw}
        assert enablers=={'source/src/eval/carrier_runtime.c','source/src/eval/carrier_runtime.h'}
    contains('duplicate-driver.c',b'ASSERT_EQ(gc_get_stats().num_objects,0)')
    unit=files['source/tests/test_eval_carrier_string_scopes.c']
    assert b'ASSERT_EQ(stats.current_usage,baseline.current_usage)' in unit
    assert b'ASSERT_EQ(gc_get_stats().num_objects,0)' in unit and b'ASSERT_EQ(gc_get_stats().current_usage,0)' in unit
    values=files['source/src/eval/carrier_values.c'];assert b'free(a->data); free(a); reclaimed_arrays++; break;' in values
    assert b's_static_array_reclaims + nl_carrier_values_reclaimed_arrays()' in files['source/src/env.c']
    actual=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines();assert set(actual)==changed
    tools=data('current-tools.json');assert tools['host'][4]=='aarch64'
    assert '13.3.0' in tools['cc']['stdout'] and '2.42' in tools['as']['stdout']
    assert tools['cc']['exit']==tools['as']['exit']==0
    for p,r in data('image-guards.json').items():
        key='images/bin/nanoc_stage2' if p=='bin/nanoc' else 'images/'+p
        assert (len(files[key]),sha(files[key]))==(r['bytes'],r['sha256'])
    closure=data('provider-closure.json');assert closure['exit']==0 and len(closure['identities'])==9
    assert closure['observation']=='after terminal qualification; no before-after closure claim'
    ids=data('retained-inputs.json');assert len(ids['products'])==146 and len(ids['objects'])==1207
    assert len(manifest['product_paths'])==38
    for p in manifest['product_paths']:
        raw=files['products/'+str(Path(p).relative_to('/tmp'))]
        assert raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little')==183
        assert (len(raw),sha(raw))==(ids['products'][p]['bytes'],ids['products'][p]['sha256'])
    assert sum(p.endswith('.so') for p in manifest['product_paths'])==2
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; private STRING scopes qualified; ordinary byte/sibling/default ownership failures retained')

if __name__=='__main__':main()
