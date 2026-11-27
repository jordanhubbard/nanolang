"""I verify retained native failures, diagnostics and preflight without accepting a release."""
import argparse,gzip,hashlib,json,re,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);p.add_argument('--repo',required=True,type=Path);p.add_argument('--workspace',required=True,type=Path);a=p.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='094fc7216b16acda845459c01c9fb374dfa48cd8e8cbbf4f0dd995b1b3f9c371';manifest=json.loads(raw)
assert manifest['whole_release'] is False

def git_blobs(revision,names):
 names=list(names);request=''.join(revision+':'+name+'\n' for name in names).encode()
 data=subprocess.check_output(['git','-C',str(a.repo),'cat-file','--batch'],input=request)
 out={};offset=0
 for name in names:
  end=data.index(b'\n',offset);header=data[offset:end].split();assert len(header)==3,(name,header)
  size=int(header[-1]);out[name]=data[end+1:end+1+size];offset=end+size+2
 assert offset==len(data)
 return out

with tempfile.TemporaryDirectory(prefix='native-followups-reader-',dir=a.workspace) as directory:
 root=Path(directory);seen=set();used=set();links=[]
 for r in manifest['files']:
  rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
  assert r['path'] not in seen;seen.add(r['path'])
  if r['kind']=='symlink':links.append(r);continue
  assert r['kind']=='file';f=root/rel;f.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with f.open('wb') as stream:
   for key in r['chunks']:
    packed=(a.archive/'objects'/(key+'.gz')).read_bytes();meta=manifest['chunks'][key]
    assert sha(packed)==meta['gzip_sha256'];block=gzip.decompress(packed)
    assert len(block)==meta['bytes'] and sha(block)==key
    stream.write(block);h.update(block);size+=len(block);used.add(key)
  assert size==r['size'] and h.hexdigest()==r['sha256'],r['path']
 assert used==set(manifest['chunks']) and len(seen)==2278 and len(links)==7
 load=lambda f:json.loads(f.read_text())
 orig=root/'madmax-native-original';q=load(orig/'result.json');base='/home/jkh/.worktrees/nanolang/c-repairs-gcc15-20261126/'
 assert q['head']=='d90b3feb9aab481387e9317b0a620ee071a1314b'
 assert q['terminal'] and q['guards_match'] and not q['qualification_complete']
 assert q['sources_before']==q['sources_after'] and q['tools_before']==q['tools_after']
 names=[n[len(base):] for n in q['sources_after']];assert all(n.startswith(base) for n in q['sources_after'])
 blobs=git_blobs(q['head'],names)
 overlay_name='tests/test_module_cache_publication.py';overlay=(orig/'retained/source-overlays'/overlay_name).read_bytes()
 old=blobs[overlay_name];needle=b"c_source.write_text('''#include <stdint.h>\n#include <stdlib.h>"
 assert old.count(needle)==1 and old.replace(needle,b"c_source.write_text('''#define _DEFAULT_SOURCE 1\n#include <stdint.h>\n#include <stdlib.h>",1)==overlay
 for name,value in q['sources_after'].items():
  rel=name[len(base):];assert sha(overlay if rel==overlay_name else blobs[rel])==value,rel
 tools=load(orig/'retained/tools.json');assert set(tools)==set(q['tools_after'])
 for name,r in tools.items():assert sha((orig/r['path']).read_bytes())==q['tools_after'][name]==r['sha256'],name
 def retained(name):return orig/'retained'/Path(name).relative_to(base)
 b=load(orig/'retained/bin/nanoc_bootstrap.json')
 assert b['phase']=='stage2' and len(b['steps'])==17 and all(s['exit_code']==0 for s in b['steps'])
 assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30 and b['raw_stage1_stage2_equal']
 for r in b['steps']:assert sha(retained(r['log']).read_bytes())==r['log_sha256'],r['label']
 for name,value in b['artifacts'].items():assert sha((orig/'retained/bin'/name).read_bytes())==value,name
 for name,value in b['tools'].items():assert sha((orig/'retained/bin'/name).read_bytes())==value,name
 for name,value in b['hosts'].items():assert sha(retained(name).read_bytes())==value,name
 index=orig/'retained/modules/index.json';assert sha(index.read_bytes())==b['sources']['modules/index.json']
 for name,value in b['sources'].items():assert sha(index.read_bytes() if name=='modules/index.json' else blobs[name])==value,name
 assert (orig/'retained/bin/nanoc_stage1.nvm').read_bytes()==(orig/'retained/bin/nanoc_stage2.nvm').read_bytes()
 installed=next(r for r in links if r['path']=='madmax-native-original/retained/bin/nanoc')
 assert installed['target']=='nanoc_stage2' and sha((orig/'retained/bin/nanoc_stage2').read_bytes())==b['artifacts']['nanoc_stage2']
 assert 'bin/nanoc succeeded with bin/nanoc_c removed' in (orig/'bootstrap.stdout').read_text()
 assert [s['exit_code'] for s in q['steps']]==[0,0,2]
 assert re.search(r'Ran 36 tests in [^\n]+\n\nOK', (orig/'native.stderr').read_text())
 error=(orig/'emitter.stderr').read_text();assert 'Ran 0 tests' in error and 'timed out after 300 seconds' in error and 'nanoc_stage2' in error
 perf=root/'madmax-emitter-profile';pq=load(perf/'result.json')
 assert pq['terminal'] and pq['guards_match'] and pq['command_timeout'] and pq['exit_code']==-9 and pq['report_exit_code']==0
 assert pq['guards_before']==pq['guards_after']
 expected={**q['sources_after'],**q['tools_after'],base+'modules/index.json':sha(index.read_bytes())}
 assert pq['guards_before']==expected
 assert sha((perf/'perf.data').read_bytes())==pq['perf_sha256']
 assert '(14002 samples)' in (perf/'perf.stderr').read_text()
 report=(perf/'perf-report.txt').read_text();assert 'Total Lost Samples: 0' in report and '__memset_avx2_unaligned_erms' in report
 dsos=load(perf/'dso-receipt.json');assert len(dsos)==8
 for r in dsos:assert r['recorded_buildid']==r['current_buildid'] and sha((perf/r['stored']).read_bytes())==r['sha256']
 record=root/'record-pointer-overlay-original';oq=load(record/'overlay.json')
 original_c=retained(oq['original_source']).read_bytes();changed_c=(record/'stage2-pointer-union.c').read_bytes()
 old_decl=(record/'original-declaration.txt').read_bytes();new_decl=(record/'overlay-declaration.txt').read_bytes()
 assert original_c.count(old_decl)==1 and original_c.replace(old_decl,new_decl,1)==changed_c
 assert sha(original_c)==oq['original_sha256'] and sha(changed_c)==oq['overlay_sha256'] and oq['build_exit_code']==0 and oq['record_width']==80
 test=record/'original-emitter-tests';tq=load(test/'result.json')
 assert tq['terminal'] and tq['passed'] and tq['tests']==12 and tq['failures']==tq['errors']==tq['skips']==0 and tq['guards_match']
 assert tq['before']==tq['after']
 for name,value in tq['before'].items():
  rel=Path(name).relative_to(base)
  f=record/rel.relative_to('record-pointer-overlay-20261127') if str(rel).startswith('record-pointer-overlay-20261127/') else orig/'retained'/rel
  data=f.read_bytes() if f.is_file() else blobs[str(rel)]
  assert sha(data)==value,name
 commands=[json.loads(l) for l in (test/'commands.jsonl').read_text().splitlines()];assert len(commands)==132
 for i,r in enumerate(commands):
  assert r['sequence']==i and 'exception' not in r
  expected=list(r['requested'])
  if expected[0]==base+'bin/nanoc_stage2':expected[0]=base+'record-pointer-overlay-20261127/nanoc-pointer-union'
  assert r['actual']==expected
  assert (test/(str(i)+'.stdout')).is_file() and (test/(str(i)+'.stderr')).is_file()
 first=load(record/'raw-module-parity/result.json');fixed=load(record/'raw-module-parity-cache/result.json')
 assert first['terminal'] and first['exit_code']==0 and first['guards_match'] and not first['byte_identical']
 assert fixed['terminal'] and fixed['exit_code']==0 and fixed['guards_match'] and fixed['byte_identical']
 ref=orig/'emitter-backend-preterminal/nano-native-shadows-f0rgsqlg/nano_bytecode_product_fSE9Is/program.nvm'
 target=record/'raw-module-parity-cache/emitter.nvm';assert ref.read_bytes()==target.read_bytes() and target.stat().st_size==628200
 assert sha(target.read_bytes())=='4db7d935ac0dd8f4a248f80e2c1515ac4701d12af602ad4ca0facc5ef321605a'
 assert all(r['exit_code']==0 for r in first['verification']+fixed['verification'])
 left=(record/'raw-module-parity/reference.nasm').read_bytes();right=(record/'raw-module-parity/overlay.nasm').read_bytes()
 libs=lambda raw:re.findall(rb'/[ -~]*?\.so',raw)
 oldlibs=list(dict.fromkeys(libs(ref.read_bytes())));newlibs=list(dict.fromkeys(libs((record/'raw-module-parity/emitter.nvm').read_bytes())));assert len(oldlibs)==len(newlibs)==3
 for old,new in zip(oldlibs,newlibs):assert left.count(old)==2;left=left.replace(old,new)
 assert left==right
 closure=load(record/'library-closures/receipt.json');assert len(closure)==12
 for r in closure.values():assert sha((record/'library-closures'/r['path']).read_bytes())==r['sha256']
 pre=root/'native-gate-preflight-20261127';preq=load(pre/'receipt.json');assert preq['structured_C_passed']==2456 and preq['structured_C_failed']==2 and not preq['full_target_passed']
 for name,value in preq['files'].items():
  f=pre/name if name.startswith('bin/') else pre/'source'/name;assert sha(f.read_bytes())==value,name
 code_names=[n for n in preq['files'] if not n.startswith('bin/')];current=git_blobs(manifest['code_revision'],code_names)
 for name,data in current.items():assert (pre/'source'/name).read_bytes()==data,name
 storage=pre/'structured-storage-overlay';sr=load(storage/'receipt.json');assert sr['substitutions']==14 and sr['reversal_exact'] and sr['assertions_and_deadlines_unchanged']
 original_test=(storage/'test_nvm2c.c').read_bytes().replace(sr['alias'].encode(),b'/tmp');assert sha(original_test)==sr['source_before']
 assert original_test==git_blobs(manifest['code_revision'],['tests/nanoisa/test_nvm2c.c'])['tests/nanoisa/test_nvm2c.c']
 log=(storage/'full-gate.stdout').read_text();assert '=== 2456 passed, 2 failed ===' in log
 failures=re.findall(r'^  FAIL: (.*)$',log,re.M);assert len(failures)==2 and all('CAST_STRING' in f for f in failures)
 assert 'Shape constraints: 3890 passed, 0 failed' in log
 assert 'Ran 1 test' in (pre/'native-math-flags/pointer-storage.stderr').read_text() and '\nOK\n' in (pre/'native-math-flags/pointer-storage.stderr').read_text()
 trace=(pre/'native-math-flags/trace.stderr').read_text();assert 'aot_math_flags' in trace and 'native_shadow_command' in trace
 comp=load(pre/'record-pointer-generator/comparison.json');body=load(pre/'record-pointer-generator/body-comparison.json')
 assert not comp['reversal_exact'] and body['legacy_record_reversal_and_nonmarker_body_equal'] and body['old_host_marker_lines']==body['new_host_marker_lines']==24
 math=load(root/'darwin-math-link-control/receipt.json');assert math['exit_code']==math['execution_exit_code']==0 and math['stderr']=='' and '-lm' not in math['argv'] and '-fno-builtin' in math['argv']
 assert (root/'darwin-math-link-control/math.c').read_text()==math['source']
 print(json.dumps({'archive_verified':True,'records':len(seen),'chunks':len(used),'source_files_verified':len(names),'code_files_verified':len(code_names),'original_bootstrap_steps_passed':17,'original_native_methods_passed':36,'original_emitter_methods_discovered':0,'original_emitter_gate_passed':False,'prototype_emitter_methods_passed':12,'prototype_raw_module_parity':True,'structured_C_passed':2456,'structured_C_failed':2,'full_current_native_candidate_included':False,'whole_release':False,'current_liveness_checked':False}))
