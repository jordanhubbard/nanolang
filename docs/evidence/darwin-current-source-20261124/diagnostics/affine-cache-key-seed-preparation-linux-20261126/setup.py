from pathlib import Path
import tempfile,subprocess,hashlib,json,tarfile,io,shutil,time,os
repo=Path('/var/tmp/nanolang-darwin-publication-qkk55gi3')
j=Path(__file__).resolve().parent;print(j,flush=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base='67d99e5121521428740d385d10071b9679e66013'
ready=Path('/var/tmp/nanolang-path-copy-readiness-20261125-e0jzzo8k');old=Path('/var/tmp/nanolang-affine-global-array-report-five-20261125-vgeron2x');previous=json.loads((old/'result.json').read_text())
s=dict(terminal=False,source_base=base,working_overlay=True,whole_release=False,ordinary_seconds=10,new_bootstrap=False,stage_alias=False,commands=[],readiness_root=str(ready),readiness_sha256=sha(ready/'result.json'),genuine_readiness_sha256=sha(ready/'genuine-result.json'),readiness_publication=base,readiness_manifest_sha256='059daeb734e96daac1eaec34d926b4449a0b74a36f3df57d233c03fcd43d2ac9')
def save():(j/'result.json').write_text(json.dumps(s,indent=2)+'\n')
def run(label,args,cwd=repo,limit=180):
 row=dict(label=label,argv=list(map(str,args)),cwd=str(cwd),timeout=limit);s['commands'].append(row);save();start=time.monotonic()
 with (j/(label+'.stdout')).open('wb') as out,(j/(label+'.stderr')).open('wb') as err:
  r=subprocess.run(row['argv'],cwd=cwd,stdout=out,stderr=err,timeout=limit)
 row.update(exit=r.returncode,seconds=time.monotonic()-start);save();assert r.returncode==0
try:
 prior=Path('/var/tmp/nanolang-report1261-three-five-20261125-ul75rs9t/cohort-result.json')
 assert prior.exists(), 'I require terminal corrected three-method evidence before preparing my cache run'
 triple=json.loads(prior.read_bytes());assert triple['terminal'] and triple['accepted_component'] and triple['all_guards_match'], 'I require accepted terminal corrected three-method evidence before preparing my cache run'
 assert triple['tests']==dict(tests=3,failures=0,errors=0,skips=0,passed=True)
 publication=os.environ['NANOLANG_REPORT1261_THREE_EVIDENCE_COMMIT'];assert len(publication)==40
 proof=repo/'docs/evidence/darwin-current-source-20261124/diagnostics/affine-original-frame-three-five-linux-20261125'
 for name in ['manifest.json','read_archive.py']:
  assert (proof/name).read_bytes()==subprocess.check_output(['git','show',publication+':'+str((proof/name).relative_to(repo))],cwd=repo)
 assert subprocess.run(['git','merge-base','--is-ancestor',publication,'origin/feat/5.1-service-module-integration'],cwd=repo).returncode==0
 run('corrected-three-reader',['python3',proof/'read_archive.py'])
 s['corrected_three_evidence']=dict(commit=publication,result_sha256=sha(prior),manifest_sha256=sha(proof/'manifest.json'))
 global_root=Path('/var/tmp/nanolang-path-copy-global-array-five-20261125-jsj5x9iu')
 parent=json.loads((global_root/'cohort-result.json').read_text())
 assert parent.get('terminal') and parent.get('accepted_component') and parent.get('all_guards_match'), 'I require complete accepted global/array qualification before source preparation'
 assert parent['tests']==dict(tests=2,failures=0,errors=0,skips=0,passed=True)
 assert parent['source_pin']==base and parent['producer_sha256']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
 publication=os.environ['NANOLANG_PATH_COPY_GLOBAL_ARRAY_EVIDENCE_COMMIT']
 assert len(publication)==40 and all(c in '0123456789abcdef' for c in publication)
 proof=repo/'docs/evidence/darwin-current-source-20261124/diagnostics/ownership-path-copy-global-array-five-linux-20261125'
 for name in ['manifest.json','read_archive.py']:
  assert (proof/name).read_bytes()==subprocess.check_output(['git','show',publication+':'+str((proof/name).relative_to(repo))],cwd=repo)
 manifest=json.loads((proof/'manifest.json').read_text());assert manifest['accepted_component'] and manifest['full_five_routes'] and manifest['source_commit']==base
 assert subprocess.run(['git','merge-base','--is-ancestor',publication,'origin/feat/5.1-service-module-integration'],cwd=repo).returncode==0
 s['global_array_evidence']=dict(commit=publication,result_sha256=sha(global_root/'cohort-result.json'),manifest_sha256=sha(proof/'manifest.json'),producer_sha256=parent['producer_sha256'])
 run('complete-global-array-reader',['python3',proof/'read_archive.py'])
 foundation=Path('/var/tmp/nanolang-affine-typed-five-20261125-udz2o6c_');f=json.loads((foundation/'cohort-result.json').read_text())
 assert f['terminal'] and f['accepted_component'] and f['all_guards_match'] and f['tests']==dict(tests=5,failures=0,errors=0,skips=0,passed=True)
 fp=repo/'docs/evidence/darwin-current-source-20261124/diagnostics/affine-typed-five-linux-20261125'
 foundation_commit='b8cf8018ecf194c03434df4390ac0bb589f278b1'
 for name in ['manifest.json','read_archive.py']:
  assert (fp/name).read_bytes()==subprocess.check_output(['git','show',foundation_commit+':'+str((fp/name).relative_to(repo))],cwd=repo)
 assert subprocess.run(['git','merge-base','--is-ancestor',foundation_commit,'origin/feat/5.1-service-module-integration'],cwd=repo).returncode==0
 s['foundation_evidence']=dict(commit=foundation_commit,result_sha256=sha(foundation/'cohort-result.json'),manifest_sha256=sha(fp/'manifest.json'),producer_sha256=f['producer_sha256'])
 assert f['producer_sha256']==parent['producer_sha256']
 run('complete-foundation-reader',['python3',fp/'read_archive.py'])
 for n in ['CMD_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS','NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','NANOLANG_TRANSLATOR_SOURCE_COMPILER','NANOLANG_TRANSLATOR_SOURCE_MODULE']:assert n not in os.environ,n
 s['environment_before']={n:os.environ.get(n) for n in ['CC','PATH','CFLAGS','LDFLAGS','CMD_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS','NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','NANOLANG_TRANSLATOR_SOURCE_COMPILER','NANOLANG_TRANSLATOR_SOURCE_MODULE']}
 for name in ['result.json','genuine-result.json']:
  q=json.loads((ready/name).read_text());assert q['terminal'] and q['all_guards_match']
 reader=repo/'docs/evidence/darwin-current-source-20261124/diagnostics/ownership-path-copy-readiness-linux-20261125/read_archive.py'
 for n in ['manifest.json','read_archive.py']:
  p=reader.parent/n;assert p.read_bytes()==subprocess.check_output(['git','show',base+':'+str(p.relative_to(repo))],cwd=repo)
 assert sha(reader.parent/'manifest.json')==s['readiness_manifest_sha256']
 assert subprocess.run(['git','merge-base','--is-ancestor',base,'origin/feat/5.1-service-module-integration'],cwd=repo).returncode==0
 run('readiness-reader',['python3',reader])
 query=Path('/var/tmp/nanolang-path-copy-query-five-20261125-nb81f3jc')
 gate=json.loads((query/'cohort-result.json').read_text())
 assert gate['terminal'] and gate['accepted_component'] and gate['all_guards_match'] and gate['tests']==dict(tests=1,failures=0,errors=0,skips=0,passed=True)
 assert len(gate['commands'])==1360 and gate['original_query_cases']==3080 and gate['source_pin']==base
 publication='dec468a6bda61adf5d268c365832811c448d69f0'
 proof=repo/'docs/evidence/darwin-current-source-20261124/diagnostics/ownership-path-copy-query-five-linux-20261125'
 for name in ['manifest.json','read_archive.py']:
  assert (proof/name).read_bytes()==subprocess.check_output(['git','show',publication+':'+str((proof/name).relative_to(repo))],cwd=repo)
 assert sha(proof/'manifest.json')=='8de5afcc2d1cd1f3428618fa22ad5d1d3da101ebcbfcb01b3057020fbe02c709'
 assert subprocess.run(['git','merge-base','--is-ancestor',publication,'origin/feat/5.1-service-module-integration'],cwd=repo).returncode==0
 s['query_evidence']=dict(commit=publication,manifest_sha256=sha(proof/'manifest.json'),result_sha256=sha(query/'cohort-result.json'),source_pin=base,overlay_sha256=gate['overlay_sha256'])
 run('complete-query-reader',['python3',proof/'read_archive.py'])
 keyproof=repo/'docs/evidence/darwin-current-source-20261124/diagnostics/affine-cache-key-seed-controls-linux-20261125'
 key_publication='beecf685b2ca9bc74791be2d8abf246689e01968'
 for name in ['manifest.json','read_controls.py']:
  assert (keyproof/name).read_bytes()==subprocess.check_output(['git','show',key_publication+':'+str((keyproof/name).relative_to(repo))],cwd=repo)
 assert subprocess.run(['git','merge-base','--is-ancestor',key_publication,'origin/feat/5.1-service-module-integration'],cwd=repo).returncode==0
 run('cache-key-seed-reader',['python3',keyproof/'read_controls.py'])
 s['cache_key_seed_manifest_sha256']=sha(keyproof/'manifest.json')
 root=j/'source';root.mkdir()
 names=['src','src_nano','tests','modules','Makefile','Makefile.gnu','scripts']
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',base,*names],cwd=repo))) as t:t.extractall(root,filter='data')
 p='src_nano/nvm2c/ownership_queries.nano';original=(root/p).read_bytes();candidate=(ready/'source'/p).read_bytes();assert original!=candidate
 (root/p).write_bytes(candidate);s['overlays']={p:sha(root/p)};shutil.copy2(ready/'candidate.diff',j/'candidate.diff')
 for n,h in json.loads((ready/'result.json').read_text())['source_before'].items():assert sha(root/n)==h,n
 s['report_fix_commit']='c886e6dc73c2680d17a6db76b18e4ec00050e592'
 s['failed_parent_sha256']=sha(Path('/var/tmp/nanolang-path-copy-six-leaves-five-20261125-uz4l7yp9/cohort-result.json'))
 proof=repo/'docs/evidence/darwin-current-source-20261124/diagnostics/affine-original-frame-report-fix-linux-20261125'
 run('report-fix-reader',['python3',proof/'read_reports.py'])
 s['report_fix_manifest_sha256']=sha(proof/'manifest.json')
 for leaf in ['aggregates','effects','value_calls','analysis_cache']:
  name='tests/nvm2c_affine_'+leaf+'_driver.nano'
  (root/name).write_bytes(subprocess.check_output(['git','show',('901caef8e6f995080ac7d3c48c4cf23c75c99315' if leaf=='analysis_cache' else s['report_fix_commit'])+':'+name],cwd=repo))
  s['overlays'][name]=sha(root/name)
 s['source_before']={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
 assert {n:h for n,h in s['source_before'].items() if n!='tests/nvm2c_affine_analysis_cache_driver.nano'}=={n:h for n,h in triple['source_before'].items() if n!='tests/nvm2c_affine_analysis_cache_driver.nano'}
 s['cache_fix_commit']='901caef8e6f995080ac7d3c48c4cf23c75c99315'
 assert {n:h for n,h in s['source_before'].items() if n not in s['overlays']}=={n:h for n,h in gate['source_before'].items() if n not in s['overlays']}
 assert gate['source_before']==parent['source_before']
 s['observer_source_sha256']=sha(query/'cohort_journal.py')
 s['images_before']={}
 for n,h in previous['images_after'].items():
  p=root/n;p.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(old/'source'/n,p);assert sha(p)==h;s['images_before'][n]=h
 s['producer']=str(root/'bin/nanoc_emitter_phase');shutil.copy2(previous['producer'],s['producer']);s['producer_sha256']=sha(s['producer']);assert s['producer_sha256']==previous['producer_sha256']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
 s['bootstrap_source_pin']=previous['bootstrap_source_pin'];s['hosts_before']=previous['hosts_after'];s['tools_before']=previous['tools_after']
 for group in ['hosts_before','tools_before']:assert all(sha(n)==h for n,h in s[group].items())
 s['cpu_rounds']=[]
 for _ in range(2):
  wall=time.monotonic();cpu=time.process_time();v=1
  while time.process_time()-cpu<1:
   for k in range(1024):v=(v*1664525+1013904223)&0xffffffff
  c=time.process_time()-cpu;w=time.monotonic()-wall;s['cpu_rounds'].append(dict(cpu=c,wall=w,share=c/w));assert c/w>=.75
 for n in ['products','tmp']:(j/n).mkdir()
 s['source_after']={n:sha(root/n) for n in s['source_before']};s['images_after']={n:sha(root/n) for n in s['images_before']};s['hosts_after']={n:sha(n) for n in s['hosts_before']};s['tools_after']={n:sha(n) for n in s['tools_before']}
 s['all_guards_match']=all(s[n+'_before']==s[n+'_after'] for n in ['source','images','hosts','tools']);assert s['all_guards_match'];s['admitted_setup']=True
finally:s['terminal']=True;save()
