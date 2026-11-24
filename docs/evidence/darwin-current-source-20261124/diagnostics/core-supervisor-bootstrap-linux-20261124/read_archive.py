"""I verify an original bootstrap PASS and preserve the failed original native consumer gate."""
from pathlib import Path
import sys,json,hashlib,subprocess,re,ast
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'dc9080dac206655d64267f2e3755fcd463376d83ffdc35d359225593c36421db');sha=lambda b:hashlib.sha256(b).hexdigest();b=js('bootstrap/bootstrap-result.json');n=js('bootstrap/native-gates-result.json');s=js('bootstrap/setup-result.json');p=js('bootstrap/prepare-result.json');receipt=js('bootstrap/source/bin/nanoc_bootstrap.json');prefix='bootstrap/source/'
assert b['terminal'] and b['exit']==0 and b['original_make_pass'] and b['source_stable'] and b['images_stable'] and b['tools_stable']
assert m['original_make_pass'] and m['original_make_exit']==0 and not m['original_native_gates'] and m['native_gates_executed'] and not m['emitter_companion_executed']
assert not m['whole_release'] and not m['complete_san_acceptance'] and not m['full_source_retry'] and m['known_open_failure']==1252 and m['known_native_failure']==1253
assert b['source_pin']==m['source_commit']==s['source_pin']==p['source_pin']==n['source_pin']
assert b['source_before']==b['source_after']==n['source_before']==n['source_after']==s['source_before']
for name,h in b['source_before'].items():assert rows[prefix+name]['sha256']==h,name
assert b['images_before']==b['images_after']
for name,h in b['images_before'].items():assert rows[prefix+name]['sha256']==h,name
assert b['tools_before']==b['tools_after']==n['tools_before']==n['tools_after']
for name,h in b['tools_before'].items():assert rows['tool/'+name.lstrip('/')]['sha256']==h
links={v['path']:v['target'] for v in m['symlinks']};assert links[prefix+'bin/nanoc']=='nanoc_stage2'
for name,info in b['products'].items():
 path=prefix+('bin/nanoc_stage2' if name=='bin/nanoc' else name);assert rows[path]['sha256']==info['sha256'] and rows[path]['bytes']==info['bytes']
assert n['images_before']==n['images_after']
for name,h in n['images_before'].items():assert rows[prefix+('bin/nanoc_stage2' if name=='bin/nanoc' else name)]['sha256']==h
assert receipt['phase']=='stage2' and receipt['raw_stage1_stage2_equal'] and receipt['stage_timeout_seconds']==1800 and receipt['shadow_timeout_seconds']==30
assert b['stage_seconds']==s['bootstrap_stage_seconds']==1800 and b['compiler_suite_seconds']==30 and b['ordinary_seconds']==10
assert data(prefix+'bin/nanoc_stage1.nvm')==data(prefix+'bin/nanoc_stage2.nvm')
assert rows[prefix+'bin/nanoc_stage1.nvm']['bytes']==1049600 and rows[prefix+'bin/nanoc_stage1.nvm']['sha256']=='964957dfa356d2b630798c85599e2a83aac514040c6d91434157bd7706563900'
for name,h in receipt['artifacts'].items():assert rows[prefix+'bin/'+name]['sha256']==h
for name,h in receipt['tools'].items():assert rows[prefix+'bin/'+name]['sha256']==h
for name,h in receipt['sources'].items():assert rows[prefix+name]['sha256']==h
root=Path(m['roots']['bootstrap'])
def archive_path(path):return 'bootstrap/'+str(Path(path).relative_to(root))
for path,h in receipt['hosts'].items():assert rows[archive_path(path)]['sha256']==h
expected=['seed','seed-verify','stage1','stage1-verify','stage1-translate','stage1-native','stage1-hello-compile','stage1-hello-execute','stage2','stage2-verify','stage2-translate','stage2-native','stage2-hello-compile','stage2-hello-execute','seed-final-verify','stage1-final-verify','stage2-final-verify']
assert [v['label'] for v in receipt['steps']]==expected
for v in receipt['steps']:
 assert v['exit_code']==0 and not v.get('interrupted_or_timed_out',False) and v['elapsed_seconds']<1800
 assert rows[archive_path(v['log'])]['sha256']==v['log_sha256']
work=archive_path(receipt['work']);assert not any(x.startswith(work+'/') and x.endswith('/unexpected-native-compiler') for x in rows)
assert set(b['sentinels'])=={'.bootstrap0.built','.bootstrap1.built','.bootstrap2.built','.bootstrap3.built'}
for name in b['sentinels']:assert rows[prefix+name]['bytes']==0
out=data('bootstrap/bootstrap.stdout');assert b'I installed bin/nanoc from the verified Stage 2 module.' in out and b'bin/nanoc succeeded with bin/nanoc_c removed' in out
assert n['terminal'] and not n['accepted'] and len(n['commands'])==1 and n['commands'][0]['target']=='test-native-shadows' and n['commands'][0]['exit']==2 and not n['commands'][0].get('outer_timeout',False)
err=data('bootstrap/test-native-shadows.stderr');assert re.search(rb'Ran 18 tests in [^\n]+\n\nFAILED \(failures=5\)',err)
failures=re.findall(rb'^FAIL: (test_\w+)',err,re.M);assert set(failures)=={b'test_deadline_survives_foreign_alarm_cancellation',b'test_early_zero_exit_preserves_output',b'test_local_scopes_do_not_leak',b'test_shadow_only_source',b'test_source_only_requires_no_native_compiler'}
assert err.count(b'I cannot lower this checked program: no main')==2 and b'without executing shadows' in err and b"not found in b''" in err
assert not any(x.endswith('/test-native-shadow-emitter.stdout') for x in rows)
for filename,count in [('test_native_shadows.py',14),('test_shadow_runner.py',1),('test_make_timeouts.py',3),('test_native_shadow_emitter.py',12)]:
 tree=ast.parse(data(prefix+'tests/'+filename));assert sum(isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef)) and v.name.startswith('test_') for v in ast.walk(tree))==count
observer=js('bootstrap/dependent-native-controller.json');assert observer['terminal'] and not observer['accepted'] and observer['bootstrap_exit']==0 and observer['native_targets'][0]['exit']==2
milestone=js('bootstrap/raw-equality-milestone.json');assert milestone['bootstrap_terminal'] and milestone['stage2_native_terminal'] and milestone['original_make_exit']==0
assert not js('bootstrap/observation-provenance/stale-intermediate-flags.json')['bootstrap_terminal']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in b['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert len(head)==3 and head[1]==b'blob',name;raw=proc.stdout.read(int(head[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS archive integrity: %d records, exact-source original17-step bootstrap/equality/install/no-C-seed PASS; original native18 FAIL5 under1253, emitter12 NOT RUN. Complete SAN/paired/pure-translator/release acceptance remains open.'%len(rows))
