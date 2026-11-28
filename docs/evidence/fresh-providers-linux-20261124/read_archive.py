#!/usr/bin/env python3
"""I verify fresh complete compiler providers and retained checker refusals."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess
BASE=Path(__file__).resolve().parent;REPO=BASE.parents[2]
m=json.loads((BASE/'manifest.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();rows={r['path']:r for r in m['files']};assert len(rows)==len(m['files']);verified={};images=set()
for row in rows.values():
 p=(BASE/row['artifact']).resolve();assert p.is_relative_to(REPO/'docs/evidence');identity=(row['compressed_sha256'],row['sha256'],row['bytes'])
 if p in verified:assert verified[p]==identity;continue
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):digest.update(b)
 assert digest.hexdigest()==row['compressed_sha256']
 digest=hashlib.sha256();size=0;head=b''
 with gzip.open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):
   if not head:head=b[:64]
   digest.update(b);size+=len(b)
 assert digest.hexdigest()==row['sha256'] and size==row['bytes']
 if head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[16:18],'little') in (2,3) and int.from_bytes(head[18:20],'little')==183:images.add(row['sha256'])
 verified[p]=identity
assert {p.resolve() for p in (BASE/'objects').glob('*.gz')}=={(BASE/r['artifact']).resolve() for r in rows.values() if r['artifact'].startswith('objects/')}
assert sha((BASE/m['base_manifest']['path']).read_bytes())==m['base_manifest']['sha256']
def data(name):return gzip.decompress((BASE/rows[name]['artifact']).read_bytes())
def js(name):return json.loads(data('journal/'+name+'.json'))
def provider(path):return data('provider/'+path)
old=js('result');new=js('repaired/result');assert old['source_pin']==m['failed_source_pin']=='7c0b9b27621918587eb96320d8241ba73cae8ea4';assert new['source_pin']==m['source_pin']=='8949313b0a56c0c3bd52509fb363edb4eefba3e4'
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=REPO,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for label,key,pin in [('source','git_sources',new['source_pin']),('source-old','old_git_sources',old['source_pin'])]:
 for row in m[key]:
  proc.stdin.write((pin+':'+row['path']+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert header[1]==b'blob';raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==row['sha256']==rows[label+'/'+row['path']]['sha256'];assert len(raw)==row['bytes']
proc.stdin.close();assert proc.wait()==0
current={r['path']:r['sha256'] for r in m['git_sources']};prior={r['path']:r['sha256'] for r in m['old_git_sources']};assert len(current)==3520 and len(prior)==3519
assert {k for k,v in current.items() if prior.get(k)!=v}=={'src/typechecker.c','src/module.c','src/repl.c','src/nanolang.h','tests/test_checker_owned_contexts.c','Makefile.gnu','docs/ROADMAP.md','docs/SOURCE_C_IMPORTS.md'}
expected=['test-gc-shutdown-sanitizers','test-env-scoping','test-source-c-imports','test-checker-owned-contexts','test-module-rejection-rollback','test-repl-callable-lifetimes','test-repl-state','test-repl-session']
assert len(new['profiles'])==3 and len(old['profiles'])==2
for label,q,source in [('',old,prior),('repaired/',new,current)]:
 assert b'NANO_SHADOW_TIMEOUT_SECONDS' in data('journal/'+label+'run.py')
 assert q['build_timeout']==1800 and q['full_source_attempts']==0
 a=js(label+'audit');assert a['source_pin']==q['source_pin']
 for profile,audit in zip(q['profiles'],a['profiles'],strict=True):
  tree=profile['tree'];assert profile['initial_outputs']==dict(obj=False,lib=False,bin=False)
  assert profile['before']==profile['after_build']==profile['after_focused']==source
  assert profile['build']['exit']==0 and not profile['build']['timed_out'] and profile['build']['timeout']==1800
  assert audit['all_tracked_diff_exit']==0 and data('journal/'+label+profile['mode']+'-tracked-diff.stdout')==b''
  assert audit['compiler_sha256']==profile['image_sha256']==sha(provider(tree+'/bin/nanoc_c'))
  for name,row in profile['outputs'].items():assert sha(provider(tree+'/'+name))==row['sha256'] and len(provider(tree+'/'+name))==row['bytes']
  compiles=js(label+profile['mode']+'-compiles');execs=js(label+profile['mode']+'-execs');assert len(compiles)==221==audit['compiled_translation_units']
  for c in compiles:
   assert '-c' in c['args'] and c['args'][c['args'].index('-o')+1]==c['output'];assert sha(provider(tree+'/'+c['output']))==c['sha256']
   assert any(e['executable']=='/usr/bin/cc' and not e['argv_truncated'] and e['args']==c['args'] for e in execs)
  link=audit['compiler_link'];assert not link['argv_truncated'] and len(audit['link_inputs'])==169
  assert any(e==link for e in execs)
  for name,digest in audit['link_inputs'].items():assert sha(provider(tree+'/'+name))==digest
  for name,digest in audit['project_inputs'].items():
   if name in source:assert source[name]==digest
   else:
    raw=provider(tree+'/'+name);assert sha(raw)==digest
    if not name.startswith('obj/'):
     assert name in audit['tracked_inputs']
     assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==audit['tracked_inputs'][name]['oid']
     assert sha(subprocess.check_output(['git','show',q['source_pin']+':'+name],cwd=REPO))==digest
  assert all(e['executable'] in ('/usr/bin/env','/usr/bin/bash','/bin/bash') or e['executable'].endswith('/bash') for e in audit['truncated_shell_execs'])
  for path,digest in audit['executed_files_after'].items():assert sha(provider(path))==digest
  if label:
   assert profile['focused']['exit']==0 and profile['focused']['outer_timeout']==600 and profile['focused']['args'][-8:]==expected
   out=provider(tree+'/focused.stdout');err=provider(tree+'/focused.stderr')
   assert b'All 13 gc_struct tests passed.' in out and b'=== 60 passed, 0 failed ===' in out
   assert b'I preserve caller/registry contexts, growth, rollback and repeated match binding ownership.' in out
   assert re.findall(rb'Ran (\d+) tests? in',err)==[b'10',b'20',b'3',b'15',b'16'];assert err.count(b'\nOK\n')==5
   assert b'ERROR: AddressSanitizer' not in err and b'ERROR: LeakSanitizer' not in err and b'runtime error:' not in err
   assert any(k.startswith('provider/'+profile['source_cases']+'/') and k.endswith('.nano') for k in rows)
   if profile['mode']=='san':
    binary=provider(tree+'/bin/nanoc_c');assert b'__asan_init' in binary and b'__ubsan_handle' in binary
   if profile['mode']=='production':assert '-O3' in profile['flags'] and '-ftree-vectorize' in profile['flags']
assert old['profiles'][0]['focused']['exit']==0 and old['profiles'][1]['focused']['exit']==2
failure=provider(old['profiles'][1]['tree']+'/focused.stderr');assert b'FAILED (failures=2)' in failure and b'11 byte(s)' in failure and b'31 byte(s)' in failure
red=js('owned-context-red');assert sha(data('journal/owned-context-red-test.c'))==red['source_sha256'];assert [r['exit'] for r in red['commands']]==[0,1,1,1]
for name in ['expression','statement']:assert b'LeakSanitizer' in provider(red['work']+'/'+name+'.stderr')
assert b'AddressSanitizer' in provider(red['work']+'/contexts.stderr')
red2=js('retained-source-red');assert red2['exit']==1;assert b'FAILED (failures=2)' in provider(red2['work']+'/stderr')
local=js('owned-context-localize');assert len(local['commands'])==9 and all(r['exit']==0 for r in local['commands'])
assert new['tools']==new['tools_after']==old['tools']
for path,digest in new['tools'].items():assert sha(provider(path))==digest
assert {k for k,v in m['acceptance'].items() if v}=={'fresh_complete_normal','fresh_complete_sanitizer','fresh_complete_production','focused_original_targets','checker_context_and_match_ownership'}
print(json.dumps(dict(result='PASS',source=m['source_pin'],records=len(rows),git_inputs=len(current),native_images=len(images),fresh_profiles=3,failed_profiles_retained=2,compiler_translation_units_per_profile=221,compiler_link_inputs_per_profile=169,source_c_methods_per_profile=20,lexical_methods_per_profile=10,repl_methods_per_profile=34,full_compiler_source=False,paired_platform=False,whole_release=False),indent=2))
