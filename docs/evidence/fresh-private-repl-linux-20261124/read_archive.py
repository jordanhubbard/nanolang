#!/usr/bin/env python3
"""I verify private REPL collection on complete fresh providers."""
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
q=js('result');assert q['source_pin']==m['source_pin']=='8949313b0a56c0c3bd52509fb363edb4eefba3e4'
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=REPO,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
source={}
for row in m['git_sources']:
 proc.stdin.write((m['source_pin']+':'+row['path']+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert header[1]==b'blob';raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==row['sha256']==rows['source/'+row['path']]['sha256'];source[row['path']]=row['sha256']
proc.stdin.close();assert proc.wait()==0 and len(source)==3520
assert q['before']==q['after'] and len(q['before'])==3388
assert all(source[name]==digest for name,digest in q['before'].items())
assert not q['default_ownership_changed'] and q['full_source_attempts']==0
assert q['tools_before']==q['tools_after']
for path,digest in q['tools_before'].items():assert sha(provider(path))==digest
assert q['public_before']==q['public_after']
for path,digest in q['public_before'].items():assert sha(provider(path))==digest
assert len(q['commands'])==13 and all(row['timeout']==180 for row in q['commands'])
assert [(row['label'],row['exit']) for row in q['commands'] if row['exit']!=0]==[('san-default7',1)]
assert len(q['profiles'])==3
for profile in q['profiles']:
 mode=profile['mode'];image=provider(profile['image']);assert sha(image)==profile['image_sha256'];assert profile['image_sha256'] in images
 assert profile['result']['tests']==41 and profile['result']['calls']==62
 assert (profile['result']['failures'],profile['result']['errors'],profile['result']['skips'])==(0,0,0)
 case=profile['cases'];assert json.loads(provider(case+'/result.json'))==profile['result']
 assert b'Ran 41 tests' in provider(case+'/tests.log') and provider(case+'/tests.log').endswith(b'\nOK\n')
 calls=json.loads(provider(case+'/calls.json'));assert len(calls)==62
 for n,call in enumerate(calls):
  assert call['args']==[profile['image']] and call['timeout']==30
  assert provider(case+'/'+str(n)+'-stdin')
  err=provider(case+'/'+str(n)+'-stderr');assert b'AddressSanitizer' not in err and b'runtime error:' not in err
 for path,digest in profile['providers'].items():assert sha(provider(path))==digest
 factory=next(row for row in q['commands'] if row['label']==mode+'-factory');assert '-Dcreate_environment=nano_repl_legacy_environment' in factory['args']
 assert factory['args'][factory['args'].index('-c')+1]==profile['tree']+'/src/env.c'
 link=next(row for row in q['commands'] if row['label']==mode+'-link');assert link['args'][link['args'].index('-o')+1]==profile['image']
 assert set(profile['providers'])=={s for s in link['args'] if s.endswith(('.o','.a'))}
 if mode=='san':assert b'__asan_init' in image and b'__ubsan_handle' in image and '-fsanitize=address,undefined' in factory['args']
 if mode=='production':assert '-O3' in factory['args'] and '-ftree-vectorize' in factory['args']
shim=provider(q['work']+'/environment.c');assert b'env_carrier_enable(env)' in shim and b'env_carrier_enable_collection(env, NULL, 0)' in shim
assert b'Environment *create_environment(void)' in shim
ordinary=q['default'];assert ordinary['result']['tests']==ordinary['result']['calls']==7
assert (ordinary['result']['failures'],ordinary['result']['errors'],ordinary['result']['skips'])==(3,0,0)
assert sha(provider(ordinary['image']))==ordinary['image_sha256']
calls=json.loads(provider(ordinary['cases']+'/calls.json'));assert len(calls)==7
summaries=[]
for n,call in enumerate(calls):
 assert call['args']==[ordinary['image']] and call['timeout']==30
 assert provider(ordinary['cases']+'/'+str(n)+'-stdin')
 summaries+=re.findall(rb'SUMMARY: AddressSanitizer: (\d+) byte\(s\) leaked in 6 allocation\(s\)',provider(ordinary['cases']+'/'+str(n)+'-stderr'))
assert sorted(map(int,summaries))==[168,172,172]
assert provider(ordinary['cases']+'/tests.log').endswith(b'FAILED (failures=3)\n')
assert {k for k,v in m['acceptance'].items() if v}=={'private_repl_41_three_profiles'}
print(json.dumps(dict(result='PASS',source=m['source_pin'],records=len(rows),git_inputs=len(source),native_images=len(images),source_guards=3388,private_methods_per_profile=41,private_profiles=3,private_native_sessions=186,default_methods=7,retained_default_failures=3,retained_default_leak_bytes=[168,172,172],default_ownership_enabled=False,full_compiler_source=False,whole_release=False),indent=2))
