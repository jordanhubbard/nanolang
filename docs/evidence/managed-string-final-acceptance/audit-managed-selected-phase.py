import collections, hashlib, json, os, pathlib, subprocess, sys, time
import signal
signal.alarm(600)
start=time.monotonic(); base=pathlib.Path(sys.argv[1]); expected_tests=int(sys.argv[2]); expected_commands=int(sys.argv[3]); optimization=sys.argv[4]; problems=[]; refs={}; counters=collections.Counter()
def check(ok,message):
 if not ok: problems.append(message)
def load(name):return json.loads((base/name).read_text())
def references(value):
 if isinstance(value,dict):
  if 'object' in value and 'sha256' in value:
   h=value['sha256']; check(value['object']=='objects/'+h,'object path '+str(value['object']))
   check(h not in refs or refs[h]==value['bytes'],'conflicting length '+h);refs[h]=value['bytes'];counters['object_references']+=1
  else:
   for x in value.values():references(x)
 elif isinstance(value,list):
  for x in value:references(x)
manifest=load('report-sha256.json')
for name,digest in manifest.items():
 p=base/name;data=p.read_bytes();check(hashlib.sha256(data).hexdigest()==digest,'report digest '+name);counters['reports']+=1
 if name.endswith('.json'):references(json.loads(data))
object_git={}; object_bytes=0
for digest,size in refs.items():
 pass # unique byte hashing remains below
 data=(base/'objects'/digest).read_bytes();check(len(data)==size and hashlib.sha256(data).hexdigest()==digest,'archive '+digest)
 object_git[digest]=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();object_bytes+=len(data)
before=load('inputs-before.json');after=load('inputs-after.json');check(before==after,'phase endpoints differ')
phase=load('status.json');check(phase['passed'] and phase['tests_run']==expected_tests and phase['commands']==expected_commands and not phase['errors'] and not phase['failures'] and not phase['skipped'] and not phase['exception'] and not phase['retention_failures'],'phase result')
selected=load('selected-tests.json');check(len(selected)==expected_tests and len(set(selected))==expected_tests,'selected test inventory')
log=(base/'unittest.log').read_text();check(('Ran %d tests'%expected_tests) in log and log.rstrip().endswith('OK'),'unittest log');check(sum(' ... ok' in line for line in log.splitlines())==expected_tests,'individual test outcomes')
nonzero=[]
for i in range(1,expected_commands+1):
 prefix='command-%05d/'%i
 status=load(prefix+'status.json');a=load(prefix+'inputs-before.json');b=load(prefix+'inputs-after.json')
 check(a==b and a==before,'command input equality '+str(i));check(load(prefix+'input-equality.json')['equal'],'command equality record '+str(i))
 cleanup=status['cleanup'];check(not status['exception'] and not status['timed_out'] and cleanup and cleanup['confirmed'] and cleanup['reaped'] and cleanup['group_gone'] and not cleanup['errors'],'command cleanup '+str(i))
 req=load(prefix+'request.json');env=req['environment'];check(env['NMS_EMITTED_OPTIMIZATION']==optimization and env['LSAN_OPTIONS']=='' and 'detect_leaks=1' in env['ASAN_OPTIONS'],'command environment '+str(i))
 if status['returncode']:nonzero.append({'command':i,'returncode':status['returncode'],'argv':req['args']})
 counters['command_pairs']+=1
roots=[pathlib.Path(p).parents[1] for p in before['sources'] if p.endswith('/scripts/qualify_managed_strings.py')];check(len(roots)==1,'root discovery');root=roots[0]
pin=(base/'head.stdout').read_text().strip();check(pin==sys.argv[5],'pin')
entries=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','-z',pin]).split(b'\0'); git={}
for entry in entries:
 if entry:
  header,path=entry.split(b'\t',1);mode,kind,oid=header.decode().split();git[os.fsdecode(path)]=(mode,kind,oid)
scope=load('source-scope.json');expected={p for p in git if not p.startswith('docs/') or p in ('docs/MANAGED_STRING_FINAL_ACCEPTANCE.md','docs/ROADMAP.md')}
check(set(scope['included'])==expected,'source scope vs git');check(set(scope['excluded_documentation'])==set(git)-expected,'excluded scope vs git')
check({str(pathlib.Path(p).relative_to(root)) for p in before['sources']}==expected,'source map vs scope')
for path,row in before['sources'].items():
 rel=str(pathlib.Path(path).relative_to(root));mode,kind,oid=git[rel]
 if mode=='120000':
  data=os.readlink(path).encode();actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 else:actual=object_git.get(row.get('sha256'))
 check(actual==oid,'source vs git blob '+rel)
current_cache={}; current_maps={}; stat_mismatches=[]
for category,mapping in before.items():
 for name,row in mapping.items():
  p=pathlib.Path(name)
  if row.get('missing'):check(not p.exists(),'previous missing exists '+name);continue
  resolved=str(p.resolve());st=p.stat();ident=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
  if ident!=row['stat_identity']:stat_mismatches.append(name)
  if resolved not in current_cache:current_cache[resolved]=hashlib.sha256(p.read_bytes()).hexdigest()
  check(current_cache[resolved]==row['sha256'] and resolved==row['resolved'],'current '+name)
 current_maps[category]=len(mapping)
check(not stat_mismatches,'current stat identities changed')
print(json.dumps({'passed':not problems,'problems':problems,'base':str(base),'root':str(root),'pin':pin,'report_manifest_sha256':hashlib.sha256((base/'report-sha256.json').read_bytes()).hexdigest(),'counters':dict(counters),'unique_referenced_objects':len(refs),'archive_bytes':object_bytes,'current_map_counts':current_maps,'current_unique_files_hashed':len(current_cache),'current_stat_mismatches':stat_mismatches,'phase':phase,'selected_tests':selected,'nonzero_commands':nonzero,'limits':['Command input maps use recorded stat-keyed cached digests, not an independent byte read per command.','Phase endpoint digests are forced fresh by the frozen runner; audit independently rehashes each referenced archive object and current immutable input.','Source scope excludes historical docs except the acceptance contract and roadmap.','Tool/library inventory does not establish every transitive compiler tool or runtime loader selection.','Only the named phase audited; remaining acceptance remains separate.'],'seconds':time.monotonic()-start},indent=2))
