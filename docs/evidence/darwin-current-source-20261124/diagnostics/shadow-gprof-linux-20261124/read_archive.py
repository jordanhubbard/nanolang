"""I verify a private CPU-prefix diagnostic and an independent binding witness."""
from pathlib import Path
import sys,json,subprocess,hashlib,gzip,posixpath
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'e21ff5b5d52ab42f547dc5ec6310f9a1c6cbae6f48a4b9b021de1f9d77354fb8')
sha=lambda raw:hashlib.sha256(raw).hexdigest();s=js('journal/setup-result.json');q=js('journal/observation/result.json');analysis=js('journal/analysis.json')
assert m['diagnostic_only'] and not m['full_source_accepted'] and not m['whole_release']
assert m['source_pin']==s['pin']==q['pin']=='221255c7a56b903b5fd34472a9109972230e6ab7'
assert s['terminal'] and s['admitted'] and q['terminal'] and q['retained'] and not q['qualification'] and not q['whole_release']
assert not s['native_processes'] and len(s['cpu_rounds'])==3 and all(r['cpu']>=1 and r['share']>=.75 for r in s['cpu_rounds'])
assert s['source_before']==s['source_after']==q['source_before']==q['source_after'] and len(s['source_before'])==3522
assert s['tools']==s['tools_after']==q['tools_before']==q['tools_after']
assert s['provider_copies']==s['provider_after'] and all(c['exit']==0 and not c['timed_out'] for c in s['commands'])
assert s['production_changes_from_provider']==['src/typechecker.c']
assert s['image_sha256']==q['image_before']==q['image_after']==rows['journal/source/bin/nanoc_profile']['sha256']
assert s['observer_sha256']==rows['journal/observer.c']['sha256']
for name,h in s['tools'].items():assert rows['tool/'+name.lstrip('/')]['sha256']==h
for name in ['eval-profile-object','env-profile-object']:
 c=next(c for c in s['commands'] if c['label']==name);assert '-pg' in c['args']
link=next(c for c in s['commands'] if c['label']=='private-link')
for flag in ['-pg','-Wl,--wrap=run_shadow_tests_scope','-Wl,--wrap=env_get_function','-Wl,--wrap=env_get_function_at']:assert flag in link['args']
parent=json.loads((base.parent/'list-source-linux-20261124/manifest.json').read_text());prior={r['path']:r for r in parent['files']}
for name,h in s['linked_providers'].items():assert prior['normal/'+name]['sha256']==h
for name in ['nanoc_profile','nanoc_control','nano_vm','nanoisa','nvm2c']:
 head=data('journal/source/bin/'+name)[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
assert rows['journal/source/bin/nanoc_control']['sha256']==s['baseline_sha256']
for variant in ['baseline','profile']:
 head=data('journal/control/'+variant+'-native')[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
control=[json.loads(l) for l in data('journal/control/profile-events.jsonl').splitlines()];assert [e['stage'] for e in control]==['begin','dump-start','dump-end','returned']
assert control[1]['cpu_seconds']<4 and control[1]['calls']>0
assert data('journal/control/profile-gmon.'+str(control[0]['pid'])).startswith(b'gmon')
assert b'eval_expression' in data('journal/control-gprof.stdout')
command=q['command'];assert command['exit']==1 and not command['timed_out'] and command['timeout']==150 and command['seconds']<150 and command['shadow_timeout']=='original default10'
events=[]
for line in data('journal/observation/stderr').splitlines():
 try:events.append(json.loads(line))
 except ValueError:pass
starts=[e for e in events if e.get('event')=='shadow-start'];ends=[e for e in events if e.get('event')=='shadow-complete'];assert len(starts)==q['shadow_starts']==1011 and len(ends)==q['shadow_completions']==1010 and all(e['failures']==0 for e in ends) and not q['failed_shadows']
phases=[e for e in events if e.get('event')=='cseed-shadow-phase' and e.get('phase')=='supervisor-status'];assert len(phases)==1
supervisor=phases[0];assert supervisor['deadline_seconds']==10 and supervisor['signal']==14 and supervisor['parent_timeout'] and not supervisor['completed']
profile=[json.loads(l) for l in data('journal/observation/events.jsonl').splitlines()];assert profile==q['profile_events'] and [e['stage'] for e in profile]==['begin','dump-start','dump-end']
assert len({e['pid'] for e in profile})==1 and profile[0]['pid']==supervisor['child_pid']
assert 4<=profile[1]['cpu_seconds']<4.5 and profile[1]['calls']==2339840 and profile[1]['wall_seconds']<10
child='gmon.'+str(profile[0]['pid']);assert data('journal/observation/'+child).startswith(b'gmon') and rows['journal/observation/'+child]['sha256']==q['gmon'][child]['sha256']
assert q['gprof']['exit']==0 and q['gprof']['args'][-1].endswith('/'+child)
assert b'Flat profile' in data('journal/observation/gprof.stdout')
assert 'journal/observation/seed' not in rows
assert abs(analysis['sampled_executable_seconds']-1.8)<1e-8 and analysis['prefix_cpu_seconds']==profile[1]['cpu_seconds']
flat={r['name']:r for r in analysis['flat_profile']};assert flat['gc_alloc']['self_sampled_seconds']==.41 and flat['env_define_var_impl']['self_sampled_seconds']==.21
assert flat['env_get_function']['calls']==2339839 and flat['create_struct']['calls']==429431
assert all(c['exit']==0 for c in js('journal/admission.json'))
assert js('journal/prior-observation/perf-record-admission.json')['exit']!=0 and 'perf_event_paranoid' in js('journal/prior-observation/perf-record-admission.json')['stderr']
w=js('journal/witness-result.json');guards=js('journal/witness-provider-guards.json');assert all(c['exit']==0 for c in w['commands'])
assert w['observation']==dict(literal_binding_copied=True,managed_object_delta=1,managed_allocated_delta=46,identifier_independent=True,borrowed_identity=True)
assert guards['current_pin']==s['pin'] and guards['provider_source_pin']=='8949313b0a56c0c3bd52509fb363edb4eefba3e4'
assert w['env_source_sha256']==s['source_before']['src/env.c']==guards['matching_current_sources']['src/env.c']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(spec):
 proc.stdin.write((spec+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert head[1]==b'blob',spec;raw=proc.stdout.read(int(head[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in s['source_before'].items():assert rows['journal/source/'+name]['sha256']==h==sha(blob(s['pin']+':'+name))
archive=guards['original_archive_manifest'];old=blob(s['pin']+':'+archive['git_path']);assert sha(old)==archive['sha256'];old=json.loads(old);oldrows={r['path']:r for r in old['files']}
for name,h in guards['providers'].items():
 row=oldrows['provider/'+guards['provider_root']+'/'+name];assert row['sha256']==h
 path=posixpath.normpath(posixpath.join(posixpath.dirname(archive['git_path']),row['artifact']));assert path.startswith('docs/evidence/')
 zipped=blob(s['pin']+':'+path);assert sha(zipped)==row['compressed_sha256'];raw=gzip.decompress(zipped);assert len(raw)==row['bytes'] and sha(raw)==h
for name,h in guards['matching_current_sources'].items():assert h==s['source_before'][name]==sha(blob(guards['provider_source_pin']+':'+name))
proc.stdin.close();assert proc.wait()==0
print('PASS integrity: %d records/%d selected Git sources; baseline/profile source+native controls pass; private4CPU prefix retained, default10 source FAIL1010 completions; real fresh-record binding adds1 managed owner/46 accounted bytes. No qualification/cause/gain claim.'%(len(rows),len(s['source_before'])))
