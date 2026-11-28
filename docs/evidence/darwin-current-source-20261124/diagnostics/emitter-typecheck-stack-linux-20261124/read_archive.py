"""I verify eight bounded development-native typechecker stacks."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'8d5b6f21c99fbfa797e69e6269dbfb4df5bfd99c5f71ad4dcb64ef64103efa01');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('stack/result.json');q=js('stack/stack-result.json');a=js('stack/admission-result.json')
assert m['source_commit']==q['source_pin']==s['source_base']=='0ba525350d266bbed3d8f661db68d2fe5d87d52b'
assert m['diagnostic_only'] and not any(m[n] for n in ['product_acceptance','historical_cause','new_bootstrap','whole_release'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['stage_alias']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['free_bytes']>15*1024**3 and a['filesystem']!='nfs'
assert len(s['cpu_rounds'])==2 and all(r['cpu']>=1 and r['share']>=.75 for r in s['cpu_rounds'])
assert q['terminal'] and q['accepted_observation'] and q['all_guards_match'] and not q['inferior_remaining'] and q['gdb_exit']==0
assert q['ordinary_seconds']==10 and q['outer_seconds']==300 and q['sample_seconds']==8 and q['development_image'] and q['wait_phase']=='typecheck-start'
assert len(q['samples'])==8 and 8<=q['sample_elapsed']<9 and q['inferior_killed_monotonic']-q['start_monotonic']<300 and not q.get('outer_deadline_fired')
assert sum('nl_env_copy_symbols' in x['functions'] for x in q['samples'])==2
assert sum('nl_env_lookup' in x['functions'] for x in q['samples'])==2
assert sum('__memcpy_sve' in x['functions'] for x in q['samples'])==5
assert sum('_int_malloc' in x['functions'] for x in q['samples'])==2
assert q['inferior_argv']==q['observed_argv']==[s['producer'],str(Path(m['roots']['stack'])/'source/tests/selfhost_shadow_emitter.nano'),'-o',str(Path(m['roots']['stack'])/'products/profile.native')]
for x in q['samples']:assert x['stack'].encode() in data('stack/gdb-mi.log')
for group in ['source','images','tools','hosts']:
 assert q[group+'_after']==s[group+'_before']==s[group+'_after']
 for name,h in s[group+'_before'].items():assert rows[('stack/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']==rows['stack/source/bin/nanoc_development_observer']['sha256']
assert q['driver_sha256']==q['driver_after']==rows['stack/sample.py']['sha256']
assert q['debugger_sha256']==q['debugger_after']==rows['external/'+q['debugger'].lstrip('/')]['sha256']
assert q['phase_sources_before']==q['phase_sources_after']
for name,h in q['phase_sources_before'].items():assert rows['stack/'+name]['sha256']==h
assert str(Path(m['roots']['stack'])/'mark.py').encode() in data('stack/phase-recorder')
phases=[json.loads(line) for line in data('stack/phases.jsonl').decode().splitlines()];assert phases==q['phases']
assert [x['phase'] for x in phases]==['inputs-start','imports-done','lex-done','parse-done','checked-emit-start','typecheck-start']
assert all(phases[i]['monotonic']<phases[i+1]['monotonic'] for i in range(len(phases)-1))
assert phases[-1]['monotonic']<=q['sample_start_monotonic']<q['samples'][0]['monotonic']
controls=[n for n in rows if n.startswith('stack/mi-control-') and n.endswith('/stack-result.json')];assert len(controls)==1
control=js(controls[0]);assert control['terminal'] and control['accepted_observation'] and not control['inferior_remaining'] and len(control['samples'])==3 and control['driver_sha256']==q['driver_sha256']
assert not any(n.endswith('/source/bin/nanoc_stage2') or n.endswith('/source/bin/nanoc_bootstrap.json') or n=='stack/products/profile.native' for n in rows)
parent=base.parent/'lexer-byte-snapshot-development-linux-20261124';pm,pr,pdata,pjs=read_archive(parent,'b7caa84b508dec5cc7cc160cd8d3acdc1c855946e9f643319c1fa08c359460f8');dev=pjs('development/development-native-result.json');assert dev['development_native_accepted'] and dev['all_guards_match']
assert s['development_receipt_sha256']==pr['development/development-native-result.json']['sha256']
assert q['producer_sha256']==dev['products']['compiler-development.native']['sha256']==pr['development/products/compiler-development.native']['sha256']
assert s['hosts_before']==dev['hosts_after'] and s['development_source_base']==dev['source_base'] and s['development_overlays']==dev['overlays']
assert s['development_core_sources']=={n:h for n,h in dev['source_after'].items() if n.startswith(('src_nano/','src/','modules/','schema/'))}
for n,h in s['development_core_sources'].items():assert rows['stack/source/'+n]['sha256']==h
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; eight development-native samples after typecheck-start, symbol/record copies and allocations; exact source/development origin/host/tool/recorder guards, independent kill clock and three-sample control. No remaining inferior, product, historical cause, speedup, Stage2/emitter12/paired/pure-translator/release claim.'%len(rows))
