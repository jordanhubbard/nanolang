"""I retain an unsuccessful bounded phase observation without claiming stacks."""
from pathlib import Path
import json,sys,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'ba9b3fcf9dc29bea180733c1470df8aa0e3f28df5d564976d26a629d6dfdc362');sha=lambda b:hashlib.sha256(b).hexdigest();s=js('stack/result.json');a=js('stack/admission-result.json');q=js('stack/stack-result.json');f=js('stack/final-guard-result.json')
assert m['source_commit']==s['source_base']==q['source_pin']==f['source_pin']=='8da979937c9579c922ff515a8100ad9eaba7ab02'
assert m['diagnostic_only'] and m['development_image'] and not any(m[k] for k in ['product_acceptance','historical_cause','performance_claim','new_bootstrap','own_stage2_acceptance','whole_release'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['stage_alias']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert q['terminal'] and q['exception']=='AssertionError()' and not q['samples'] and not q.get('accepted_observation') and not q['inferior_remaining']
assert q['wait_phase']=='program-lowering-start' and q['outer_seconds']==300 and q['sample_seconds']==8 and q['ordinary_seconds']==10 and q['development_image']
assert b'(satisfied #f)' in data('stack/phase-wait.stdout') and b'(reason timeout)' in data('stack/phase-wait.stdout')
assert b'(timeout-ms 280000)' in data('stack/sample.py') and b'threading.Timer(300,expire_observation)' in data('stack/sample.py')
assert b'killed' in data('stack/gdb-mi.log') and b'14^exit' in data('stack/gdb-mi.log')
assert f['terminal'] and f['read_only_final_guard'] and f['all_guards_match'] and not f['inferior_remaining'] and not f['product_exists'] and f['sampler_receipt_sha256']==rows['stack/stack-result.json']['sha256']
assert f['driver']==q['driver_sha256']==rows['stack/sample.py']['sha256'] and f['debugger']==q['debugger_sha256']==rows['external/'+q['debugger'].lstrip('/')]['sha256']
assert f['producer']==s['producer_sha256']==q['producer_sha256']==rows['stack/source/bin/nanoc_development_observer']['sha256']
for group in ['source','images','tools','hosts']:
 assert f[group]==s[group+'_before']==s[group+'_after']
 for name,h in f[group].items():assert rows['stack/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert f['phase_sources']==q['phase_sources_before']
for name,h in f['phase_sources'].items():assert rows['stack/'+name]['sha256']==h
assert str(Path(m['roots']['stack'])/'mark.py').encode() in data('stack/phase-recorder')
phases=[json.loads(line) for line in data('stack/phases.jsonl').splitlines()];assert f['phases']==phases and [v['phase'] for v in phases]==['inputs-start','imports-done','lex-done','parse-done','checked-emit-start','typecheck-start']
assert all(phases[i]['monotonic']<phases[i+1]['monotonic'] for i in range(len(phases)-1))
controls=[n for n in rows if n.startswith('stack/mi-control-') and n.endswith('/stack-result.json')];assert len(controls)==1;c=js(controls[0]);assert c['terminal'] and c['accepted_observation'] and c['gdb_exit']==0 and not c['inferior_remaining'] and len(c['samples'])==3 and c['driver_sha256']==q['driver_sha256']
assert not any(n.endswith('/source/bin/nanoc_stage2') or n.endswith('/source/bin/nanoc_bootstrap.json') or n=='stack/products/profile.native' for n in rows)
pm,pr,pd,pj=read_archive(base.parent/'symbol-copy-compiler-development-linux-20261124','3dccdc51bd3a92e222d69f005c52ed038418d5af63e3ef4da35a0d2edf29de68');dev=pj('development/development-native-result.json');assert dev['development_native_accepted'] and dev['all_guards_match']
assert s['development_receipt_sha256']==pr['development/development-native-result.json']['sha256'] and s['producer_sha256']==dev['products']['compiler-development.native']['sha256']==pr['development/products/compiler-development.native']['sha256']
assert s['hosts_before']==dev['hosts_after'] and s['development_source_base']==dev['source_base'] and s['development_overlays']==dev['overlays']
assert s['development_core_sources']=={n:h for n,h in dev['source_after'].items() if n.startswith(('src_nano/','src/','modules/','schema/'))}
for n,h in s['development_core_sources'].items():assert rows['stack/source/'+n]['sha256']==h
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in s['source_before'].items():
 p.stdin.write((m['source_commit']+':'+n+'\n').encode());p.stdin.flush();header=p.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=p.stdout.read(int(header[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,n
p.stdin.close();assert p.wait()==0
print('PASS %d records: unsuccessful280-second lowering-phase wait, zero lowering samples, native control3 samples; raw debugger kill/exit, independent300 cap, final source/tool/host/image/producer guards and no inferior/product. No cause/gain, repair, Stage2/emitter12/paired/pure-translator/release acceptance.'%len(rows))
