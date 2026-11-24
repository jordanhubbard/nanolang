"""I verify one bounded native observation without assigning historical cause."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'29c6f69ad2cc0cdd71467c00fd49f5e9a45cd97296dc9171b8e8e6bbbb79dece');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('phase/result.json');a=js('phase/admission-result.json');q=js('phase/observation-result.json')
assert m['diagnostic_only'] and not any(m[n] for n in ['product_acceptance','historical_cause','new_bootstrap','whole_release'])
assert all(r['terminal'] for r in [s,a,q]) and s['admitted_setup'] and a['accepted'] and q['all_guards_match']
assert q['source_pin']==s['source_base']==m['source_commit']=='3021c729bdb5a204b23de28a3baa48777c0ae199'
assert q['diagnostic_only'] and not q['unchanged_full_retry'] and not q['new_bootstrap'] and not q['whole_release']
assert q['timeout']==300 and q['ordinary_seconds']==10 and q['timed_out'] and q['exit']==-9 and not q['product']
assert q['deadline_observed_monotonic']-q['start_monotonic']>=300
assert q['environment']['NANO_SHADOW_TIMEOUT_SECONDS'] is None and q['environment']['NANO_CC'] is None and q['environment']['NANO_CFLAGS'] is None and q['environment']['NANO_LDFLAGS'] is None and q['environment']['CMD_TIMEOUT'] is None
for group in ['source','images','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']
 for name,h in q[group+'_before'].items():assert rows[('phase/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['producer_sha256']==q['producer_after']==rows['phase/source/bin/nanoc_emitter_phase']['sha256']
assert q['phase_sources']==q['phase_sources_after']
for name,h in q['phase_sources'].items():assert rows['phase/'+name]['sha256']==h
assert str(Path(m['roots']['phase'])/'mark.py').encode() in data('phase/phase-recorder')
phases=[json.loads(line) for line in data('phase/phases.jsonl').decode().splitlines()];assert phases==q['phases']
assert [p['phase'] for p in phases]==['inputs-start','imports-done','lex-done','parse-done','checked-emit-start','typecheck-start']
assert all(phases[i]['monotonic']<phases[i+1]['monotonic'] for i in range(len(phases)-1))
assert q['argv']==[q['producer'],str(Path(m['roots']['phase'])/'source/tests/selfhost_shadow_emitter.nano'),'-o',str(Path(m['roots']['phase'])/'products/emitter.native')]
assert not any(n.endswith('/source/bin/nanoc_stage2') or n.endswith('/source/bin/nanoc_bootstrap.json') for n in rows)
parent=base.parent/'shadow-helper-cold-bootstrap-linux-20261124';pm,pr,pdata,pjs=read_archive(parent,'c8c6cfe80993b0610a07576d13338b282a402d87fa6c636231d7680f923300e5');b=pjs('bootstrap/bootstrap-result.json');assert b['terminal'] and b['original_make_pass']
assert q['producer_sha256']==pr['bootstrap/source/bin/nanoc_stage2']['sha256']
for name,h in q['images_before'].items():assert h==pr['bootstrap/source/'+name]['sha256']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in q['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: single300/default10 timeout, no product, stable guards; phase offsets %s. New-root observation only, no historical cause, performance gain, full native/paired/pure-translator/release acceptance.'%(len(rows),[(p['phase'],round(p['monotonic']-q['start_monotonic'],3)) for p in phases]))
