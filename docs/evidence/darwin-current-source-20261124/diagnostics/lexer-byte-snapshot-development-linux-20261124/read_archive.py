"""I verify lexer semantics and preserve the unsuccessful factory boundary."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'b7caa84b508dec5cc7cc160cd8d3acdc1c855946e9f643319c1fa08c359460f8');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('development/result.json');p=js('development/probe-result.json');sem=js('development/semantic-result.json');dev=js('development/development-native-result.json');seed=js('development/compiler-seed-result.json');reg=js('development/regression-result.json');gate=js('development/native36-result.json');factory=js('development/factory-result.json');boundary=js('development/evidence-boundaries.json')
assert all(q['terminal'] for q in [s,p,sem,dev,seed,reg,gate,factory])
assert m['diagnostic_only'] and not any(m[n] for n in ['product_acceptance','historical_cause','new_bootstrap','whole_release'])
assert m['source_commit']==s['source_base']=='3021c729bdb5a204b23de28a3baa48777c0ae199'
assert m['overlay_files']==['src_nano/compiler/lexer.nano'] and s['working_overlay']
assert p['accepted'] and p['all_five_routes_pass'] and len(p['routes'])==5
expected=b'0:4:4\n1:1:2\n2:3:8\n3:3:6\n4:3:2\n5:70:3\n6:7:0\n7:8:0\n8:0:0\n'
assert all(x['output'].encode()==expected for x in p['routes'])
assert sem['accepted'] and sem['all_24_cases_five_routes_pass'] and len(sem['cases'])==24 and len(sem['commands'])==152
for index,case in enumerate(sem['cases']):
 key='development/semantic/'+case['case'];assert rows[key]['sha256']==case['sha256']
 report=data('development/semantic/'+Path(case['case']).stem+'.expected');assert sha(report)==case['report_sha256']
 assert len(case['routes'])==5
 for offset in range(6):assert data('development/semantic-command-%d.stdout'%(9+index*6+offset))==report
candidate=data('development/semantic/candidate-report.nano');baseline=data('development/semantic/baseline-report.nano');import_path=str(Path(m['roots']['development'])/'semantic/baseline-lexer.nano')
assert baseline.replace(('"'+import_path+'"').encode(),b'"src_nano/compiler/lexer.nano"',1)==candidate
assert rows['development/semantic/baseline-lexer.nano']['sha256']!=s['overlays']['src_nano/compiler/lexer.nano']
assert candidate==data('regression-source/tests/lexer_snapshot_report.nano')
corpus=js('regression-source/tests/lexer_snapshot_cases.json');assert corpus['baseline_commit']==m['source_commit'] and corpus['baseline_lexer_sha256']==rows['development/semantic/baseline-lexer.nano']['sha256']
assert len(corpus['cases'])==24
for index,case in enumerate(corpus['cases']):
 assert bytes.fromhex(case['source_hex'])==data('development/semantic/case-%02d.input'%index)
 assert case['expected'].encode()==data('development/semantic/case-%02d.expected'%index)
assert reg['accepted'] and reg['tests_run']==1 and not any(reg[n] for n in ['failures','errors','skips']) and len(reg['route_names'])==5 and len(reg['commands'])==127
for index,case in enumerate(corpus['cases']):
 for route in range(5):assert data('development/regression-command-%d.stdout'%(8+index*5+route))==case['expected'].encode()
for q in [s,p,sem,seed,dev,reg,gate,factory]:
 assert q['all_guards_match']
 for group in ['source','images','tools','hosts']:
  if group+'_before' not in q:continue
  assert q[group+'_before']==q[group+'_after']
  for name,h in q[group+'_before'].items():assert rows[('development/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
for name,h in reg['new_inputs_before'].items():assert h==reg['new_inputs_after'][name]==rows['regression-source/'+name]['sha256']
assert seed['development_seed_accepted'] and seed['ordinary_seconds']==10 and seed['timeout']==600 and seed['exit']==0
assert seed['product']==dev['input_module'] and dev['development_native_accepted'] and all(c['exit']==0 and c['timeout']==600 for c in dev['commands'])
for name,identity in dev['products'].items():assert identity['sha256']==rows['development/products/'+name]['sha256'] and identity['bytes']==rows['development/products/'+name]['bytes']
assert gate['accepted_component'] and gate['methods']==36 and gate['commands'][0]['exit']==0 and gate['ordinary_seconds']==10 and not gate['own_stage2_acceptance']
assert b'Ran 36 tests' in data('development/native36-command-1.stderr') and data('development/native36-command-1.stderr').rstrip().endswith(b'OK')
assert gate['producer_before']==gate['producer_after']==factory['producer_before']==factory['producer_after']==dev['products']['compiler-development.native']['sha256']
assert not factory.get('accepted_component') and factory['ordinary_seconds']==10 and len(factory['commands'])==1
assert factory['commands'][0]['budget']==300 and factory['commands'][0]['seconds']>=300 and factory['exception'].startswith('TimeoutExpired(')
assert not boundary['factory_published_product_exists'] and not boundary['factory_partial_stdout_stderr_preserved'] and boundary['no_unchanged_factory_retry'] and boundary['emitter12_not_run']
assert 'development/products/emitter-development.native' not in rows
for q in [p,sem,reg,gate]:assert all(c.get('exit')==0 for c in q['commands'])
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git_hash(commit,name):
 proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';return sha(raw)
for name,h in s['source_before'].items():assert git_hash(m['overlay_source_commit'] if name in m['overlay_files'] else m['source_commit'],name)==h,name
assert git_hash(m['source_commit'],'src_nano/compiler/lexer.nano')==corpus['baseline_lexer_sha256']
for name,h in reg['new_inputs_before'].items():assert git_hash(m['regression_source_commit'],name)==h,name
proc.stdin.close();assert proc.wait()==0
parent=base.parent/'shadow-helper-cold-bootstrap-linux-20261124';pm,pr,pdata,pjs=read_archive(parent,'c8c6cfe80993b0610a07576d13338b282a402d87fa6c636231d7680f923300e5');assert pjs('bootstrap/bootstrap-result.json')['original_make_pass']
assert s['producer_sha256']==pr['bootstrap/source/bin/nanoc_stage2']['sha256']==rows['development/source/bin/nanoc_emitter_phase']['sha256']
print('PASS %d records: token-byte five routes, original-lexer24x5 full records, permanent regression24x5, guarded development compiler/native36; original300 factory TIMEOUT/no product, partial child streams absent and disclosed. No current Stage2/emitter12/fresh/paired/pure-translator/release acceptance.'%len(rows))
