"""I verify native stacks without assigning a historical timeout cause."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'9d9a4d0c0ac4e06cb7990371ab54cf740d84871c5ad38f288b09e8fbae26eff3');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('stack/result.json');q=js('stack/stack-result.json')
assert m['source_commit']==q['source_pin']==s['source_base']=='3021c729bdb5a204b23de28a3baa48777c0ae199'
assert m['diagnostic_only'] and not any(m[n] for n in ['product_acceptance','historical_cause','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_observation'] and q['all_guards_match'] and not q['inferior_remaining']
assert q['ordinary_seconds']==10 and q['outer_seconds']==300 and q['sample_seconds']==8
assert len(q['samples'])==8 and 8<=q['sample_elapsed']<9
assert sum('__strlen_asimd' in x['functions'] and 'nstr_char_at' in x['functions'] and any('tokenize_string' in f for f in x['functions']) for x in q['samples'])==7
assert q['inferior_argv']==q['observed_argv']
for x in q['samples']:assert x['stack'].encode() in data('stack/gdb-mi.log')
for group in ['source','images','tools','hosts']:
 assert q[group+'_after']==s[group+'_before']==s[group+'_after']
 for name,h in s[group+'_before'].items():assert rows[('stack/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']==rows['stack/source/bin/nanoc_emitter_phase']['sha256']
assert q['driver_sha256']==q['driver_after']==rows['stack/sample.py']['sha256']
assert q['debugger_sha256']==q['debugger_after']==rows['external/'+q['debugger'].lstrip('/')]['sha256']
assert str(Path(m['roots']['stack'])/'mark.py').encode() in data('stack/phase-recorder')
assert [x['phase'] for x in q['phases']]==['inputs-start','imports-done']
assert [json.loads(line) for line in data('stack/phases.jsonl').decode().splitlines()]==q['phases']
control=js('stack/mi-control-gvv57yys/stack-result.json');assert control['accepted_observation'] and control['terminal'] and not control['inferior_remaining'] and len(control['samples'])==3
first=js('stack/mi-control/stack-result.json');assert first['terminal'] and 'exception' in first and not first['inferior_remaining']
parent=base.parent/'shadow-helper-cold-bootstrap-linux-20261124';pm,pr,pdata,pjs=read_archive(parent,'c8c6cfe80993b0610a07576d13338b282a402d87fa6c636231d7680f923300e5');b=pjs('bootstrap/bootstrap-result.json');assert b['original_make_pass'] and q['producer_sha256']==pr['bootstrap/source/bin/nanoc_stage2']['sha256']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; eight bounded native samples, seven strlen/char_at lexer stacks, control failure and repair retained, stable source/tool/host/producer. No performance gain, historical cause, complete native or release claim.'%len(rows))
