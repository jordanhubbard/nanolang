from pathlib import Path
import re,json,hashlib
p=Path('/tmp/nanolang-admission-stage-once');rows=[]
for line in (p/'emitter.log').read_text().splitlines():
 if not line.startswith('[shadow-progress]'):continue
 d={k:float(v) if '.' in v else int(v) for k,v in re.findall(r'(\w+)=([0-9.]+)(?=\s)',line)}
 m=re.search(r' (enter|leave) fn=(\d+) pc=(\d+) name=(\S+)$',line);assert m
 d.update(event=m[1],fn=int(m[2]),pc=int(m[3]),name=m[4]);rows.append(d)
 assert d['full_calls']+d['reuse_calls']==d['admission_calls']
 assert d['full_ns']+d['reuse_ns']==d['admission_ns']
 assert sum(v for k,v in d.items() if k.startswith('outer_'))==d['admission_ns']
def delta(a,b):
 d={k:b[k]-a[k] for k in a if isinstance(a[k],(int,float)) and k not in ('fn','pc','admission_ms')}
 d['admission_fraction']=d['admission_ns']/1e6/d['elapsed_ms'];return d
before=json.loads((p/'emitter-providers-before.json').read_text());after=json.loads((p/'emitter-providers-after.json').read_text())
r={'markers':len(rows),'first':rows[0],'last':rows[-1],'delta':delta(rows[0],rows[-1]),'intervals':[],'sources_equal':(p/'sources-before.json').read_bytes()==(p/'sources-after.json').read_bytes(),'tools_equal':(p/'tools-before.json').read_bytes()==(p/'tools-after.json').read_bytes(),'providers':{'added':len(after.keys()-before.keys()),'removed':len(before.keys()-after.keys()),'changed':sum(before[k]!=after[k] for k in before.keys()&after.keys())}}
for name in ['$shadow_200_parse_owned_pattern','$shadow_199_parser_mark_owned']:
 selected=[d for d in rows if d['name']==name]
 if len(selected)==2:r['intervals'].append({'name':name,'delta':delta(*selected)})
old=json.loads(Path('/tmp/nanolang-admission-followup-once/observation.json').read_text());r['baseline']={'pin':'1e53cba55','delta':old['delta'],'identical_marker_boundaries':all(rows[i][k]==old[t][k] for i,t in [(0,'first'),(-1,'last')] for k in ['event','fn','pc','name'])}
r['limits']='Nested inclusive query durations overlap; do not sum. Only successful core intervals; final unmarked interval and early refusals unmeasured.'
(p/'observation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['delta'],indent=2));print(r['last']);print(r['providers'])
