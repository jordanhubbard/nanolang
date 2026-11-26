"""I verify the bounded caller-context reductions and optionally extract the nested case."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
args=argparse.ArgumentParser();args.add_argument('--extract',type=Path);options=args.parse_args()
m,records,data,js=read_archive(base,'af05221123efd8afe0e3aa237cf0aed5a16e26ed2b2139ca1099d9d57bfcbc9c')
assert m['accepted_reproducer_matrix'] and not any(m[n] for n in ['accepted_component','whole_release','full_five_routes'])
q=js('reduction/result.json');d=js('reduction/localization/result.json')
assert q['terminal'] and q['all_guards_match'] and q['ordinary_seconds']==10 and len(q['commands'])==24
assert all(v is None for v in q['environment_before'].values())
for group in ['source','tools']:
 assert q[group+'_before']==q[group+'_after']
 for n,h in q[group+'_before'].items():assert records['reduction/'+n if group=='source' else 'external/'+n.lstrip('/')]['sha256']==h
bad={'nested-c-translation','nested-genuine-translation','recursive-c-translation','recursive-genuine-translation'}
assert {c['label'] for c in q['commands'] if c['exit']}==bad and set(q['refusing_translations'])==bad
probes=0
for c in q['commands']:
 label=c['label'];assert c['exit']==(1 if label in bad else 0) and not c.get('timeout_expired') and c['timeout']==(60 if label.endswith('-probe') else 600)
 if label in bad:assert b'I require the matching scalar shape at this array write' in data('reduction/'+label+'.stderr')
 if label.endswith('-probe'):assert c['probe_matches'] and data('reduction/'+label+'.stdout')==b'7\n';probes+=1
assert probes==9
library=data('production/affine_analysis.nano');assert library==subprocess.check_output(['git','show',m['source_commit']+':src_nano/nvm2c/affine_analysis.nano'],cwd=base)
a=library.index(b'fn put(');b=library.index(b'\nfn transferred(',a);helper=library[a:b];assert helper==data('reduction/production-put.nano')
assert data('reduction/imported/ops.nano')==helper.replace(b'fn put(',b'pub fn put(',1)
for variant in ['nested','recursive']:assert helper in data('reduction/'+variant+'/ops.nano')
assert d['terminal'] and d['read_only_debugger'] and not d['product_acceptance'] and len(d['traces'])==2
for row in d['traces']:
 producer=row['producer'];assert row['exit']==0 and row['before']==row['after'] and row['offset']==42
 for n,h in row['before'].items():assert records['external/'+n.lstrip('/')]['sha256']==h
 text=data('reduction/localization/'+producer+'.stdout').decode()
 for phrase in ['start = 42','val.kind = 3','arr.kind = 2','arr.origin = 0','facts->final = 0','b->canonical_records = false']:assert phrase in text
 functions=re.findall(r'^\.function (\S+)',data('reduction/localization/'+producer+'-module.stdout').decode(),re.M);assert functions[row['function']]==row['name'] and (row['name']=='put' or row['name'].endswith('_put'))
 at={r['offset']:r for r in js('reduction/localization/'+producer+'-decoded.json')}
 assert [(at[n]['name'],at[n]['operands']) for n in [33,36,39,42]]==[('LOAD_LOCAL',[0]),('LOAD_LOCAL',[1]),('LOAD_LOCAL',[2]),('ARR_SET',[])]
if options.extract:
 target=options.extract.resolve();target.mkdir(parents=True,exist_ok=False)
 old=str(Path(m['roots']['reduction'])/'nested').encode()
 for name in ['main.nano','ops.nano','rows.nano']:(target/name).write_bytes(data('reduction/nested/'+name).replace(old,str(target).encode()))
 print('I extract the nested case to '+str(target)+' and rewrite only its absolute import prefix.')
print(json.dumps(dict(terminal=True,accepted_reproducer_matrix=True,commands=24,probes=9,translation_refusals=4,recursion_required=False,accepted_component=False,whole_release=False)))
