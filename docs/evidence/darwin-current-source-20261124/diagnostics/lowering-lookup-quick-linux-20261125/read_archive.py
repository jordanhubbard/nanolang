"""I verify exact lookup projections, complete reports and retained effect branches."""
from pathlib import Path
import sys,json,hashlib,subprocess,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'54fcdcd050fb74fb6e870a2d6c9c17348d1c26978001f73d19e0d15239310d13');sha=lambda raw:hashlib.sha256(raw).hexdigest();s=js('comparison/result.json');q=js('comparison/quick-result.json');a=js('comparison/admission-result.json');inspection=js('comparison/generated-c-inspection.json')
assert m['component_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['working_overlay'] and s['ordinary_seconds']==10 and not s['new_bootstrap'] and not s['stage_alias']
assert a['accepted'] and a['terminal'] and a['filesystem']!='nfs' and not a['other_jobs'] and a['free_bytes']>15*1024**3 and all(x['cpu']>=1 and x['share']>=.75 for x in s['cpu_rounds'])
assert q['terminal'] and q['accepted_c_native_vm_boundary'] and q['all_guards_match'] and q['ordinary_seconds']==10 and len(q['commands'])==108 and len(q['cases'])==24
for group in ['source','images','tools','hosts']:
 assert s[group+'_before']==s[group+'_after']==q[group+'_before']==q[group+'_after']
 for name,h in s[group+'_after'].items():assert rows['comparison/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
for name,h in q['inputs_before'].items():assert q['inputs_after'][name]==rows['comparison/source/'+name]['sha256']==h
for command in q['commands']:assert command['exit']==0 and not command.get('timed_out') and command['timeout'] in [600,60]
for axis in ['baseline','candidate']:
 for offset in [2,5]:assert data('comparison/quick-%03d.stdout'%(offset if axis=='baseline' else offset+6))==b'LOOKUP-PROJECTION-OK\n'
 expected=json.loads(data('candidate-inputs/tests/nanoisa_lookup_projection_cases.json'))
 assert len(expected)==24
 for i,case in enumerate(q['cases']):
  assert case['source_sha256']==expected[i]['source_sha256'] and case['expected_sha256']==expected[i]['report_sha256']
  assert rows['comparison/diagnostics/case-%02d.nano'%i]['sha256']==case['source_sha256'] and rows['comparison/diagnostics/case-%02d.expected'%i]['sha256']==case['expected_sha256']
  assert len(case['commands'])==4
  for n in case['commands']:assert rows['comparison/quick-%03d.stdout'%n]['sha256']==case['expected_sha256']
 c=data('comparison/products/'+axis+'.c').decode()
 for name,row in inspection[axis].items():
  match=re.search(r'static int64_t (nl_[a-zA-Z0-9_]*'+name+r')\([^;\n]*\) \{',c);assert match,name
  raw=c[match.start():c.index('\n}\n',match.start())+3].encode();assert raw==data('comparison/'+axis+'-'+name+'.c') and sha(raw)==row['sha256']
  assert re.findall(rb'calloc\((\d+), sizeof \*rl\)',raw)==[str(v).encode() for v in row['locals']]
  assert re.findall(rb'nrec_t \*r = (\d+) \?',raw)==[str(v).encode() for v in row['temporary_records']]
  for field,function in [('parser_get_function_calls','parser_get_function'),('parser_get_count_calls','parser_get_function_count'),('parser_get_union_calls','parser_get_union'),('parser_get_union_count_calls','parser_get_union_count')]:assert len(re.findall(rb'nl_[a-zA-Z0-9_]*'+function.encode()+rb'\(',raw))==row[field]
  if name=='nisa_fn_index':assert b'fflush(stdout);' in raw and row['empty_print_flush']
for name,old,new in [('nisa_fn_index','2','1'),('nisa_identifier_function','3','2'),('nisa_call_index','3','2')]:assert inspection['baseline'][name]['locals']==[old] and inspection['candidate'][name]['locals']==[new]
for name in inspection['candidate']:assert inspection['candidate'][name]['parser_get_function_calls']==inspection['candidate'][name]['parser_get_union_calls']==0
assert inspection['baseline']['nisa_union_index']['temporary_records']==['6'] and inspection['candidate']['nisa_union_index']['temporary_records']==['5']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(pin,name):
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in s['source_before'].items():assert sha(blob(m['source_commit'] if name in s['overlays'] else s['source_base'],name))==h,name
for name in ['tests/test_nanoisa_lookup_projection.py','tests/nanoisa_lookup_projection_cases.json']:assert blob(m['source_commit'],name)==data('candidate-inputs/'+name)
old=blob(s['source_base'],'src_nano/compiler/nanoisa_codegen.nano').decode().splitlines();new=blob(m['source_commit'],'src_nano/compiler/nanoisa_codegen.nano').decode().splitlines();old=[line for line in old if 'assert ' in line];new=[line for line in new if 'assert ' in line];assert len(old)==1636 and len(new)==1646;it=iter(new);assert all(any(value==line for value in it) for line in old)
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: exact baseline/candidate source,108 C-native+VM commands,24 complete raw reports across both,1636 prior+10 new assertions, projected C locals and preserved print/flush. Full five routes/permanent invocation/compiler/factory/bootstrap/paired/cutover/release and gain remain unclaimed.'%len(rows))
