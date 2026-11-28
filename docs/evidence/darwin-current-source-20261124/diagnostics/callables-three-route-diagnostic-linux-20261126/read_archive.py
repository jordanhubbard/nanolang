"""I verify diagnostic evidence without granting five-route/native acceptance."""
from pathlib import Path
import ast,hashlib,json,random,re,struct,sys,yaml
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda raw:hashlib.sha256(raw).hexdigest()
m,rows,data,js=read_archive(base,'de324b36e969f672e9fe49074b4166b16284c789a102415191d40c60b9a8694b')
pm,pr,pd,pj=read_archive(base.parent/'effect-flow-five-linux-20261126','36adf522f9ad9c54bad67528ed0aa0736db6eb897c837abe9df1a8ad06a22918')
bm,br,bd,bj=read_archive(base.parent/'typechecker-vector-prefix-cold-bootstrap-linux-20261126','3c2f440e292cb2c0c54f8b3d344b239c927223bed505c874e90831b842f1c5d7')
q=js('qualified/qualification-result.json');old=pj('qualified/qualification-result.json')
assert m['component_only'] and m['pure_nanolang'] and m['diagnostic_only']
assert not any(m[k] for k in ['full_five_routes','native_acceptance','component_accepted','new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert q['terminal'] and q['diagnostic_pass'] and q['tests_pass'] and q['all_guards_match']
assert q['diagnostic_only'] and q['acceptance_withheld'] and not q['accepted'] and not q['full_five_routes']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(3,0,0,0)
assert (q['ordinary_seconds'],q['build_seconds'],q['run_seconds'])==(10,600,60)
new={'src_nano/nvm2c/callables.nano','tests/nvm2c_callables_driver.nano','tests/nvm2c_callables_reference.c','tests/nvm2c_callables_original_trace.c','tests/test_nanolang_translator_callables.py'}
assert set(q['source_after'])==set(old['source_after'])|new and len(q['source_after'])==4002
for name,h in q['source_after'].items():
 if name not in new|{'Makefile.gnu'}:assert old['source_after'][name]==h
block=b'.PHONY: test-nanolang-translator-callables\ntest-nanolang-translator-callables: $(COMPILER_C) nano_virt nano_vm nvm2c nvm2c-runtime $(NANOISA_OBJECTS) $(NANOISA_UTF8)\n\tpython3 -m unittest -v tests.test_nanolang_translator_callables\n\ntest-units: test-nanolang-translator-callables\n\n'
make=data('qualified/source/Makefile.gnu');assert make.count(block)==1 and make.replace(block,b'',1)==pd('qualified/source/Makefile.gnu')
assert len(q['images_after'])==49
for name,h in old['images_after'].items():assert q['images_after'][name]==h
for name,h in q['images_after'].items():
 if name not in old['images_after']:assert br['bootstrap/source/'+name]['sha256']==h
labels=['qualified','pre-expanded','failed-expansion','failed-native','failed-parse']
for label in labels:
 state=js(label+'/qualification-result.json');preparation=js(label+'/preparation.json')
 assert state['terminal'] and not state['accepted'] and state['all_guards_match']
 assert state['source_before']==state['source_after']==preparation['source_before']
 assert not state['owned_cleanup']['remaining'] and not state['owned_cleanup']['records']
 for name,h in state['source_after'].items():assert rows[label+'/source/'+name]['sha256']==h
 assert state['images_before']==state['images_after']==q['images_after']
 for name,h in state['images_after'].items():assert rows[label+'/source/'+name]['sha256']==h
 assert state['host_tools_before']==state['host_tools_after']==old['host_tools_after'] and len(state['host_tools_after'])==22
 assert state['producer_before']==state['producer_after']==old['producer_after']=='f28b2bf49e94b5033e6b579b07f5663f0fc7151f1a933ab09efef53e0e13cd74'
 assert state['producer_hosts_before']==state['producer_hosts_after']==old['producer_hosts_after'] and len(state['producer_hosts_after'])==6
 assert state['producer_receipt_before']==state['producer_receipt_after']==old['producer_receipt_after']
 assert not state['other_jobs'] and not state['unexpected_sanitizer'] and state['filesystem']!='nfs' and state['free_bytes']>15*1024**3
 assert all(r['cpu']>=1 and r['share']>=.75 for r in state['cpu_rounds'])
 for n,c in enumerate(state['commands'],1):
  assert c['number']==n and c['streams_complete'] and c['effective_budget']==(600 if c['build'] else 60) and c['timeout_requested']==60
  assert not c['cleanup']['remaining'] and not c['cleanup']['records']
  for stream in ['stdout','stderr']:assert rows[label+'/command-%03d.%s'%(n,stream)]['sha256']==c[stream+'_sha256']
  for item in c['artifacts']:
   if item.get('kind')!='symlink':assert rows[label+'/retained/'+item['sha256']]['sha256']==item['sha256'] and rows[label+'/retained/'+item['sha256']]['bytes']==item['bytes']
 for name,h in q['source_after'].items():
  if name not in {'src_nano/nvm2c/callables.nano','tests/test_nanolang_translator_callables.py'}:assert state['source_after'][name]==h
namespace={};exec(compile(data('collector-source/replay.py'),'replay.py','exec'),namespace)
replays={}
for label in labels:
 replays[label]=namespace['replay'](lambda path,label=label:data(label+'/'+path.removeprefix('qualified/')))
 assert replays[label]['product_commands_executed']==0 and not replays[label]['component_accepted']
assert replays['qualified']['passed'] and replays['pre-expanded']['passed']
assert (replays['failed-native']['methods'],replays['failed-native']['errors'])==(0,1)
assert (replays['failed-parse']['methods'],replays['failed-parse']['errors'])==(0,1)
assert (replays['failed-expansion']['methods'],replays['failed-expansion']['failures'])==(3,1)
# I preserve all original C assertions and every captured module input.
def original_packets(label):
 state=js(label+'/qualification-result.json')
 command=next(c for c in state['commands'] if c['argv'][0].endswith('/original-callables'))
 assert b'505 passed, 0 failed' in data(label+'/command-%03d.stdout'%command['number'])
 item=next(a for a in command['artifacts'] if a['path'].endswith('/original.packets'))
 return data(label+'/retained/'+item['sha256'])
original=original_packets('qualified')
assert original==original_packets('pre-expanded')==original_packets('failed-expansion')
position=0;original_count=0
while position<len(original):
 length=struct.unpack_from('<I',original,position)[0];position+=4;assert length<=len(original)-position
 position+=length;original_count+=1
assert original_count>15
assert data('qualified/source/src_nano/nvm2c/callables.nano')==data('pre-expanded/source/src_nano/nvm2c/callables.nano')==data('failed-native/source/src_nano/nvm2c/callables.nano')
assert re.sub(rb'\bfrom\b',b'origin',data('failed-parse/source/src_nano/nvm2c/callables.nano'))==data('qualified/source/src_nano/nvm2c/callables.nano')
failed_test=data('failed-expansion/source/tests/test_nanolang_translator_callables.py')
repaired=failed_test.replace(b"('HANDLER_POP',)",b"('HANDLER_POP',1)").replace(b"row=OPS[name];assert len(args)==len(row['operands'])",b"row=OPS[name];assert len(args)==len(row['operands']),(name,args,row['operands'])")
assert repaired==data('qualified/source/tests/test_nanolang_translator_callables.py')
ops={v['name']:v for v in yaml.safe_load(data('qualified/source/spec/nanoisa.yaml'))['legacy_opcodes']}
def fixtures(raw):
 tree=ast.parse(raw);scope=dict(struct=struct,random=random,re=re,OPS=ops,FMT={'U8':'B','U16':'H','U32':'I','I32':'i','I64':'q','F64':'Q'})
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'pure-fixture-functions','exec'),scope)
 return scope['cases']()
before=fixtures(data('pre-expanded/source/tests/test_nanolang_translator_callables.py'))
after=fixtures(data('qualified/source/tests/test_nanolang_translator_callables.py'))
kept=[row for row in after if not row[0].startswith(('live-stack-opcode-','effect-'))]
assert len(kept)==len(before)
for oldrow,newrow in zip(before,kept):
 assert oldrow[1:]==newrow[1:]
 if not oldrow[0].startswith('metadata-'):assert oldrow[0]==newrow[0]
assert sum(row[0].startswith('effect-') for row in after)==6 and len(after)>len(before)
# I bind both read-only debugger observations to exact module instruction bytes.
guards=js('localization/guards.json');assert guards['read_only'] and guards['source_matches_failed_parent'] and guards['images_match_failed_parent']
assert guards['main_c_sha256']==q['source_after']['src/nanoisa/nvm2c.c']=='994d83e225343cfee19f5674cc30259fcea6bc3db44d65911764fd63489f31c0'
assert rows['external/usr/bin/gdb']['sha256']==guards['gdb_sha256']
source=data('qualified/source/src/nanoisa/nvm2c.c');assert b'#define NVM2C_VK_UNK 2' in source and b'#define NVM2C_VK_ARR 3' in source
by_code={v['code']:v for v in ops.values()};formats={'U8':'B','U16':'H','U32':'I','I32':'i','I64':'q','F64':'Q'}
for binding,trace,prefix,function,pc in zip(js('localization/instruction-bindings.json'),['trace','source-trace'],['','source-'],[14,22],[191,496]):
 raw=data('localization/'+binding['file']);assert sha(raw)==binding['sha256'] and raw[:4]==b'NVM\x02'
 directory,count=struct.unpack_from('<II',raw,20);sections={}
 for k in range(count):
  kind,flags,at,size=struct.unpack_from('<IIQQ',raw,directory+k*24);sections[kind]=raw[at:at+size]
 fields=struct.unpack_from('<IIQQHHHH',sections[5],4+function*32);assert list(fields)==binding['function_fields']
 code=sections[6][fields[2]:fields[2]+fields[3]];assert code[pc]==ops['ARR_SET']['code']
 for item in binding['instructions']:
  at=item['offset'];schema=by_code[code[at]];size=1;values=[]
  for kind in schema['operands']:values.append(struct.unpack_from('<'+formats[kind],code,at+size)[0]);size+=struct.calcsize('<'+formats[kind])
  assert item==dict(offset=at,name=schema['name'],operands=values,bytes=code[at:at+size].hex())
 text=data('localization/'+trace+'.stdout').decode()
 expected=[str(function),str(function),str(pc),'2','-1','3','3','0','false','0']
 for number,value in enumerate(expected,1):assert re.search(r'\$'+str(number)+r' = '+re.escape(value)+r'(?: |\n)',text)
 assert 'CALLABLE_ARRAY_WRITE_REFUSAL' in text and 'nvm2c.c:2891' in text and 'exited with code 01' in text
 state=js('localization/'+trace+'.json');assert state['exit']==0 and state['timeout']==60 and state['prior_output_preserved']
 assert data('localization/'+prefix+'prior.c')==b'prior output\n'
 assert b'I require the matching scalar shape at this array write' in data('localization/'+trace+'.stderr')
controls=js('collector-source/collector-controls.json');assert len(controls)==2
for control in controls:
 assert control['exit']==1 and not control['output_created'] and control['collector_sha256']==sha(data('collector-source/package.py'))
 assert b'AssertionError' in data('collector-source/'+control['label']+'-refusal.stderr')
control=js('nonterminal-control/qualification-result.json');expected=dict(q);expected['terminal']=False;assert control==expected
print(json.dumps(dict(evidence_verified=True,component_accepted=False,native_acceptance=False,full_five_routes=False,diagnostic_pass=True,
 records=len(rows),matrix=len(after),explicit_expectations=sum(row[2] is not None for row in after),original_modules=original_count,
 replays=replays,product_commands_executed=0)))
