"""I verify the isolated test-report formatter comparison at original limits."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'3d98ddb9bf7c4c28e8258a9bb8f10664edb0ee308ee275b7e75537e5204c0ce8');sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/benchmark-result.json');p=js('comparison/preparation.json');proof=js('comparison/admission-preparation.json')
assert m['private_experiment'] and m['component_only'] and not m['accepted_component'] and not m['full_five_routes'] and not m['whole_release']
assert q['terminal'] and q['accepted_experiment'] and q['all_guards_match'] and not q['accepted_component'] and not q['whole_release']
assert m['source_commit']==q['source_pin']==p['source_pin']=='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8'
root=Path(m['roots']['comparison'])
def mapped(name):
 path=Path(name)
 return 'comparison/'+str(path.relative_to(root)) if path.is_relative_to(root) else 'external/'+str(path).lstrip('/')
for group in ['source','artifacts','hosts','native_inputs','original_c_source']:
 assert q[group+'_before']==q[group+'_after']
 for n,h in q[group+'_before'].items():assert rows['comparison/source/'+n if group=='source' else mapped(n)]['sha256']==h
assert q['observer_before']==q['observer_after'] and len(q['observer_before'])==1
assert next(iter(q['observer_before'].values()))==proof['script_sha256']==proof['script_after']==rows['comparison/benchmark.py']['sha256']
assert q['source_before']==p['source_after']==p['source_before'] and p['terminal'] and p['all_guards_match'] and not p['executed_benchmark']
assert not q['other_jobs'] and q['filesystem']!='nfs' and q['free_bytes']>15*1024**3 and q['available_memory']>4*1024**3
assert len(q['cpu_rounds'])==2 and all(c['share']>=.75 for c in q['cpu_rounds'])
assert q['provider']['before_translation'] and q['provider']['read_only'] and q['provider']['libraries_before']==q['provider']['libraries_after']
for n,h in q['provider']['libraries_before'].items():assert rows[mapped(n)]['sha256']==h
assert rows['external/usr/bin/cc']['sha256']==q['cc_sha256']
assert len(q['commands'])==20 and all(c['exit']==0 and 'interrupted_or_timeout' not in c for c in q['commands'])
assert q['commands'][0]['label']=='foundation-reader' and q['commands'][0]['timeout']==180
assert b'19802 commands' in data('comparison/foundation-reader.stdout') and b'257934 complete reports' in data('comparison/foundation-reader.stdout')
for label in ['translate','native-build']:assert next(c for c in q['commands'] if c['label']==label)['timeout']==600
builds=js('comparison/original-builds.json');native=next(c for c in q['commands'] if c['label']=='native-build')['argv']
assert native==q['native_commands']['candidate'] and q['native_commands']['original']==builds['native']['argv']
assert '-std=c11' in native and '-O1' in native and '-fsanitize=address,undefined' in native and '-fno-sanitize-recover=all' in native
oldwork=str(Path(builds['native']['argv'][builds['native']['argv'].index('-o')+1]).parent)
assert native==[str(root/'candidate.c') if arg==oldwork+'/decoder.c' else str(root/'candidate-native') if arg==oldwork+'/decoder-aot' else arg for arg in builds['native']['argv']]
cases=js('comparison/benchmark-cases.json');assert len(cases)==9 and len({c['name'] for c in cases})==9
assert cases[0]['name']=='load-4-True-True-256' and [c['name'] for c in cases[1:]]==['exclusion-'+str(n) for n in [0,63,64,127,128,191,192,255]]
reports=0
for round_ in range(2):
 for product in ['old-vm','new-vm','old-native','new-native']:
  for group in ['locals','original']:
   label='round-%d-%s-%s'%(round_,product,group);command=next(c for c in q['commands'] if c['label']==label);selected=[c for c in cases if c['group']==group]
   assert command['timeout']==(60 if group=='locals' else 600) and len(selected)==(1 if group=='locals' else 8)
   expected=''.join(c['expected']+'\n' for c in selected).encode();assert data('comparison/'+label+'.stdout')==expected and data('comparison/'+label+'.stderr')==b''
   assert command['argv'][-len(selected):]==[c['input'] for c in selected]
   for c in selected:assert rows[mapped(c['input'])]['sha256']==c['input_sha256']
   reports+=len(selected)
assert reports==72
for family in ['vm','native']:
 old=[next(c['seconds'] for c in q['commands'] if c['label']=='round-%d-old-%s-locals'%(r,family)) for r in range(2)]
 new=[next(c['seconds'] for c in q['commands'] if c['label']=='round-%d-new-%s-locals'%(r,family)) for r in range(2)]
 assert old==q['timings']['old-'+family] and new==q['timings']['new-'+family] and all(n<o for n,o in zip(new,old)) and q['faster_in_both_pairs'][family]
old=data('comparison/old-driver.nano');new=data('comparison/source/tests/nvm2c_affine_locals_driver.nano')
left=b'        for arm in (range 0 256) { set text (+ text (flag (at item.excluded arm))) }';right=b'        set text (+ text (report_text.exclusions item.excluded))'
assert new==b'module "tests/nvm2c_affine_report.nano" as report_text\n'+old.replace(left,right)
assert [line for line in new.splitlines() if b'assert ' in line]==[line for line in old.splitlines() if b'assert ' in line]
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
foundation='docs/evidence/darwin-current-source-20261124/diagnostics/affine-typed-five-linux-20261125/'
raw=subprocess.check_output(['git','show',q['foundation_evidence']['commit']+':'+foundation+'manifest.json'],cwd=repo);fm=json.loads(raw)
assert sha(raw)==q['foundation_evidence']['manifest_sha256'] and fm['source_commit']==q['source_pin'] and fm['accepted_component'] and fm['full_five_routes']
assert next(r for r in fm['files'] if r['path']=='comparison/cohort-result.json')['sha256']==q['foundation_evidence']['result_sha256']
# I bind the exact prepared overlay/controller to the previously published receipt.
prep='docs/evidence/darwin-current-source-20261124/diagnostics/affine-report-preparation-linux-20261125/'
pm=json.loads(subprocess.check_output(['git','show','b8cf8018ecf194c03434df4390ac0bb589f278b1:'+prep+'manifest.json'],cwd=repo))
for n in ['benchmark.py','candidate.nvm','preparation.json']:
 assert next(r for r in pm['files'] if r['path']=='preparation/'+n)['sha256']==rows['comparison/'+n]['sha256']
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in q['source_before'].items():
 if name=='tests/nvm2c_affine_report.nano':continue
 if name=='tests/nvm2c_affine_locals_driver.nano':h=sha(old)
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; published complete-foundation/idle admission, exact formatter-only delta and immutable source/tool/host/provider/artifact guards, all20 commands and72 complete old/new VM/sanitized-native C reports at original limits, two counterbalanced faster pairs per family. Test-report case only; no driver integration/full-corpus/current-source/whole-release or compiler-wide gain acceptance.'%len(rows))
