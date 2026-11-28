"""I verify failed adoption, the owned-literal repair, and original source outcomes."""
from pathlib import Path
import sys,json,subprocess,hashlib
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'20a61ccd3741c31aaf2b154166fbee392359849f7e16cbf6acaf9885dbdaa4f0')
sha=lambda b:hashlib.sha256(b).hexdigest()
q=js('journal/cold-matrix-result.json');p=js('journal/private-result.json');source=js('journal/full-source/result.json')
assert not m['whole_release'] and not m['diagnostic_only'] and m['full_source_accepted']==source.get('accepted',False)
assert q['terminal'] and q['accepted'] and p['terminal'] and p['accepted'] and source['terminal']
assert m['source_pin']==q['pin']==source['source_pin'] and m['candidate']==q['candidate']==source['candidate']
assert q['tools']==q['tools_after']==source['tools_before']==source['tools_after']
for name,h in q['tools'].items():assert rows['tool/'+name.lstrip('/')]['sha256']==h
assert len(q['profiles'])==len(p['profiles'])==3
for profile,private in zip(q['profiles'],p['profiles']):
 mode=profile['mode'];prefix='journal/cold-source-'+mode+'/'
 assert profile['before']==profile['after']==private['before']==private['after'] and mode==private['mode']
 assert not profile['provider_copies'] and profile['build']['exit']==profile['focused']['exit']==0
 for name,h in profile['before'].items():assert rows[prefix+name]['sha256']==h
 for name,h in q['candidate'].items():assert profile['before'][name]==h
 assert rows[prefix+'bin/nanoc_c']['sha256']==profile['image_sha256']==rows['journal/cold-'+mode+'-before-focused-image']['sha256']
 for name in ['nanoc_c','nano_vm','nvm2c','nanoisa','nanolang_private']:
  head=data(prefix+'bin/'+name)[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
 assert rows[prefix+'bin/nanolang_private']['sha256']==private['image_sha256']
 t=js('journal/'+mode+'-private-cases/result.json');assert t==private['tests'] and t['tests']==41 and t['failures']==t['errors']==0
 assert len(js('journal/'+mode+'-private-cases/calls.json'))==62
 assert b'already-owned literals without duplicate registration' in data('journal/'+mode+'-private-bindings.stdout')
 assert all(c['exit']==0 for c in p['commands'] if c['label'].startswith(mode+'-'))
old=js('predecessor/private-result.json');assert old['terminal'] and not old.get('accepted',False)
t=js('predecessor/normal-private-cases/result.json');assert t['tests']==41 and t['failures']==2 and t['errors']==0
assert b'I cannot account for interpreter carrier ownership.' in data('predecessor/tracked-red-run.stderr')
red=js('predecessor/tracked-red-result.json');assert red[0]['exit']==0 and red[1]['exit']==1
assert b'fourteen literal/copy/borrowed boundaries' in data('predecessor/tracked-red-run.stdout')
assert all(c['exit']==0 for c in js('predecessor/owned-green-result.json'))
for stage in source['source_stages']:
 name=stage['name'];command=stage['command'];assert command['timeout']==600 and command['args'][1].endswith('/src_nano/nanoc_v06.nano')
 events=[]
 for line in data('journal/full-source/'+name+'-source.stderr').splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 starts=[e for e in events if e.get('event')=='shadow-start'];ends=[e for e in events if e.get('event')=='shadow-complete']
 assert len(starts)==stage['starts'] and len(ends)==stage['completions'] and not stage['failed_shadows'] and all(e.get('failures',0)==0 for e in ends)
 if command['exit']==0:
  assert len(starts)==len(ends)==1306 and not command['timed_out'];assert rows['journal/full-source/'+name+'-source/seed']['sha256']==stage['product']['sha256']
 else:assert 'journal/full-source/'+name+'-source/seed' not in rows
witness=js('journal/allocation-result.json');assert all(c['exit']==0 for c in witness['commands'])
assert witness['observation']==dict(literal_binding_copied=False,managed_object_delta=0,managed_allocated_delta=0,identifier_independent=True,borrowed_identity=True)
assert witness['source_sha256']==rows['journal/allocation-witness.c']['sha256']
assert [s['name'] for s in source['source_stages']] in [['ordinary'],['ordinary','private']]
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
changed=set(subprocess.check_output(['git','diff','--name-only',q['pin'],m['source_commit']],cwd=repo,text=True).splitlines());assert changed==set(q['candidate'])|{'docs/ROADMAP.md'}
def blob(spec):
 proc.stdin.write((spec+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert head[1]==b'blob',spec;raw=proc.stdout.read(int(head[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in q['profiles'][0]['before'].items():
 assert sha(blob((q['pin'] if name=='docs/ROADMAP.md' else m['source_commit'])+':'+name))==h
 if name not in q['candidate']:assert sha(blob(q['pin']+':'+name))==h
for name,h in q['candidate'].items():assert sha(blob(m['source_commit']+':'+name))==h
for name in ['src/eval/carrier_runtime.c','src/eval/carrier_values.c','src/eval/carrier_heap.c','src/runtime/gc.c']:
 assert blob(q['pin']+':'+name)==blob(m['source_commit']+':'+name)
proc.stdin.close();assert proc.wait()==0
print('PASS integrity: %d records; failed V1 ownership retained; corrected cold normal/SAN/O3 + private41 each pass. Full source outcomes: %s. Whole release acceptance remains open.'%(len(rows),[(s['name'],s['command']['exit'],s['starts'],s['completions']) for s in source['source_stages']]))
