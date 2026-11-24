"""I verify three Darwin profiles and retain the pre-shadow source timeout."""
from pathlib import Path
import sys,json,gzip,hashlib,subprocess
base=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
manifest=json.loads((base/'manifest.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();rows={r['path']:r for r in manifest['files']};assert len(rows)==len(manifest['files'])
for digest,chunk in manifest['chunks'].items():
 path=base/'objects'/(digest+'.gz');zipped=path.read_bytes();assert len(zipped)==chunk['compressed_bytes'] and sha(zipped)==chunk['compressed_sha256'];raw=gzip.decompress(zipped);assert len(raw)==chunk['bytes'] and sha(raw)==digest
assert {p.name for p in (base/'objects').glob('*.gz')}=={h+'.gz' for h in manifest['chunks']}
def blocks(row):
 for h in row['chunks']:yield gzip.decompress((base/'objects'/(h+'.gz')).read_bytes())
images=set()
for row in rows.values():
 d=hashlib.sha256();size=0;head=b''
 for raw in blocks(row):
  d.update(raw);size+=len(raw)
  if not head:head=raw[:64]
 assert size==row['bytes'] and d.hexdigest()==row['sha256'],row['path']
 if head[:4]==b'\xcf\xfa\xed\xfe' and int.from_bytes(head[4:8],'little')==0x100000c and int.from_bytes(head[12:16],'little') in (2,6,8):images.add(row['sha256'])
def data(name):return b''.join(blocks(rows[name]))
def js(name):return json.loads(data('qualification/'+name))
q=js('result.json');cfg=js('config.json');pre=js('preflight.json');root=Path(manifest['qualification_root'])
def native_name(path):return 'qualification/'+str(Path(path).relative_to(root))
def image(path):
 row=rows[native_name(path)];assert row['sha256'] in images;return row
assert q['terminal'] and q['controller_exit']==manifest['controller_exit']==2
assert q['source_pin']==cfg['pin']==manifest['source_pin']
assert q['tools_before']==q['tools_at_terminal']
assert not q['ordinary_ownership_enabled'] and not q['raw_bootstrap'] and not q['whole_release']
assert pre['admitted'] and pre['system']=='Darwin' and pre['machine']=='arm64' and not pre['native']
for observation in [pre]+q['health']:
 assert len(observation['rounds'])==3
 for r in observation['rounds']:assert r['cpu']>=1 and r['share']>=.75
for name,v in manifest['source_guards'].items():assert v['expected']==v['actual']==cfg['sources'][name]
for name,v in manifest['tool_guards'].items():
 assert v['actual']==v['expected']==q['tools_before'][name]
 assert rows['tool/'+name.lstrip('/')]['sha256']==v['actual']
repo=Path(subprocess.check_output(['git','-C',str(Path(__file__).resolve().parent),'rev-parse','--show-toplevel'],text=True).strip())
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,expected in cfg['sources'].items():
 proc.stdin.write((cfg['pin']+':'+name+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert head[1]==b'blob';raw=proc.stdout.read(int(head[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==expected
 for p in q['profiles']:assert sha(data(native_name(Path(p['tree'])/name)))==expected
proc.stdin.close();assert proc.wait()==0 and len(cfg['sources'])==3520
assert [p['mode'] for p in q['profiles']]==['normal','san','production']
for p in q['profiles']:
 assert not any(p['initial_outputs'].values())
 assert p['before']==p['after_focused']==p['after_private_repl']==cfg['sources']
 assert p['base']==['/usr/bin/make','-j1','-f','Makefile.gnu','--no-print-directory','CC=/usr/bin/cc','CFLAGS='+' '.join(cfg['flags'][p['mode']])]
 assert p['build']['exit']==p['focused']['exit']==0 and p['build']['timeout']==1800 and p['focused']['timeout']==600
 assert p['build']['args'][-4:]==['bin/nanoc_c','bin/nano_vm','bin/nvm2c','bin/nanoisa']
 assert image(Path(p['tree'])/'bin/nanoc_c')['sha256']==p['image_sha256']
 for name in ['nano_vm','nvm2c','nanoisa']:image(Path(p['tree'])/'bin'/name)
 r=p['private_repl'];assert (r['tests'],r['failures'],r['errors'],r['skips'],r['calls'])==(41,0,0,1,58)
 assert image(p['private_repl_image']['path'])['sha256']==p['private_repl_image']['sha256']
 assert b'Linux descriptor fault observer' in data('qualification/'+p['mode']+'-private-repl-cases/tests.log')
 calls=js(p['mode']+'-private-repl-cases/calls.json');assert len(calls)==58 and all(r['exit']==0 and r['timeout']==30 for r in calls)
assert q['tool_controls']==dict(tests=1,failures=0,errors=0,skips=0,calls=20)
assert len(js('tool-cases/calls.json'))==20
assert all(c['exit']==0 and not c['timed_out'] for c in q['commands'][:-1])
assert q['commands'][-1]['label']=='ordinary-source'
assert [s['name'] for s in q['source_stages']]==['ordinary']
stage=q['source_stages'][0];command=stage['command']
assert stage['after']==cfg['sources'] and command['exit']==-9 and command['timed_out'] and command['timeout']==600
assert 600<=command['seconds']<610 and command['user_seconds']>0 and command['system_seconds']>0
assert stage['starts']==stage['completions']==0 and not stage['failed_shadows'] and not stage['phases']
events=[]
for line in data('qualification/ordinary-source.stderr').splitlines():
 try:events.append(json.loads(line))
 except ValueError:pass
assert not any(e.get('event') in ['shadow-start','shadow-complete','cseed-shadow-phase'] for e in events)
assert native_name(root/'ordinary-source/seed') not in rows
assert not any(c['label']=='private-source' for c in q['commands'])
changes=subprocess.check_output(['git','diff','--name-only','8949313b0a56c0c3bd52509fb363edb4eefba3e4',cfg['pin'],'--','src','src_nano','tests','scripts','modules','std','stdlib','Makefile.gnu'],cwd=repo,text=True).splitlines()
assert changes==['tests/test_module_rejection_rollback.c']
print('PASS integrity: %d records, %d selected Git sources, %d native ARM64 images; three profiles/private41 pass, ordinary source FAIL600 before shadows, private source NOT RUN.'%(len(rows),len(cfg['sources']),len(images)))
