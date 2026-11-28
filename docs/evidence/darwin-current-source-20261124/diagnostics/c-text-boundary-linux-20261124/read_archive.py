"""I verify actual C-token round trips, retained failures and the C-keyword witness."""
from pathlib import Path
import sys,json,hashlib,subprocess,struct
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'00aff789d9b818708ad4dc84687cfec3a07c254a69943e47673ed22736742342')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['component_only'] and m['execution_routes']==5
assert not any(m[n] for n in ['new_bootstrap','current_stage2','whole_release','full_translator','cutover','keyword_repair'])
cm,cr,cd,cj=read_archive(base.parent/'shadow-entry-cold-bootstrap-linux-20261124','b3c85f73568c2c21784de9df63b80b68f9df1e296abdafab91e983de3559ae15')
cold=cj('bootstrap/bootstrap-result.json');assert cold['terminal'] and cold['original_make_pass']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(pin,name):
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();h=proc.stdout.readline().split();assert len(h)==3 and h[1]==b'blob',(pin,name)
 raw=proc.stdout.read(int(h[2]));assert proc.stdout.read(1)==b'\n';return raw
for label in ['first-probe','first-routes','second-routes','final-routes']:
 p=js(label+'/result.json');assert p['terminal'] and p['probe_pass'] and p['all_guards_match']
 for group in ['source','images','tools']:assert p[group+'_before']==p[group+'_after']
 for name,h in p['source_before'].items():
  assert rows[label+'/source/'+name]['sha256']==h
  assert h==(p['overlays'][name] if name in p['overlays'] else sha(blob(p['source_base'],name)))
 for name,h in p['images_before'].items():assert rows[label+'/source/'+name]['sha256']==h==cr['bootstrap/source/'+name]['sha256']
 for name,h in p['tools_before'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
 for name,h in cold['source_after'].items():
  if name.endswith(('.c','.h','.inc')):assert p['source_before'][name]==h
 for c in p['commands']:assert c['exit']==0 and not c.get('timed_out')
 if label=='first-probe':continue
 q=js(label+'/qualification-result.json');assert q['terminal'] and q['all_guards_match']
 assert q['producer_source']=='16b081fdf472dfe233ebf5975f149f6aa9d33783' and not q['new_bootstrap'] and not q['current_stage2'] and not q['whole_release']
 assert q['producer_sha256']==q['producer_after']==cr['bootstrap/source/bin/nanoc_stage2']['sha256']==rows[label+'/source/bin/nanoc_c_text_producer']['sha256']
 for group in ['source','images','tools','hosts']:
  assert q[group+'_before']==q[group+'_after']
  for name,h in q[group+'_before'].items():
   key=label+'/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')
   assert rows[key]['sha256']==h
 for c in q['commands']:assert c['terminal'] and not c.get('observed_timeout')
 if label=='first-routes':
  assert q['tests_run']==0 and not q['accepted'] and not q['failures'] and len(q['errors'])==1
  assert b"before \xe2\x80\x98signed\xe2\x80\x99" in data(label+'/qualification.stderr') or b"before 'signed'" in data(label+'/qualification.stderr')
 elif label=='second-routes':
  assert q['tests_run']==2 and not q['accepted'] and len(q['failures'])==5 and not q['errors']
  assert b"First differing element" in data(label+'/qualify.py.stderr')
 else:
  assert q['accepted'] and q['tests_run']==2 and not q['failures'] and not q['errors'] and not q['skipped']
  assert b'Ran 2 tests' in data(label+'/qualify.py.stderr') and data(label+'/qualify.py.stderr').rstrip().endswith(b'OK')
  for name,h in q['overlays'].items():assert sha(blob(m['source_commit'],name))==h
final=js('final-routes/qualification-result.json')
preflight=js('final-routes/preflight-result.json');assert preflight['terminal'] and preflight['accepted'] and len(preflight['commands'])==2 and all(c['exit']==0 for c in preflight['commands'])
assert b'0 reserved names' in data('final-routes/fixture-corpus.stdout')
snapshots=final['snapshots'];assert len(snapshots)==1
retained='final-routes/retained/'+Path(snapshots[0]['retained']).name+'/'
raw_cases=[b'',bytes(range(256)),b'\x00\xff"\\\n\r\t?/?=/* */123\x80\xc0\xaf','I preserve λ and 日本語 as bytes'.encode()]
for n in [1,2,3,254,255,256,257,511,512,513,4095,4096,4097,65536]:raw_cases.append(bytes((i*73+19)%256 for i in range(n)))
for i,b in enumerate(raw_cases):assert data(retained+'bytes-%d.bin'%i)==b
assert rows[retained+'oversized.bin']['bytes']==16777217 and data(retained+'oversized.bin')==bytes(16777217)
halves=[-1,0,1,2147483647,2147483648,4294967295,4294967296]
expected=bytearray()
for low in halves:
 if 0<=low<1<<32:expected+=struct.pack('<I',low)
 for high in halves:
  if 0<=low<1<<32 and 0<=high<1<<32:expected+=struct.pack('<Q',high<<32|low)
 for kind in range(6):
  if 0<=low<1<<32:expected+=struct.pack('<Q',kind<<32|low)
for value in [-(1<<63),-(1<<63)+1,-(1<<31),-1,0,1,(1<<31)-1,(1<<63)-1]:expected+=struct.pack('<q',value)
expected+=bytes(range(256))
for b in raw_cases:expected+=struct.pack('<I',len(b))+b
roundtrips=[];oversized=[];producers=set();sanitized=[]
for i,c in enumerate(final['commands']):
 executable=Path(c['argv'][0]).name
 stdout=data('final-routes/processes/%04d/stdout'%i);stderr=data('final-routes/processes/%04d/stderr'%i)
 assert sha(stdout)==c['stdout_sha256'] and sha(stderr)==c['stderr_sha256']
 if executable.startswith('roundtrip-'):
  assert c['exit']==0 and stdout==expected and not stderr and c['communicate_timeouts']==[60]
  roundtrips.append(executable)
 if c['argv'][-1].endswith('/oversized.bin'):
  assert c['exit']==2 and stdout.splitlines()[-1]==b'B|0|' and not stderr and c['communicate_timeouts']==[60];oversized.append(executable)
 if executable in ['nanoc_c','nano_virt','nanoc_c_text_producer']:producers.add(executable)
 if executable=='cc' and any(Path(a).name.startswith('roundtrip-') and a.endswith('.c') for a in c['argv']):
  assert all(a in c['argv'] for a in ['-std=c11','-O2','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fno-sanitize-recover=all']) and c['communicate_timeouts']==[600]
 if '-fsanitize=address,undefined' in c['argv']:sanitized.append(c)
assert set(roundtrips)=={'roundtrip-'+s for s in ['c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native']}
assert len(roundtrips)==len(oversized)==5 and producers=={'nanoc_c','nano_virt','nanoc_c_text_producer'} and len(sanitized)==2
keyword=js('keyword-probe/result.json');assert keyword['terminal'] and keyword['accepted_diagnostic'] and keyword['all_guards_match'] and not keyword['product_repair'] and not keyword['whole_release']
for group in ['source','images','tools']:
 assert keyword[group+'_before']==keyword[group+'_after']
 for name,h in keyword[group+'_before'].items():assert rows[('external/'+name.lstrip('/')) if group=='tools' else 'first-routes/source/'+name]['sha256']==h
commands={c['label']:c for c in keyword['commands']};assert len(commands)==len(keyword['commands'])
for name,v in keyword['cases'].items():
 for suffix,key in [('.nano','source_sha256'),('.c','c_sha256'),('.nvm','module_sha256')]:assert rows['keyword-probe/'+name+suffix]['sha256']==v[key]
 for suffix in ['emit-c','emit-nvm','verify','vm']:assert commands[name+'-'+suffix]['exit']==0 and not commands[name+'-'+suffix].get('timed_out')
 if name=='control':assert v['syntax_exit']==v['native_exit']==commands[name+'-execute']['exit']==0
 else:
  assert v['syntax_exit']!=0 and v['native_exit']!=0 and v['prior_preserved'] and data('keyword-probe/'+name+'.native')==b'prior-value'
  assert b'signed' in data('keyword-probe/'+name+'-syntax.stderr')
assert set(keyword['cases'])=={'control','local','parameter','shadow'}
assert b'extern fn' not in blob(m['source_commit'],'src_nano/nvm2c/c_text.nano')
proc.stdin.close();assert proc.wait()==0
print('PASS: %d records; strict-C output %d bytes SHA256 %s matches across five routes; oversized refusal5/5; two retained terminals and C-keyword defect remain explicit. No full translator, current-stage, paired, whole-runtime sanitizer, cutover or release acceptance.'%(len(rows),len(expected),sha(expected)))
