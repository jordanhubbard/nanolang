"""I verify the actual packaged-interpreter link repair and retained original test."""
from pathlib import Path
import sys,json,hashlib,subprocess,struct
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'1af280889d8bbca8a6f5b929dbcb47fb8979382e230c4e71792a6f3f077af212');sha=lambda raw:hashlib.sha256(raw).hexdigest();s=js('qualification/setup-result.json');q=js('qualification/focused-result.json');r=js('red-green/result.json');old=js('baseline/setup-result.json')
assert m['source_commit']==s['source_pin']=='95a4062efebe0e97d8cde271117eac1deab21595' and m['wrapper_link_repaired'] and not any(m[k] for k in ['carrier_lifetime_acceptance','new_bootstrap','whole_release'])
assert s['terminal'] and s['admitted_setup'] and s['cold_tools'] and not s['working_overlay'] and not s['overlays'] and not s['provider_copies'] and s['ordinary_shadow_seconds']==10
assert s['source_before']==s['source_after'] and s['tools']==s['tools_after'] and s['filesystem']!='nfs' and not s['other_nanolang_jobs'] and all(x['cpu_seconds']>=1 and x['share']>=.75 for x in s['cpu_rounds'])
assert q['terminal'] and q['passed'] and q['all_guards_match'] and q['tests']==q['discovered']==15 and q['failures']==q['errors']==q['skips']==0
for group,values in [('source',s['source_after']),('images',s['qualified_images']),('tools',s['tools_after'])]:
 assert q[group+'_before']==q[group+'_after']==values
 for name,h in values.items():assert rows['qualification/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert old['terminal'] and old['admitted_setup'] and old['flags']==s['flags']
diff={n:dict(baseline=h,candidate=s['source_after'].get(n)) for n,h in old['source_after'].items() if n.endswith(('.c','.h','.inc')) and h!=s['source_after'].get(n)};assert diff==js('red-green/c-source-diff.json') and list(diff)==['src/nanovirt/wrapper_gen.c']
assert r['terminal'] and r['passed_red_green'] and not r['carrier_lifetime_acceptance'] and r['test_sha256']==rows['qualification/source/tests/test_native_effect_execution.py']['sha256']
assert r['axes']['baseline']['tests']==1 and r['axes']['baseline']['failures']==1 and not r['axes']['baseline']['passed'] and r['axes']['candidate']['tests']==1 and r['axes']['candidate']['passed'] and r['axes']['candidate']['failures']==0
for axis in ['baseline','candidate']:
 assert r['axes'][axis]['errors']==r['axes'][axis]['skips']==0 and r['axes'][axis]['objects']==r['axes'][axis]['objects_after']
 for name in ['obj/eval/carrier_heap.o','obj/eval/carrier_values.o','obj/eval/carrier_runtime.o','obj/nanovirt/wrapper_gen.o']:
  assert rows[('baseline/' if axis=='baseline' else 'qualification/source/')+name]['sha256']==r['axes'][axis]['objects'][name]
assert r['axes']['baseline']['tool_sha256']==rows['baseline/nano_virt']['sha256']==old['qualified_images']['bin/nano_virt']
assert r['axes']['candidate']['tool_sha256']==s['qualified_images']['bin/nano_virt']
assert [(x['axis'],x['exit'],x['timeout']) for x in r['commands']]==[('baseline',1,60),('candidate',0,60),('candidate',0,10),('embedded',0,60),('embedded',0,10),('embedded',0,60)]
assert b'undefined reference' in data('red-green/command-1.stderr') and b'env_carrier_' in data('red-green/command-1.stderr')
raw=data('red-green/candidate/wrapper');assert sha(raw)==r['packaged_wrapper_sha256'] and raw[:6]==b'\x7fELF\x02\x01';offset=struct.unpack_from('<Q',raw,40)[0];size,count=struct.unpack_from('<HH',raw,58);sections=[struct.unpack_from('<IIQQQQIIQQ',raw,offset+i*size) for i in range(count)];blobs=[];symbols=set()
for section in sections:
 if section[1]!=2:continue
 strings=sections[section[6]];names=raw[strings[4]:strings[4]+strings[5]]
 for at in range(section[4],section[4]+section[5],section[9]):
  name,info,other,index,value,length=struct.unpack_from('<IBBHQQ',raw,at);spelling=names[name:names.find(b'\0',name)]
  if index:symbols.add(spelling)
  if spelling==b'nvm_blob':
   sec=sections[index];start=sec[4]+value-sec[3];blobs.append(raw[start:start+length])
assert len(blobs)==1 and blobs[0]==data('red-green/candidate/embedded.nvm') and sha(blobs[0])==r['embedded_module_sha256'] and blobs[0][:4]==b'NVM\x02'
for name in [b'env_carrier_enter',b'env_carrier_track',b'nl_carrier_value',b'nl_carrier_values_reclaimed_arrays',b'nl_carrier_heap_new']:assert name in symbols and name in data('red-green/command-6.stdout')
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';value=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(value)==h,name
proc.stdin.close();assert proc.wait()==0
source=data('qualification/source/src/nanovirt/wrapper_gen.c');addition=b'        "eval/carrier_heap.o", "eval/carrier_values.o", "eval/carrier_runtime.o",\n';assert source.count(addition)==1 and sha(source.replace(addition,b'',1))==diff['src/nanovirt/wrapper_gen.c']['baseline']
print('PASS %d records: fresh strict tools and15 unchanged tests, original wrapper baseline red/candidate green at60/10, four carrier/wrapper objects, actual ELF symbols and extracted NVM verify/execute. No carrier-lifetime, bootstrap, AOT/cutover, paired or release acceptance.'%len(rows))
