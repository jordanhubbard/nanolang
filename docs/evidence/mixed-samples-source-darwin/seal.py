import pathlib as p,json,hashlib,shutil,subprocess,re
root=p.Path('/private/tmp/nanolang-mixed-source-70c511dad');out=p.Path('/private/tmp/nanolang-mixed-source-darwin-seal-70c511dad');assert not out.exists()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
bases={n:p.Path('/private/tmp/nanolang-mixed-source-darwin-'+n+'-70c511dad') for n in ('bootstrap','qualified','runtime')}
bases['adjacency-corrected']=p.Path('/private/tmp/nanolang-mixed-source-darwin-adjacency-corrected-70c511dad')
bases['selectors-corrected']=p.Path('/private/tmp/nanolang-mixed-source-darwin-selectors-corrected-70c511dad')
for n,b in bases.items():
 s=json.loads((b/'status.json').read_text());assert s['sources_unchanged'] and s['clean'],(n,s)
 if n=='qualified':assert s['status']==0
 elif n in ('runtime','adjacency-corrected'):assert [x['status'] for x in s['steps']]==[0,2]
 else:assert all(x['status']==0 for x in s['steps'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='70c511dad439f77eba15fe0bdc189e90ff9c9486'
out.mkdir();reports={}
for n,b in bases.items():
 for f in sorted(b.rglob('*')):
  if f.is_file() and f.suffix in ('.json','.log','.py','.mk') and 'native' not in f.relative_to(b).parts:
   rel=p.Path(n)/f.relative_to(b);dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);assert sha(f)==sha(dest);reports[str(rel)]=sha(dest)
for name in ('bootstrap-launch.log','sequence.log','sequence-status.json','sequence.py','tests.py','qualify.py','runtime.py','host-tools-before.json','xcode-resolved.json','adjacency-corrected-launch.log','selectors-corrected-launch.log','seal.py'):
 f=p.Path('/private/tmp/nanolang-mixed-source-darwin-'+name)
 if f.is_file():shutil.copyfile(f,out/name);reports[name]=sha(out/name)
before=json.loads(p.Path('/private/tmp/nanolang-mixed-source-darwin-host-tools-before.json').read_text());after={'SDKROOT':subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip(),'tools':{name:{'resolved':str(p.Path(name).resolve()),'sha256':sha(p.Path(name))} for name in before['tools']}};assert before==after
f=out/'host-tools-after.json';f.write_text(json.dumps(after,indent=2)+'\n');reports[f.name]=sha(f)
source_state=json.loads((bases['qualified']/'tests/status.json').read_text());work=p.Path(source_state['work']);service=re.search(r'at (/\S+nano-service-module-\S+)',(bases['selectors-corrected']/'service.log').read_text()).group(1)
for name,b in {'source-artifacts':work,'native-artifacts-first-apple':bases['runtime']/'native','native-artifacts-homebrew-lsan':bases['selectors-corrected']/'native','service-artifacts':p.Path(service)}.items():
 d={str(f):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(b.rglob('*')) if f.is_file()};assert d;f=out/(name+'.json');f.write_text(json.dumps(d,indent=2)+'\n');reports[f.name]=sha(f)
first_service=re.search(r'at (/\S+nano-service-module-\S+)',(bases['adjacency-corrected']/'service.log').read_text()).group(1)
f=out/'first-service-artifacts.json';f.write_text(json.dumps({str(f):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(p.Path(first_service).rglob('*')) if f.is_file()},indent=2)+'\n');reports[f.name]=sha(f)
current={str(f):sha(f) for n in ('bin','obj','lib') for f in (root/n).rglob('*') if f.is_file()};f=out/'final-inputs.json';f.write_text(json.dumps(current,indent=2)+'\n');reports[f.name]=sha(f)
corrected=p.Path('/private/tmp/nanolang-mixed-source-adjacency-70c511dad');f=out/'corrected-final-inputs.json';f.write_text(json.dumps({str(f):sha(f) for n in ('bin','obj','lib') for f in (corrected/n).rglob('*') if f.is_file()},indent=2)+'\n');reports[f.name]=sha(f)
(out/'manifest.json').write_text(json.dumps({'source_pin':'70c511dad439f77eba15fe0bdc189e90ff9c9486','reports':reports,'scope':'Darwin same frozen source; Apple bootstrap CC, Homebrew LLVM23 sanitizer CC; source setup cache changes separately captured'},indent=2)+'\n')
for name,h in reports.items():assert sha(out/name)==h
print(len(reports),'sealed Darwin reports')
