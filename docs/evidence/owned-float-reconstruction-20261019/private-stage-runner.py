import json,subprocess,hashlib,re,sys
from pathlib import Path
r=Path.cwd();w=Path('/private/tmp/owned-float-private-stages-20261019');w.mkdir(exist_ok=True)
base=Path('/Users/jkh/Src/nanolang-51-harness/obj/bootstrap-nanoisa/run-cfsrcu5q');d=json.loads((base/'manifest.json').read_text())
sources=[(r/'modules/nanoisa'/p).resolve() for p in json.loads((r/'modules/nanoisa/module.json').read_text())['c_sources']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={str(p):sha(p) for p in sources}
lib=w/'libnanoisa.dylib'
a=['cc','-std=c11','-O1','-dynamiclib','-D_GNU_SOURCE','-Isrc','-Isrc/nanoisa','-Imodules/nanoisa',*[str(p) for p in sources],'-lm','-lcrypto','-lffi','-L/opt/homebrew/Cellar/openssl@4/4.0.3/lib','-I/opt/homebrew/Cellar/openssl@4/4.0.3/include','-o',str(lib)]
def run(name,a):
 p=subprocess.run(a,capture_output=True,timeout=600);(w/(name+'.log')).write_bytes(p.stdout+p.stderr);(w/(name+'.command.json')).write_text(json.dumps(a,indent=2));assert p.returncode==0,(name,p.returncode,p.stderr[-2000:]);print(name,'passed',flush=True)
run('library',a)
receipt={'source_hashes':pins,'library':sha(lib),'stages':{}}
for name in ('stage1','stage2'):
 original=base/(name+'.c');text=original.read_text();paths=set(re.findall(r'dlopen\("([^"]*libnanoisa.dylib)"',text));assert len(paths)==1,paths
 old=paths.pop();count=text.count(old);assert count>0
 changed=text.replace(old,str(lib));assert changed.replace(str(lib),old)==text
 copy=w/(name+'.c');copy.write_text(changed)
 step=next(s for s in d['steps'] if s['label']==name+'-native');a=[str(copy) if x==str(original) else str(w/name) if x==str(base/name) else x for x in step['argv']]
 receipt['stages'][name]={'original_c':sha(original),'changed_c':sha(copy),'original_library_path':old,'original_library_sha256':sha(Path(old)),'replacement_count':count,'raw_stage_sha256':sha(base/(name+'.nvm')) if (base/(name+'.nvm')).exists() else None}
 run(name,a);receipt['stages'][name]['native_sha256']=sha(w/name)
(w/'receipt.json').write_text(json.dumps(receipt,indent=2));assert all(sha(Path(p))==v for p,v in pins.items())
