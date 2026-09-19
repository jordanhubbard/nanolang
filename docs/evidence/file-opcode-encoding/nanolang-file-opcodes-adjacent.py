import os,sys,hashlib,json,subprocess,time,shutil
from pathlib import Path
root=Path(sys.argv[1]);report=Path(sys.argv[2]);report.mkdir(parents=True,exist_ok=True)
os.chdir(root)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():return {p:digest(p) for p in subprocess.check_output(['git','ls-files','-z']).decode().split('\0') if p and Path(p).is_file()}
def tools():
 names=['make','cc','gcc','clang','python3','ar','ld']
 paths={n:shutil.which(n) for n in names}
 paths['sanitizer']=shutil.which(os.environ.get('NANO_FILE_OPCODE_TEST_CC','gcc'))
 return {n:{'path':p,'sha256':digest(p)} for n,p in paths.items() if p}
def write(name,obj):(report/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
write('source-before.json',sources());write('tools-before.json',tools())
steps=[('clang',['make','test-file-opcodes']),('adjacent',['make','test-nanoisa','test-disasm-roundtrip','test-nvm-v2-convert','test-service-module','test-file-nominal-module']),('wrapper',['make','-j4','test-wrapper-gen'])]
results=[]
for label,cmd in steps:
 start=time.monotonic()
 with (report/(label+'.log')).open('w') as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
 result={'phase':label,'command':cmd,'status':r.returncode,'seconds':round(time.monotonic()-start,3)};results.append(result);write('results.json',results)
 artifacts=report/(label+'-artifacts');artifacts.mkdir(exist_ok=True)
 entries={}
 for base in ['obj','bin']:
  for p in Path(base).rglob('*') if Path(base).exists() else []:
   if p.is_file() and (p.suffix in ('.o','.a','.so') or os.access(p,os.X_OK)):
    sha=digest(p);dest=artifacts/sha
    if not dest.exists():shutil.copyfile(p,dest)
    entries[str(p)]={'sha256':sha,'artifact':str(dest)}
 # unittest emits its retained executable directory; archive each real binary.
 import re
 text=(report/(label+'.log')).read_text()
 for directory in re.findall(r'I retain File encoding artifacts at (\S+)',text):
  for p in Path(directory).iterdir():
   if p.is_file():
    sha=digest(p);dest=artifacts/sha
    if not dest.exists():shutil.copyfile(p,dest)
    entries[str(p)]={'sha256':sha,'artifact':str(dest)}
 write(label+'-artifacts.json',entries)
 print(json.dumps(result),flush=True)
 if r.returncode:break
write('source-after.json',sources());write('tools-after.json',tools())
sys.exit(results[-1]['status'])
