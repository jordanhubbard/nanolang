import sys,os,json,hashlib,subprocess,shutil
from pathlib import Path
root=Path(sys.argv[1]);report=Path(sys.argv[2]);os.chdir(root)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={p:sha(p) for p in subprocess.check_output(['git','ls-files','-z']).decode().split('\0') if p and Path(p).is_file()}
paths={n:shutil.which(n) for n in ['make','cc','gcc','clang','python3','ar','ld']};paths['sanitizer']=shutil.which(os.environ.get('NANO_FILE_FLOW_CC','gcc'))
tools={n:{'path':p,'sha256':sha(p)} for n,p in paths.items() if p}
for name,obj in [('source-after.json',sources),('tools-after.json',tools),('first-terminal.json',{'status':1,'phase':'wrapper','error':'FileNotFoundError: bin/nano_virt','fixture_started':False,'cause':'Driver omitted provider build prerequisite in fresh checkout; preceding four phases passed.'})]:(report/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
