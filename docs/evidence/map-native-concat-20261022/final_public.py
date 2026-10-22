from pathlib import Path
import json,sys,os,hashlib
sys.path.insert(0,str(Path.cwd()))
from tests.native_process import run_captured
root=Path.cwd();agg=Path('/Users/jkh/Src/nanolang-51-harness');out=Path(__file__).parent;dest=out/'qualified';dest.mkdir(exist_ok=True)
w=Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-map-public-8d6uwbpz');s=json.loads((w/'selection.json').read_text());rows=[]
def run(name,args,build=False):
 r=run_captured(args,cwd=root,env=dict(os.environ,RECORD_MAP_PUBLIC_OUTPUT=str(dest)),timeout=180,build=build);(out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr);rows.append(dict(name=name,argv=args,status=r.returncode));(out/'final-public-status.json').write_text(json.dumps(rows,indent=2)+'\n');assert r.returncode==0 and not r.stderr,r.stderr;return r.stdout
obj=out/'current-nvm2c.o';run('current-native-provider',s['cc']+s['flags']+['-c','src/nanoisa/nvm2c.c','-o',str(obj)],True)
objects=[str(obj) if Path(x).stem=='nvm2c' else str(w/('public-switch-'+Path(x).stem+'.o')) for x in s['sources']]+[str(agg/x) if not Path(x).is_absolute() else x for x in s['objects']]
source=(out/'annotated.c').read_text().replace('native_expected=i<=4;','native_expected=true;').replace('CHECK(argc==9);','CHECK(argc==9);public_native_boundaries();');(out/'qualified.c').write_text(source)
run('qualified-build',s['cc']+s['flags']+['-DNANO_NO_COMPUTED_GOTO',str(out/'qualified.c')]+objects+s['ldflags']+['-o',str(out/'qualified-check')],True)
modules=[str(out/(n+'-'+p+'.nvm')) for n in ['unchanged','concat'] for p in ['seed','c-hosted','stage1','stage2']]
print(run('qualified-public',[str(out/'qualified-check')]+modules).decode())
# Fresh ordinary public output through the changed default translator, raw inputs.
run('raw-current-build',s['cc']+s['flags']+['/private/tmp/nanolang-map-native-final-checks/golden.c']+objects+s['ldflags']+['-o',str(out/'raw-current')],True)
comparison=[]
for i,m in enumerate(modules):
 output=run('raw-current-'+str(i),[str(out/'raw-current'),m]);prior=Path(m).with_suffix('.c').read_bytes();assert output==prior;comparison.append(dict(module=m,sha256=hashlib.sha256(output).hexdigest(),bytes=len(output)))
for i in range(4):
 # Exact old selected output generated before the exception is unchanged too.
 assert (dest/('original-'+str(i)+'.c')).read_bytes()==(out/('original-'+str(i)+'.c')).read_bytes()
(out/'old-output-bytes.json').write_text(json.dumps(comparison,indent=2)+'\n')
for compiler in ['/usr/bin/cc','/opt/homebrew/bin/gcc-16']:
 for private in [False,True]:
  args=[compiler,'-std=c11','-D_DEFAULT_SOURCE','-Isrc','-Isrc/nanoisa','-Wall','-Wextra','-Werror','-fsyntax-only','src/nanoisa/nvm2c.c']
  if private:args+=['-DNANO_RECORD_MAP_PRIVATE_NATIVE']
  run(Path(compiler).name+('-private' if private else '-public'),args,True)
print('PASS actual public boundaries, 8 raw byte goldens, 4 selected old byte goldens, strict4')
