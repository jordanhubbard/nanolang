from pathlib import Path
import subprocess,json,time,hashlib,os
p=Path('/private/tmp/nanolang-union-native-target-20261022');root=Path('/Users/jkh/Src/nanolang-51-platform-qualification')
argv=json.loads(Path('/private/tmp/nanolang-union-native-baseline-20261022/annotate-build.argv.json').read_text())
prefix=argv[:argv.index('/private/tmp/nanolang-union-native-baseline-20261022/annotate.c')]
status=[]
def run(name,args):
 (p/(name+'.argv.json')).write_text(json.dumps(args,indent=2)+'\n');start=time.monotonic()
 with (p/(name+'.stdout')).open('w') as out,(p/(name+'.stderr')).open('w') as err:
  r=subprocess.run(args,cwd=root,stdout=out,stderr=err,timeout=180,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:abort_on_error=1'})
 status.append({'name':name,'code':r.returncode,'seconds':time.monotonic()-start});(p/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(status[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
inputs=[str(x) for x in root.glob('src/nanoisa/*union*') if x.is_file()]+[str(root/'src/nanoisa/nvm2c.c')]+[x for x in argv if Path(x).is_file()]
(p/'pins-before.json').write_text(json.dumps({x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in inputs},indent=2)+'\n')
for mode in ['default','private']:
 flags=[x for x in prefix if not x.startswith('-DNANO_')]
 if mode=='private':flags+=['-DNANO_UNION_FIELD_PRIVATE_NATIVE']
 run('strict-'+mode,flags+['-fsyntax-only','src/nanoisa/nvm2c.c'])
args=[str(p/'target.c') if x=='/private/tmp/nanolang-union-native-baseline-20261022/annotate.c' else str(p/'target') if x=='/private/tmp/nanolang-union-native-baseline-20261022/annotate' else x for x in argv]
run('build',args)
mods=[str(Path('/private/tmp/nanolang-union-native-baseline-20261022')/(x+'-selected.nvm')) for x in ['seed','c-shadow','canonical','canonical-shadow','raw-program','raw-shadow']]
run('six-originals',[str(p/'target')]+mods)
(p/'pins-after.json').write_text(json.dumps({x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in inputs},indent=2)+'\n')
assert (p/'pins-before.json').read_bytes()==(p/'pins-after.json').read_bytes()
