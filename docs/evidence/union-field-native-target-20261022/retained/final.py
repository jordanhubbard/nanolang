from pathlib import Path
import subprocess,json,time,os,hashlib
p=Path('/private/tmp/nanolang-union-native-target-20261022');root=Path.cwd();status=[]
def run(name,args):
 (p/(name+'.argv.json')).write_text(json.dumps(args,indent=2)+'\n');start=time.monotonic()
 with (p/(name+'.stdout')).open('w') as out,(p/(name+'.stderr')).open('w') as err:
  r=subprocess.run(args,stdout=out,stderr=err,timeout=180,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:abort_on_error=1'})
 status.append({'name':name,'code':r.returncode,'seconds':time.monotonic()-start});(p/'final-status.json').write_text(json.dumps(status,indent=2)+'\n');print(status[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
a=json.loads((p/'observed.argv.json').read_text());inputs=[str(x.resolve()) for x in map(Path,a) if x.is_file()]
inputs += [str(x.resolve()) for x in Path('src/nanoisa').glob('*union*') if x.is_file()]+[str(Path('src/nanoisa/nvm2c.c').resolve())]
(p/'final-pins-before.json').write_text(json.dumps({x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in inputs},indent=2)+'\n')
run('final-observed-build',a);run('final-observed-run',[str(p/'observed')])
a=json.loads(Path('/private/tmp/nanolang-union-native-baseline-20261022/annotate-build.argv.json').read_text());at=a.index('/private/tmp/nanolang-union-native-baseline-20261022/annotate.c')
prefix=[x for x in a[:at] if x!='-g' and not x.startswith('-DNANO_')];prefix+=['-gline-tables-only']
for mode in ['default','private']:
 run('final-strict-'+mode,prefix+(['-DNANO_UNION_FIELD_PRIVATE_NATIVE'] if mode=='private' else [])+['-fsyntax-only','src/nanoisa/nvm2c.c'])
run('default-provider',prefix+['-c','src/nanoisa/nvm2c.c','-o',str(p/'default-nvm2c.o')])
link=[str(p/'target.c') if x==a[at] else str(p/'target') if x==a[-1] else str(p/'default-nvm2c.o') if x.endswith('/nanoisa/nvm2c.o') else x for x in a]
link=[x for x in link if x!='-g'];link.insert(1,'-gline-tables-only')
run('final-linked-build',link)
mods=[str(Path('/private/tmp/nanolang-union-native-baseline-20261022')/(x+'-selected.nvm')) for x in ['seed','c-shadow','canonical','canonical-shadow','raw-program','raw-shadow']]
run('final-linked-run',[str(p/'target')]+mods)
for source,label in [(str(p/'target.c'),'target-stack'),('src/nanoisa/managed_array_shapes.c','getters-stack')]:
 flags=[x for x in prefix if not x.startswith('-fsanitize=') and x!='-fno-sanitize-recover=all' and x!='-gline-tables-only']
 run(label,flags+['-fstack-usage','-c',source,'-o',str(p/(label+'.o'))])
(p/'final-pins-after.json').write_text(json.dumps({x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in inputs},indent=2)+'\n')
assert (p/'final-pins-before.json').read_bytes()==(p/'final-pins-after.json').read_bytes()
