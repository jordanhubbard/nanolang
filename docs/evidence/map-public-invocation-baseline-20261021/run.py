from pathlib import Path
import json,shlex,os,subprocess,sys,time,hashlib
r=Path.cwd();w=Path(__file__).parent;sys.path.insert(0,str(r));from tests.native_process import run_captured
e=json.loads((r/'docs/evidence/map-private-native-20261021/environment.json').read_text());cc=shlex.split(e['RECORD_ARRAY_VM_CC']);objects=shlex.split(e['RECORD_ARRAY_VM_OBJECTS']);rows=[]
def run(n,a,build=False):
 t=time.monotonic();p=run_captured(list(map(str,a)),cwd=r,env=dict(os.environ,LSAN_OPTIONS=''),timeout=180,build=build);(w/(n+'.stdout')).write_bytes(p.stdout);(w/(n+'.stderr')).write_bytes(p.stderr);rows.append(dict(name=n,argv=list(map(str,a)),exit=p.returncode,seconds=time.monotonic()-t));(w/'status.json').write_text(json.dumps(rows,indent=2));print(n,p.returncode,flush=True)
 if p.returncode:print(p.stderr.decode());sys.exit(1)
 return p
run('build',cc+shlex.split(e['RECORD_ARRAY_VM_CFLAGS'])+['-O0','-std=c11','-D_DEFAULT_SOURCE','-Isrc/nanovm','-Itests/nanoisa',w/'baseline.c','tests/nanoisa/record_array_alloc.c',*objects,*shlex.split(e['RECORD_ARRAY_VM_LDFLAGS']),'-o',w/'baseline'],True)
p=run('ordinary',[w/'baseline',*shlex.split(e['RECORD_MAP_MODULES'])]);print(p.stdout.decode(),flush=True)
(w/'pins.json').write_text(json.dumps({str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [*cc,*objects,*shlex.split(e['RECORD_MAP_MODULES']),w/'baseline.c']},indent=2))
