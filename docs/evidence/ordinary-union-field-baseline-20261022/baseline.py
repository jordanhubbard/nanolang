from pathlib import Path
import os,sys,json,hashlib,subprocess,time,re,ctypes
root=Path.cwd();p=Path(__file__).parent;agg=Path('/Users/jkh/Src/nanolang-51-harness');old=Path('/private/tmp/nanolang-map-string-publication-20261022')
source=root/'docs/evidence/generic-record-field-context-20261021/unchanged-direct.nano';(p/'unchanged-direct.nano').write_bytes(source.read_bytes());source=p/'unchanged-direct.nano'
tools={n:agg/'bin'/n for n in ['nano_virt','nanoisa','nano_vm','nvm2c']};tools.update({'c-shadow':agg/'obj/borrow_shadow_names','canonical':old/'canonical','raw':old/'raw','cc':Path('/opt/homebrew/opt/llvm/bin/clang')})
libs=list((old/'module-cache').rglob('libnanoisa.dylib'));assert len(libs)==1;tools['query-library']=libs[0];sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();pins={str(q):sha(q) for q in tools.values()};(p/'pins-before.json').write_text(json.dumps(pins,indent=2));env=dict(os.environ,NANO_VM=str(tools['nano_vm']),LSAN_OPTIONS='');rows=[];modules=[]
def run(name,args,extra=None):
 t=time.monotonic();r=subprocess.run(list(map(str,args)),cwd=root,env=extra or env,capture_output=True,timeout=180);(p/(name+'.stdout')).write_bytes(r.stdout);(p/(name+'.stderr')).write_bytes(r.stderr);rows.append(dict(name=name,argv=list(map(str,args)),status=r.returncode,seconds=time.monotonic()-t));(p/'status.json').write_text(json.dumps(rows,indent=2));assert not r.returncode,(name,r.stderr);assert not any(x in r.stderr for x in [b'AddressSanitizer',b'runtime error:']),r.stderr;return r.stdout
lib=ctypes.CDLL(str(libs[0]));query=lib.nl_nanoisa_array_publication;query.argtypes=[ctypes.c_char_p];query.restype=ctypes.c_int64
run('seed-source',[tools['nano_virt'],source,'--emit-nvm','-o',p/'seed.nvm']);modules.append('seed')
asm=run('c-shadow-source',[tools['c-shadow'],source]);(p/'c-shadow.nasm').write_bytes(asm);run('c-shadow-asm',[tools['nanoisa'],'asm',p/'c-shadow.nasm','-o',p/'c-shadow.nvm']);modules.append('c-shadow')
wrapper=p/'capture-vm';wrapper.write_text('#!'+sys.executable+'\n'+'''import os,sys,json,hashlib
from pathlib import Path
args=sys.argv[1:];assert len(args)==2 and args[0]=='--check-shadows'
out=Path(os.environ['CAPTURE_MODULE']);assert not out.exists()
data=Path(args[1]).read_bytes();out.write_bytes(data);out.with_suffix('.capture.json').write_text(json.dumps(dict(argv=args,sha256=hashlib.sha256(data).hexdigest(),vm=os.environ['REAL_VM'])))
os.execv(os.environ['REAL_VM'],[os.environ['REAL_VM'],*args])
''');wrapper.chmod(0o755)
run('canonical-source',[tools['canonical'],source,'--emit-nvm','-o',p/'canonical.nvm'],dict(env,NANO_VM=str(wrapper),REAL_VM=str(tools['nano_vm']),CAPTURE_MODULE=str(p/'canonical-shadow.nvm')));modules+=['canonical','canonical-shadow']
for mode,label in [('program','raw-program'),('raw','raw-shadow')]:
 asm=run(label+'-source',[tools['raw'],source,0,mode]);(p/(label+'.nasm')).write_bytes(asm);run(label+'-asm',[tools['nanoisa'],'asm',p/(label+'.nasm'),'-o',p/(label+'.nvm')]);modules.append(label)
summary=[]
for name in modules:
 m=p/(name+'.nvm');run(name+'-verify',[tools['nano_vm'],'--verify-only',m]);run(name+'-vm',[tools['nano_vm'],m]);dump=run(name+'-dump',[tools['nanoisa'],'dump',m]);assert b'.ownership ' not in dump and b'.layouts ' not in dump,(name,'unexpected authority');q=query(dump);assert q==0,(name,q)
 counts={'name':name,'sha256':sha(m),'bytes':m.stat().st_size,'types':re.search(rb'^\.types (.+)$',dump,re.M).group(1).decode(),'functions':len(re.findall(rb'^\.function ',dump,re.M)),'instructions':len(re.findall(rb'^\s+[A-Z][A-Z0-9_]*(?:\s|$)',dump,re.M)),'query':q};summary.append(counts);(p/'counts.json').write_text(json.dumps(summary,indent=2))
 run(name+'-native',[tools['nvm2c'],m,'-o',p/(name+'.c')])
 for opt in ['0','2']:
  exe=p/(name+'-O'+opt);run(name+'-build-O'+opt,[tools['cc'],'-std=c11','-O'+opt,'-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',p/(name+'.c'),'-lm','-o',exe]);run(name+'-run-O'+opt,[exe])
 print(name,'verified VM/native O0/O2; metadata absent; query0',flush=True)
after={q:sha(Path(q)) for q in pins};assert after==pins;(p/'pins-after.json').write_text(json.dumps(after,indent=2));print('PASS unchanged union-field baseline six independently produced ordinary/shadow modules',flush=True)
