from pathlib import Path
import subprocess,os,json,time,hashlib
r=Path(__file__).parent;old=Path('/private/tmp/nanolang-union-field-baseline-20261022');agg=Path('/Users/jkh/Src/nanolang-51-harness/bin');env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1');rows=[]
def run(name,args,want):
 t=time.monotonic();p=subprocess.run(list(map(str,args)),env=env,capture_output=True,timeout=60);(r/(name+'.stdout')).write_bytes(p.stdout);(r/(name+'.stderr')).write_bytes(p.stderr);rows.append(dict(name=name,command=list(map(str,args)),code=p.returncode,seconds=time.monotonic()-t));(r/'selected-status.json').write_text(json.dumps(rows,indent=2));print(name,p.returncode,flush=True);assert not any(x in p.stderr for x in [b'AddressSanitizer',b'runtime error:']);assert (p.returncode==0)==want;return p
for n in ['seed','c-shadow','canonical','canonical-shadow','raw-program','raw-shadow']:
 selected=r/(n+'-selected.nvm');run(n+'-annotate',[r/'annotate',old/(n+'.nvm'),selected],True);run(n+'-selected-verify',[agg/'nano_vm','--verify-only',selected],False);out=r/(n+'-refused.c');out.write_bytes(b'prior output sentinel\n');run(n+'-selected-native',[agg/'nvm2c',selected,'-o',out],False);assert out.read_bytes()==b'prior output sentinel\n'
pins=json.loads((r/'pins-before.json').read_text());after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in pins};assert after==pins;(r/'pins-after.json').write_text(json.dumps(after,indent=2));print('PASS six independently annotated selected refusals and prior CLI output preservation',flush=True)
