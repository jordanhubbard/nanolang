from pathlib import Path
import os,json,hashlib,subprocess,time
r=Path(__file__).parent/'contrast-resume';r.mkdir(exist_ok=True);old=Path('/private/tmp/nanolang-union-field-baseline-20261022');halt=Path('/private/tmp/nanolang-union-halt-baseline-20261022');agg=Path('/Users/jkh/Src/nanolang-51-harness');tools={k:agg/'bin'/k for k in ['nano_vm','nvm2c','nanoisa']};tools['cc']=Path('/opt/homebrew/opt/llvm/bin/clang');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();names=['seed','c-shadow','canonical','canonical-shadow','raw-program','raw-shadow'];paths=list(tools.values())+[old/(n+'.nvm') for n in names];pins={str(p):sha(p) for p in paths};(r/'pins-before.json').write_text(json.dumps(pins,indent=2));rows=[];env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1',LSAN_OPTIONS='')
def run(name,cmd,expect=0):
 t=time.monotonic();p=subprocess.run(list(map(str,cmd)),env=env,capture_output=True,timeout=180);(r/(name+'.stdout')).write_bytes(p.stdout);(r/(name+'.stderr')).write_bytes(p.stderr);rows.append(dict(name=name,command=list(map(str,cmd)),code=p.returncode,seconds=time.monotonic()-t));(r/'status.json').write_text(json.dumps(rows,indent=2));print(name,p.returncode,flush=True);assert not any(x in p.stderr for x in [b'AddressSanitizer',b'runtime error:']),p.stderr
 if expect is not None:assert p.returncode==expect,(name,p.stderr)
 return p
for n in ['false-assert','init-halt','init-return']:
 m=r/(n+'.nvm');run(n+'-assemble',[tools['nanoisa'],'asm',halt/(n+'.nasm'),'-o',m]);run(n+'-verify',[tools['nano_vm'],'--verify-only',m]);run(n+'-native',[tools['nvm2c'],m,'-o',r/(n+'.c')]);exe=r/n;run(n+'-build',[tools['cc'],'-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',r/(n+'.c'),'-lm','-o',exe]);p=run(n+'-run',[exe],None)
 if n=='false-assert':assert p.returncode==-6 and p.stderr==b'I stopped at a native invariant in nl_main at generated C line 56.\n'
 else:assert p.returncode>=0
 if n in ['init-halt','init-return']:
  text='#define main retained_generated_main\n#include "'+str(r/(n+'.c'))+'"\n#undef main\nint main(void){int result=retained_generated_main();printf("result=%d global_kind=%u global=%lld\\n",result,(unsigned)nglobal[0].kind,(long long)nglobal[0].integer);return 0;}\n';probe=r/(n+'-probe.c');probe.write_text(text);run(n+'-probe-build',[tools['cc'],'-std=c11','-O1',probe,'-lm','-o',r/(n+'-probe')]);run(n+'-probe',[r/(n+'-probe')])
after={str(p):sha(p) for p in paths};assert after==pins;(r/'pins-after.json').write_text(json.dumps(after,indent=2));print('baseline complete',flush=True)
