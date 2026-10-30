from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,str(Path.cwd()));from tests.native_process import run_captured,native_compile_timeout
w=Path(__file__).parent;r=Path('/Users/jkh/Src/nanolang-51-harness');tools={n:r/'bin'/n for n in ['nanoisa','nano_vm','nvm2c']};cc=Path('/opt/homebrew/opt/llvm/bin/clang');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={str(p):sha(p) for p in [*tools.values(),cc]};receipt={'before':pins,'commands':[]}
def run(n,args,build=False):
 t=time.monotonic();p=run_captured(list(map(str,args)),cwd=Path.cwd(),timeout=native_compile_timeout(default=900) if build else 180,build=build);(w/(n+'.stdout')).write_bytes(p.stdout);(w/(n+'.stderr')).write_bytes(p.stderr);receipt['commands'].append({'name':n,'argv':list(map(str,args)),'exit':p.returncode,'seconds':time.monotonic()-t});(w/'receipt.json').write_text(json.dumps(receipt,indent=2));print(n,p.returncode,flush=True);return p
for name in ['keys','missing']:
 if name=='missing':
  text=(w/'keys.nasm').read_text();text=text.replace('"a\\0x"','"ax"').replace('"a\\0y"','"ay"');(w/'missing.nasm').write_text(text)
 src=w/(name+'.nasm');mod=w/(name+'.nvm');c=w/(name+'.c');exe=w/(name+'.native')
 assert run(name+'-asm',[tools['nanoisa'],'asm',src,'-o',mod]).returncode==0
 assert run(name+'-verify',[tools['nano_vm'],'--verify-only',mod]).returncode==0
 assert run(name+'-vm',[tools['nano_vm'],mod]).returncode==0
 assert run(name+'-dump',[tools['nanoisa'],'dump',mod]).returncode==0
 p=run(name+'-translate',[tools['nvm2c'],mod,'-o',c])
 if p.returncode==0:
  assert run(name+'-build',[cc,'-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',exe],True).returncode==0
  run(name+'-native',[exe])
receipt['after']={p:sha(p) for p in pins};assert receipt['after']==pins;receipt['sources']={p.name:sha(p) for p in w.glob('*.nasm')};(w/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
