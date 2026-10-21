from pathlib import Path
import hashlib,json,shlex,sys
root=Path('/Users/jkh/Src/nanolang-51-harness')
work=Path('/Users/jkh/Src/nanolang-51-owned-float-design')
out=Path(__file__).parent
baseline=Path('/private/tmp/borrowed-float-local-baseline-20261019')
sys.path.insert(0,str(work));sys.path.insert(0,str(work/'scripts'))
from tests.native_process import run_captured
from nanoisa_owned_reconstruction import OwnedAnalyze,OwnedEmit
counter=0
def run(argv,cwd=work,build=False):
 global counter
 counter+=1
 stem=out/('command-'+str(counter))
 stem.with_suffix('.json').write_text(json.dumps({'argv':list(map(str,argv)),'cwd':str(cwd)},indent=2))
 r=run_captured(argv,cwd=cwd,timeout=60,build=build,text=True)
 stem.with_suffix('.stdout').write_text(r.stdout);stem.with_suffix('.stderr').write_text(r.stderr)
 stem.with_suffix('.status').write_text(str(r.returncode))
 assert r.returncode==0,(argv,r.stderr[-3000:])
 return r.stdout
plan=run(['make','-n','-f','Makefile.gnu','-j1','CC=/opt/homebrew/opt/llvm/bin/clang','nvm2hl'],root)
line=next(line for line in plan.splitlines() if 'hl_facts_main.c' in line)
argv=shlex.split(line)
argv[argv.index('src/nanoisa/hl_facts_main.c')]=str(work/'src/nanoisa/hl_facts_main.c')
argv[argv.index('-o')+1]=str(out/'facts')
inputs=[Path(x) if Path(x).is_absolute() else root/x for x in argv if x.endswith(('.o','.a'))]
inputs += [work/'src/nanoisa/hl_owned_facts.inc',work/'scripts/nanoisa_owned_reconstruction.py']
(out/'inputs.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2))
run(argv,root,True)
link=work/'bin/nanoisa_hl_facts';assert link.is_symlink();link.unlink();link.symlink_to(out/'facts')
rows=[]
for name in ('nano_virt','original-shadows'):
 module=baseline/(name+'.nvm')
 facts=json.loads(run([out/'facts',module]));(out/(name+'.facts.json')).write_text(json.dumps(facts))
 functions=[OwnedAnalyze(facts,i).run() for i in range(len(facts['functions']))]
 emitter=OwnedEmit(functions,'nano',facts['layouts'])
 row={'name':name,'estimated':[emitter.nano_local_cost(f) for f in emitter.functions]}
 for lang in ('c','nano'):
  target=out/(name+'.'+lang);target.write_text('previous verified output')
  run([sys.executable,work/'scripts/nvm2hl.py','--language',lang,module,'-o',target])
  if lang=='nano':
   rebuilt=out/(name+'-rebuilt.nvm');run([root/'bin/nano_virt',target,'--emit-nvm','-o',rebuilt])
   run([root/'bin/nano_vm','--verify-only',rebuilt]);run([root/'bin/nano_vm',rebuilt])
   decoded=json.loads(run([out/'facts',rebuilt]));row['actual']=[f['locals'] for f in decoded['functions']]
 rows.append(row);(out/'counts.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
