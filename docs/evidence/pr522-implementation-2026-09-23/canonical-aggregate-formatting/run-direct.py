from pathlib import Path
import subprocess,json,os,shlex,sys
root=Path.cwd();inputs=Path('/tmp/pr522-aggregate-lowering-cases')
p=Path('/tmp/pr522-aggregate-lowering-direct-'+sys.argv[1]);p.mkdir(exist_ok=True);rows=[]
for source in sorted(inputs.glob('*.nano')):
 src=p/source.name;src.write_text(source.read_text())
 asm=src.with_suffix('.nasm');nvm=src.with_suffix('.nvm');c=src.with_suffix('.c');exe=src.with_suffix('.exe')
 row={'source':src.name,'steps':[]}
 commands=[[root/'bin/nanoisa_emit',src,'-o',asm],[root/'bin/nanoisa','asm',asm,'-o',nvm],[root/'bin/nano_vm',nvm],[root/'bin/nvm2c',nvm,'-o',c],
           [*shlex.split(os.environ.get('NANO_CC','cc')),'-std=c11','-Wall','-Wextra','-Werror',*shlex.split(os.environ.get('NANO_CFLAGS','')),c,'-lm',*shlex.split(os.environ.get('NANO_LDFLAGS','')),'-o',exe],[exe]]
 for cmd in commands:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=120);row['steps'].append({'command':list(map(str,cmd)),'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
  if r.returncode:break
 rows.append(row);print(src.name,[(r['command'][0],r['exit']) for r in row['steps']],flush=True)
p.joinpath('results.json').write_text(json.dumps(rows,indent=2))
raise SystemExit(any(row['steps'][-1]['exit'] for row in rows))
