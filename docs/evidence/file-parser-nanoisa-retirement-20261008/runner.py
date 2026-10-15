import json,os,subprocess,time,hashlib,shutil
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang');out=Path('/private/tmp/nanolang-parser-retirement-evidence-20261008');out.mkdir()
env=dict(os.environ);env['PATH']='/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:'+env['PATH'];env['NANOLANG_ROOT']=str(root)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
def run(name,args):
 start=time.monotonic()
 r=subprocess.run(list(map(str,args)),cwd=root,env=env,capture_output=True,timeout=600)
 (out/(name+'.log')).write_bytes(r.stdout+b'\n'+r.stderr)
 rows.append(dict(name=name,command=list(map(str,args)),exit_code=r.returncode,seconds=time.monotonic()-start))
 (out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 return r
shutil.copy2(__file__,out/'runner.py')
run('publisher',[root/'bin/nsi-file-binding',root/'tests/fixtures/nsi_file_plan.json','--file-binding-dir',out/'published'])
template=root/'tests/file_service_parser.nano.in';source=out/'parser.nano';source.write_text(template.read_text().replace('@PUBLISHER_SOURCE@',json.dumps((out/'published/binding.nano').read_text(),ensure_ascii=False)))
(out/'inputs.json').write_text(json.dumps({'template_sha':sha(template),'source_sha':sha(source),'tools':{name:sha(root/'bin'/name) for name in ['nanoc_c','nanoc_stage1','nanoc_stage2']}},indent=2)+'\n')
for name in ['nanoc_c','nanoc_stage1','nanoc_stage2']:
 exe=out/name
 run(name+'-compile',[root/'bin'/name,source,'-o',exe])
 r=run(name+'-execute',[exe]);assert r.stdout==b'publisher:1:5:retained\n',r.stdout
print('PASS all three fixture products',flush=True)
