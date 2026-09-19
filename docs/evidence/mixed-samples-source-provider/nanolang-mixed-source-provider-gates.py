import pathlib as p,subprocess as s,os,shutil,json,hashlib,time
root=p.Path('/home/jkh/Src/nanolang-mixed-source-provider');old=p.Path('/home/jkh/Src/nanolang-mixed-source-final');out=p.Path('/tmp/nanolang-mixed-source-provider-fbd3b5a9a');out.mkdir(exist_ok=False)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
git=lambda *a:s.check_output(['git',*a],cwd=root,text=True).strip()
assert git('rev-parse','HEAD').startswith('fbd3b5a9a') and not git('status','--porcelain')
for name in ('bin','obj'):shutil.copytree(old/name,root/name,symlinks=True)
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
sources=lambda:{f:sha(root/f) for f in files if (root/f).is_file()}
inputs=lambda:{str(f):sha(f) for name in ('bin','obj','lib') for f in (root/name).rglob('*') if f.is_file()}
before=sources();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n');(out/'inputs-before.json').write_text(json.dumps(inputs(),indent=2)+'\n')
stages={n:sha(root/'bin'/n) for n in ('nanoc_stage1','nanoc_stage2','nanoisa_emit')};(out/'stages-before.json').write_text(json.dumps(stages,indent=2)+'\n')
for f in (p.Path(__file__),p.Path('/tmp/nanolang-mixed-source-final-smoke.py')):shutil.copy2(f,out/f.name)
tools={n:{'path':str(p.Path(shutil.which(n)).resolve()),'sha256':sha(p.Path(shutil.which(n)).resolve())} for n in ('cc','gcc','clang','make','python3','ar','ld','opt')};(out/'tools-before.json').write_text(json.dumps(tools,indent=2)+'\n')
env=os.environ.copy();env.update(PYTHONPATH=str(root),CC='/usr/bin/gcc',NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),NANO_MIXED_RUNTIME_DIR=str(out/'native'));(out/'native').mkdir()
commands=[('provider-tools',['make','-j2','bin/nanoc_c','nano_virt','nano_vm','nvm2c','nvm2llvm','nvm2hl','nanoisa_dump','test-local-binding-metadata']),('source-smoke',['python3','/tmp/nanolang-mixed-source-final-smoke.py'])]
report={'pin':git('rev-parse','HEAD'),'steps':[]};status=1
try:
 for name,cmd in commands:
  print('START',name,flush=True);t=time.monotonic()
  with (out/(name+'.log')).open('wb') as f:
   try:status=s.run(cmd,cwd=root,env=env,stdout=f,stderr=s.STDOUT,timeout=900).returncode
   except s.TimeoutExpired:status=124
  report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-t,3)});(out/'status.json').write_text(json.dumps(report,indent=2)+'\n');print('END',name,status,flush=True)
  if status:break
finally:
 after=sources();(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n');(out/'inputs-after.json').write_text(json.dumps(inputs(),indent=2)+'\n');final={n:sha(root/'bin'/n) for n in stages};(out/'stages-after.json').write_text(json.dumps(final,indent=2)+'\n');(out/'tools-after.json').write_text(json.dumps({n:{'path':v['path'],'sha256':sha(p.Path(v['path']))} for n,v in tools.items()},indent=2)+'\n');report.update(sources_unchanged=before==after,stages_unchanged=stages==final,clean=not git('status','--porcelain','--untracked-files=no'));(out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(status)
