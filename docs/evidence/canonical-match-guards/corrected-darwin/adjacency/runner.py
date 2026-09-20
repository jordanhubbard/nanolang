import hashlib,json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path('/private/tmp/nanolang-canonical-guards-eaf');prior=pathlib.Path('/private/tmp/nanolang-guards-eaf-darwin');out=pathlib.Path('/private/tmp/nanolang-guards-eaf-darwin-adjacency');out.mkdir(exist_ok=False);store=out/'artifacts';store.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
def archive(paths):
 rows={}
 for p in sorted(paths):
  if not p.is_file():continue
  h=sha(p);q=store/h
  if not q.exists():q.write_bytes(p.read_bytes())
  rows[str(p)]={'sha256':h,'bytes':p.stat().st_size,'archive':str(q)}
 return rows
original=json.loads((prior/'guards-tools-before.json').read_text());after=json.loads((prior/'guards-tools-after.json').read_text());assert all(after.get(k)==v for k,v in original.items());added=set(after)-set(original);assert len(added)==4 and all('/obj/module_cache/' in k for k in added)
assert all(sha(pathlib.Path(k))==v for k,v in after.items())
fixed={k:v for k,v in after.items() if '/obj/module_cache/' not in k and '/obj/nano_modules/' not in k};sources=json.loads((prior/'sources-after.json').read_text())
def check_sources():return {k:sha(root/k) for k in sources}
def fixed_now():return {k:sha(pathlib.Path(k)) for k in fixed}
def generated():return archive(p for d in ('obj/nano_modules','obj/module_cache') for p in (root/d).rglob('*'))
fixture=pathlib.Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-canonical-match-guards-4ftsjf58');command=fixture/'command-0051.json';terminal=fixture/'command-0051.terminal.json';library=next(pathlib.Path(p) for p in added if p.endswith('.dylib'));assert command.stat().st_mtime<=library.stat().st_mtime<=terminal.stat().st_mtime
save('prior-inventory-attribution.json',{'prior_files_unchanged':len(original),'added':{k:after[k] for k in sorted(added)},'command':json.loads(command.read_text()),'command_mtime':command.stat().st_mtime,'library_mtime':library.stat().st_mtime,'terminal_mtime':terminal.stat().st_mtime,'limits':'Timestamp correlation and cache source manifests; no per-command cache snapshot was recorded historically. Raw-driver attribution was incorrect.'})
save('attribution-artifacts.json',archive([command,terminal,*library.parent.iterdir()]))
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());env=os.environ.copy();env.update(CC='/usr/bin/clang',NANOLANG_GUARD_SAN_CC='/opt/homebrew/opt/llvm/bin/clang',NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'))
report={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'phases':[],'prior':str(prior/'manifest.json'),'repeated_prior_phases':False};save('manifest.json',report)
for name,modules in [('shared-policy',['tests.test_shared_match_policy_docs','tests.test_cseed_match_totality']),('owned-refusals',['tests.test_selected_variant_ownership.SelectedVariantOwnership.test_guarded_owned_match_remains_rejected','tests.test_generic_selected_ownership.GenericSelectedOwnership.test_guarded_generic_match_remains_rejected'])]:
 save(name+'-fixed-before.json',fixed_now());save(name+'-sources-before.json',check_sources());save(name+'-generated-before.json',generated());assert fixed_now()==fixed and check_sources()==sources
 args=[sys.executable,'-m','unittest','-v',*modules];start=time.monotonic();timeout=False;cleanup=[]
 with (out/(name+'.log')).open('wb') as log:
  p=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:p.wait(timeout=1200)
  except subprocess.TimeoutExpired:
   timeout=True
   for sig in (signal.SIGTERM,signal.SIGKILL):
    try:os.killpg(p.pid,sig);cleanup.append(sig.name)
    except ProcessLookupError:pass
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
 status=124 if timeout else p.returncode;f=fixed_now();s=check_sources();save(name+'-fixed-after.json',f);save(name+'-sources-after.json',s);save(name+'-generated-after.json',generated());save(name+'-fixed-artifacts.json',archive(map(pathlib.Path,f)))
 row={'name':name,'command':args,'status':status,'seconds':round(time.monotonic()-start,3),'timeout':timeout,'cleanup':cleanup,'fixed_equal':f==fixed,'source_equal':s==sources};report['phases'].append(row);save('manifest.json',report);print(row,flush=True)
 if status or f!=fixed or s!=sources:raise SystemExit(status or 65)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==report['head']
