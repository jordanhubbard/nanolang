import hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-mixed-samples-runtime')
out=pathlib.Path('/tmp/nanolang-managed-native-embedding-fe321533');out.mkdir()
def digest(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
names=['gcc','clang','python3','readelf','ld','as']
paths={n:str(pathlib.Path(shutil.which(n)).resolve()) for n in names}
paths['clang-wrapper']='/tmp/nanolang-projection-clang'
for name,flag in [('cc1','-print-prog-name=cc1'),('collect2','-print-prog-name=collect2')]:
 paths[name]=subprocess.check_output(['gcc',flag],text=True).strip()
def inventory():
 files=subprocess.check_output(['git','ls-files','src','tests','scripts','Makefile.gnu'],cwd=root,text=True).splitlines()
 return {'sources':{p:digest(root/p) for p in files if (root/p).is_file()},'tools':{p:digest(p) for p in paths.values()}}
before=inventory();(out/'before.json').write_text(json.dumps(before,indent=2)+'\n')
env=os.environ.copy();env.update(NMS_EMBED_EVIDENCE=str(out/'programs'),NMS_EMBED_CLANG=paths['clang-wrapper'])
command=['python3','-m','unittest','tests.test_managed_native_embedding','-v']
start=time.monotonic()
with open(out/'gate.log','wb') as log:
 result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
report=dict(pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),command=command,status=result.returncode,seconds=time.monotonic()-start,tools=paths,environment={k:env[k] for k in ['NMS_EMBED_EVIDENCE','NMS_EMBED_CLANG']})
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
after=inventory();(out/'after.json').write_text(json.dumps(after,indent=2)+'\n');assert before==after
print(json.dumps(report))
raise SystemExit(result.returncode)
