import hashlib,json,pathlib,shutil,subprocess,sys
old=pathlib.Path(sys.argv[1]).resolve();new=pathlib.Path(sys.argv[2]).resolve();reports=pathlib.Path(sys.argv[3]).resolve();out=pathlib.Path(sys.argv[4]);out.mkdir(exist_ok=False);store=out/'objects';store.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
assert git(old,'rev-parse','HEAD')=='ca3779e08b9cfc92414719a27e5e1456245dcfac'
assert git(new,'rev-parse','HEAD')=='e7614728b60b479dd1040bbf89268ed259bd456e'
rows={}
for name in ('full-test-sources-after.json','full-test-tools-after.json','full-test-products-after.json'):
 m=json.loads((reports/name).read_text())
 for path,row in m.items():assert sha(pathlib.Path(path))==row['sha256'],path
 rows[name]={'paths':len(m),'equal':True,'sha256':sha(reports/name)}
tracked=set(git(old,'ls-files').splitlines());changes=[];copied={}
for rel in tracked:
 src=old/rel;dst=new/rel
 if src.is_file() and dst.is_file():
  if sha(src)==sha(dst):shutil.copystat(src,dst)
  else:changes.append(rel)
assert set(changes)=={'docs/ROADMAP.md','tests/nanoisa/test_nvm_v2_layouts.c'},changes
for src in sorted(old.rglob('*')):
 rel=str(src.relative_to(old))
 if rel=='.git' or rel in tracked or not src.is_file():continue
 dst=new/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 if src.is_symlink():
  if dst.exists() or dst.is_symlink():dst.unlink()
  dst.symlink_to(src.readlink())
 else:shutil.copy2(src,dst)
 digest=sha(src);assert sha(dst)==digest
 if not (store/digest).exists():shutil.copyfile(src,store/digest)
 copied[rel]={'sha256':digest,'bytes':src.stat().st_size,'archive':str(store/digest),'original':str(src),'copy':str(dst),'symlink':str(src.readlink()) if src.is_symlink() else None}
for name in rows:
 m=json.loads((reports/name).read_text())
 assert all(sha(pathlib.Path(path))==row['sha256'] for path,row in m.items())
(out/'reuse.json').write_text(json.dumps({'old':str(old),'new':str(new),'original_endpoint_maps_verified':rows,'changed_tracked_files':changes,'copied_products':copied,'scope':'source-equal prior ca377 build/bootstrap products copied and freshly rehashed; no bootstrap rerun; additionally retained ignored products are current-copy evidence, not retroactive endpoint inventory'},indent=2)+'\n')
shutil.copyfile(__file__,out/'runner.py')
print('REUSE PASS',len(copied))
