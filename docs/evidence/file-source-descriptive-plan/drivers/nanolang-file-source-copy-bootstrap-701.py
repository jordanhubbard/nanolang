import hashlib,json,pathlib,shutil,sys
old=pathlib.Path(sys.argv[1]);new=pathlib.Path(sys.argv[2]);output=pathlib.Path(sys.argv[3]);inputs=json.loads(pathlib.Path(sys.argv[4]).read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
source={};changed=[]
for name in inputs:
 if name.startswith(('src/','src_nano/')):
  before=digest(old/name);after=digest(new/name)
  source[name]={'original':before,'copied_tree':after}
  if before!=after:changed.append(name)
if changed!=['src_nano/compiler/file_source_plan.nano']:raise RuntimeError('Unexpected compiler-source delta: '+repr(changed))
for name in inputs:
 if name.startswith('src_nano/') and name.endswith('.nano') and name!='src_nano/compiler/file_source_plan.nano':
  if 'file_source_plan' in (new/name).read_text():raise RuntimeError('Plan appears in another compiler source: '+name)
copied={}
for base in ('bin','obj'):
 for path in sorted((old/base).rglob('*')):
  if not path.is_file():continue
  rel=path.relative_to(old)
  if any(x in ('module_cache','nano_modules') for x in rel.parts):continue
  dest=new/rel
  if dest.exists():raise RuntimeError('Destination already contains product: '+str(dest))
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
  a=digest(path);b=digest(dest)
  if a!=b:raise RuntimeError('Copy mismatch: '+str(rel))
  copied[str(rel)]={'original':a,'copied':b,'bytes':dest.stat().st_size}
for bad in (new/'obj/module_cache',new/'obj/nano_modules',new/'modules/file_source_catalog/.build'):
 if bad.exists():raise RuntimeError('Stale module cache: '+str(bad))
output.write_text(json.dumps({'bootstrap_pin':'32ade5b5d','fixture_pin':'701dca50f','original_root':str(old),'new_root':str(new),'compiler_source_equal_except_unimported_plan':True,'changed_tested_plan':changed,'dependency_boundary':'No other src_nano file references file_source_plan; all other src/src_nano inputs exact; old bootstrap compilers/providers copied by hash; tested plan recompiles as fresh imported module','sources':source,'copied_products':copied,'excluded':['obj/module_cache','obj/nano_modules','all module .build directories']},indent=2)+'\n')
print('Exact compiler sources:',len(source),'hash-copied products:',len(copied))
