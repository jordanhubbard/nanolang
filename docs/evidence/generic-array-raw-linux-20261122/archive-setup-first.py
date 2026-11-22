from pathlib import Path
import subprocess,json,hashlib,gzip,time,os,sys
repo=Path.cwd();root=Path('/home/jkh/.worktrees/nanolang/generic-array-raw-linux-20261122');out=repo/'docs/evidence/generic-array-raw-linux-20261122';(out/'objects').mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root));from scripts.bootstrap_nanoisa import Bootstrap
b=Bootstrap(root,1800,30);b.load();b.check_inputs();b.check_artifacts(list(b.manifest['artifacts']));controller=json.loads((root/'.test_output/raw-qualification/controller.json').read_text());assert controller['status']=='passed';files=[];objects={};start=time.monotonic();sha=lambda raw:hashlib.sha256(raw).hexdigest()
def add(logical,p):
 raw=p.read_bytes();h=sha(raw)
 if h not in objects:
  packed=gzip.compress(raw,compresslevel=6,mtime=0);dest=out/'objects'/(h+'.gz');dest.write_bytes(packed);objects[h]=dict(artifact='objects/'+dest.name,compressed_bytes=len(packed),compressed_sha256=sha(packed))
 files.append(dict(path=logical,bytes=len(raw),sha256=h,**objects[h]))
def tree(logical,path):
 for p in sorted(path.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:add(logical+'/'+str(p.relative_to(path)),p)
tree('bootstrap/'+b.work.name,b.work);tree('controller',root/'.test_output/raw-qualification');tree('original-contexts',root/'.test_output/empty-record-array-contexts')
for name in ('original-contexts-gcc.log','original-contexts-clang.log'):add('post-bootstrap/'+name,root/'.test_output/post-bootstrap'/name)
for label,path in [('raw',repo/'.test_output/generic-array-raw-linux-controller-20261122.py'),('post',repo/'.test_output/generic-array-post-bootstrap-linux-20261122.py'),('archive',repo/'.test_output/archive-generic-array-raw-linux-20261122.py')]:add('observers/'+label+'.py',path)
post=json.loads((root/'.test_output/post-bootstrap/controller.json').read_text());(out/'post-bootstrap-cutoff.json').write_text(json.dumps(dict(scope='I snapshot the two terminal original generic-context cohorts. The complete original route audit is live and is not accepted or archived here.',steps=post['steps'][:2]),indent=2)+'\n')
for path in b.manifest['hosts']:add('hosts/'+str(Path(path).relative_to(root)),Path(path))
for name in ('bin/nano','bin/nanoc_c','bin/nano_virt','bin/nano_vm','bin/nanoisa','bin/nvm2c','bin/nano_aot_runtime.o','bin/nano_as_capture.so','bin/nanoc_seed.nvm','bin/nanoc_stage1.nvm','bin/nanoc_stage2.nvm','bin/nanoc_stage1','bin/nanoc_stage2','bin/nanoc_bootstrap.json','.bootstrap0.built','.bootstrap1.built','.bootstrap2.built','.bootstrap3.built','tests/nanovm/test_vm'):
 add('images/'+name,root/name)
refs=[]
for name in sorted(set(subprocess.check_output(['git','ls-files','src','src_nano','modules','std','stdlib','include','scripts','tests','spec','Makefile.gnu','GNUmakefile'],cwd=root,text=True).splitlines())):
 p=root/name
 if p.is_file():
  raw=p.read_bytes();blob=subprocess.check_output(['git','hash-object','-w',name],cwd=root,text=True).strip();refs.append(dict(path=name,bytes=len(raw),sha256=sha(raw),git_blob=blob))
b.check_inputs();b.check_artifacts(list(b.manifest['artifacts']));assert not subprocess.check_output(['git','status','--porcelain'],cwd=root)
record=dict(version=1,scope='I retain exact806 Linux original full raw Make bootstrap3 and original four-provider GCC13/Clang18 generic-context success. Full original route audit, paired Darwin and later d8efb39fd source acceptance are not claimed.',source_commit='806668bb32326915f5e217026eb95f83d8b35625',symlinks={'bin/nanoc':os.readlink(root/'bin/nanoc')},files=files,git_sources=refs,counts=dict(files=len(files),git_sources=len(refs),raw_bytes=sum(x['bytes'] for x in files),unique_objects=len(objects),compressed_bytes=sum(x['compressed_bytes'] for x in objects.values())),seconds=time.monotonic()-start)
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record['counts']),flush=True)
