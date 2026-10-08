import ctypes,hashlib,json,os,subprocess,time
from pathlib import Path
source=Path('/Users/jordanh/Src/nanolang');target=Path('/private/tmp/nanolang-closure-arrays-9790d9a91')
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)
assert git(target,'rev-parse','HEAD').decode().strip()=='9790d9a91aac6b412e48e2f6af04c8f5a6392be1'
assert not git(target,'status','--porcelain')
def tree(root):
 out={}
 for line in git(root,'ls-tree','-r','-z','HEAD','--','docs/evidence').split(b'\0'):
  if line:
   meta,path=line.split(b'\t',1);out[os.fsdecode(path)]=meta
 return out
src=tree(source);dst=tree(target)
lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True);lib.clonefile.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];lib.clonefile.restype=ctypes.c_int
before=subprocess.check_output(['df','-h','/private/tmp'],text=True);count=0;total=0;start=time.monotonic()
for rel,meta in dst.items():
 if src.get(rel)!=meta or not meta.startswith(b'100'):continue
 a=source/rel;b=target/rel
 if not a.is_file() or a.is_symlink() or not b.is_file() or b.is_symlink():continue
 data=a.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 assert blob==meta.decode().split()[2],rel
 temp=b.with_name(b.name+'.cow-dedup-tmp')
 if lib.clonefile(os.fsencode(a),os.fsencode(temp),0):raise OSError(ctypes.get_errno(),str(temp))
 os.replace(temp,b);count+=1;total+=len(data)
assert not git(target,'status','--porcelain')
record={'target':str(target),'pin':git(target,'rev-parse','HEAD').decode().strip(),'files_replaced':count,'logical_bytes':total,'method':'APFS clonefile only for identical tracked Git blob IDs; source bytes verified against blob hash; old checkout clean before and after; host-library paths unchanged','seconds':time.monotonic()-start,'df_before':before,'df_after':subprocess.check_output(['df','-h','/private/tmp'],text=True)}
Path('/private/tmp/nanolang-capture-space-dedup.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
