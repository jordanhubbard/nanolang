import os,sys,json,pathlib,hashlib,subprocess,shlex,shutil
root=pathlib.Path(sys.argv[1]).resolve();report=pathlib.Path(sys.argv[2]).resolve();report.mkdir(exist_ok=False);os.chdir(root);sys.path.insert(0,str(root))
from tests.test_portable_read_adapters import PortableReadAdapters
h=PortableReadAdapters();h.artifacts=report;h.work=report/'products';h.store=report/'objects';h.work.mkdir();h.store.mkdir();h.index=0
shutil.copyfile(__file__,report/'driver.py')
darwin=sys.platform=='darwin';cc='/usr/bin/clang' if darwin else '/usr/bin/gcc';clang='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/usr/local/bin/clang';sancc=clang if darwin else cc
llvmflags=[] if darwin else ['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
extra={'cc':cc,'clang':clang,'python':sys.executable,'make':'/usr/bin/make','ld':shutil.which('ld'),'ar':shutil.which('ar'),'pkg-config':'/opt/homebrew/bin/pkg-config' if darwin else shutil.which('pkg-config')}
if darwin:
 extra['asan']='/opt/homebrew/Cellar/llvm/23.1.1/lib/clang/23/lib/darwin/libclang_rt.asan_osx_dynamic.dylib';extra['xcrun']='/usr/bin/xcrun'
else:
 for name in ['libasan.so','libubsan.so','cc1','collect2']:
  flag='-print-prog-name=' if name in ('cc1','collect2') else '-print-file-name='
  extra[name]=subprocess.check_output([cc,flag+name],text=True).strip()
assert all(p and pathlib.Path(p).is_file() for p in extra.values()),extra
h.env=dict(os.environ,CC=cc,PORTABLE_ADAPTER_CC=cc,PORTABLE_ADAPTER_CLANG=clang,PORTABLE_ADAPTER_CFLAGS='',PORTABLE_ADAPTER_LLVM_FLAGS=shlex.join(llvmflags),PORTABLE_ADAPTER_EXTRA_TOOLS=json.dumps(extra),PORTABLE_ADAPTER_ARTIFACTS=str(report/'native-adapter'),PORTABLE_READ_CC=cc,PORTABLE_READ_CFLAGS='',PORTABLE_READ_ARTIFACTS=str(report/'query'),ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='',PYTHONDONTWRITEBYTECODE='1')
h.env.pop('PYTHONOPTIMIZE',None)
if darwin:
 h.env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
 h.env['SDKROOT']=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip()
tracked=(root/'.qualification-tracked').read_text().splitlines() if (root/'.qualification-tracked').exists() else subprocess.check_output(['git','ls-files'],text=True).splitlines()
def source():return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in tracked if (root/p).is_file()}
h.inputs=sorted(set([pathlib.Path(p).resolve() for p in extra.values()]+[p for p in (root/'src').rglob('*') if p.suffix in ('.c','.h','.inc')]+[root/'Makefile.gnu',root/'tests/nanoisa/test_managed_strings.c',root/'tests/nanoisa/managed_string_smoke.c',root/'tests/nanoisa/test_managed_module.c',pathlib.Path(__file__).resolve()]))
h.dump('configuration.json',{'tools':extra,'env':{k:v for k,v in h.env.items() if k.startswith('PORTABLE_') or k in ['CC','PATH','SDKROOT','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS']},'scope':'ordinary native adapter and query; GCC/HB ASan UBSan LSan native core/module controls only; no installed CLI/package claim'})
phases=[]
def phase(name,fn):
 before=h.inventory();sb=source();h.dump(name+'-inputs-before.json',before);h.dump(name+'-source-before.json',sb)
 status={'phase':name,'passed':False}
 try:fn();status['passed']=True
 finally:
  h.dump(name+'-provider-products.json',{str(p.relative_to(root)):h.retain(p) for p in sorted((root/'obj').rglob('*')) if p.is_file()})
  after=h.inventory();sa=source();h.dump(name+'-inputs-after.json',after);h.dump(name+'-source-after.json',sa)
  status.update(inputs_equal=before==after,source_equal=sb==sa);phases.append(status);h.dump('phases.json',phases);assert before==after and sb==sa
 print(json.dumps(status),flush=True)
def core():
 flags=['-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all']
 for name,src,testing in [('observed','tests/nanoisa/test_managed_strings.c',True),('ordinary','tests/nanoisa/managed_string_smoke.c',False)]:
  exe=h.work/name;h.command([sancc,*flags,*(['-DNMS_TESTING'] if testing else []),'src/nanoisa/managed_strings.c',src,'-o',exe]);h.command([exe])
 exe=h.work/'module';h.command([sancc,*flags,'tests/nanoisa/test_managed_module.c','-o',exe]);h.command([exe]);h.command([exe,'dispose-first'])
def query():
 h.command(['/usr/bin/make','-f','Makefile.gnu','CC='+cc,'test-portable-read-plan'])
 h.dump('query-prepared-products.json',{str(p.relative_to(root)):h.retain(p) for p in sorted((root/'obj').rglob('*')) if p.is_file()})
phase('managed-native',core)
phase('native-adapter',lambda:h.command([sys.executable,'-m','unittest','-f','-v','tests.test_portable_read_adapters']))
phase('declaration-query',query)
h.dump('summary.json',{'passed':all(x['passed'] for x in phases),'commands':h.index,'phases':phases})
