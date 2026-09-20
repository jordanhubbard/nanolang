import hashlib,json,os,pathlib,shutil,sys,time
root=pathlib.Path('/home/jkh/Src/nanolang-file-source-diagnostic-74c')
report=pathlib.Path('/tmp/nanolang-file-source-diagnostic-74c');report.mkdir(exist_ok=False)
sys.path.insert(0,str(root))
from tests.test_file_source_plan import FileSourcePlan
FileSourcePlan.work=report
source=pathlib.Path('/tmp/nanolang-file-source-74c-linux/gcc-ordinary/nano-file-source-plan-6ie1lq1a/corpus.nano')
shutil.copy2(source,report/'corpus.nano')
paths=json.loads(pathlib.Path('/tmp/nanolang-file-source-32ade-inputs.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():return {name:digest(root/name) for name in paths}
def tools():return {str(p):digest(p.resolve()) for p in [root/'bin/nanoc_c',pathlib.Path('/bin/stdbuf'),pathlib.Path('/usr/libexec/coreutils/libstdbuf.so'),pathlib.Path('/usr/bin/gcc'),pathlib.Path(sys.executable)]}
def dump(name,value):(report/name).write_text(json.dumps(value,indent=2)+'\n')
dump('source-before.json',sources());dump('tools-before.json',tools());dump('input.json',{'original':str(source),'original_sha256':digest(source),'copied_sha256':digest(report/'corpus.nano'),'scope':'one root-authorized diagnostic; unchanged complete fixture and existing ten-second shadow supervisor; not qualification'})
start=time.monotonic()
try:
 FileSourcePlan.command('unbuffered',['/bin/stdbuf','-o0','-e0',root/'bin/nanoc_c',report/'corpus.nano','-o',report/'program','--verbose','--llm-shadow-json',report/'shadows.json'],timeout=120,extra={'CC':'/usr/bin/gcc','NANO_SHADOW_TRACE':'1','NMS_NATIVE_CLANG_FLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'})
 outcome='success'
except AssertionError:outcome='compiler_failure'
dump('source-after.json',sources());dump('tools-after.json',tools());dump('diagnostic-terminal.json',{'outcome':outcome,'seconds':time.monotonic()-start})
print(outcome,flush=True)
