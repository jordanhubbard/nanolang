import sys,os,json
from pathlib import Path
root=Path('/tmp/nanolang-record-generated-qualified-4fd32');sys.path.insert(0,str(root));os.chdir(root)
from tests import test_file_cyclic
art=Path('/tmp/nanolang-record-generated-hoist-cost');art.mkdir(exist_ok=False)
t=test_file_cyclic.FileCyclic();t.artifacts=art
os.environ.update(LSAN_OPTIONS='',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
base=['-std=c11','-D_DEFAULT_SOURCE','-O0','-g','-Wall','-Wextra','-Werror','-DNANO_RECORD_ARRAY_GENERATED_PRIVATE','-DNMS_TESTING','-Isrc/nanoisa','-Itests/nanoisa']
configs=[('gcc-ordinary',['/usr/bin/gcc-13'],[]),('gcc-sanitizer',['/usr/bin/gcc-13'],['-fsanitize=address,undefined','-fno-omit-frame-pointer']),('clang-sanitizer',['/usr/local/bin/clang','--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'],['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]
for name,cc,flags in configs:
 runtime=art/(name+'.o');exe=art/name
 t.command(name+'-runtime',cc+base+flags+['-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h','-c','/tmp/record-generated-hoisted-runtime.c','-o',str(runtime)])
 t.command(name+'-build',cc+base+flags+['/tmp/record-generated-init-cost.c','tests/nanoisa/record_array_alloc.c',str(runtime),'-lm','-pthread','-o',str(exe)])
 print(name,t.command(name+'-run',[str(exe)]).decode(),flush=True)
