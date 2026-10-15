from pathlib import Path
import shlex,subprocess,os,sys
work=Path('/private/tmp/nl51-generic')
commands=[shlex.split(x) for x in (work/'context-build.log').read_text().replace('\\\n',' ').splitlines() if x.startswith('/opt/homebrew/opt/llvm/bin/clang ')]
base=commands[0]; base=base[:base.index('-MMD')];base=[x for x in base if x!='-O3']+['-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all']
replace={}
for source in ['src/typechecker.c','src/nanovirt/codegen.c','src/nanoisa/nvm_format.c','src/nanoisa/assembler.c']:
 target=work/('producer-'+Path(source).stem+'.o')
 subprocess.run(base+['-c',source,'-o',str(target)],check=True)
 replace[source.replace('src/','obj/').replace('.c','.o')]=str(target)
link=commands[1];link=[replace.get(x,x) for x in link];link[link.index('-o')+1]=str(work/'producer-asan')
subprocess.run(link+['-fsanitize=address,undefined','-fno-sanitize-recover=all'],check=True)
subprocess.run([sys.executable,'-m','unittest','-v','tests.test_generic_binding_metadata'],env={**os.environ,'NANOLANG_GENERIC_METADATA_COMPILERS':str(work/'producer-asan'),'ASAN_OPTIONS':'detect_leaks=0'},check=True)
