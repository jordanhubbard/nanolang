from pathlib import Path
import shlex,subprocess,os
root=Path.cwd();work=Path('/private/tmp/nl51-purity');compiler='/opt/homebrew/opt/llvm/bin/clang'
flags=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-g']
log=(work/'build.log').read_text()
commands=[shlex.split(line) for line in log.splitlines() if line.startswith(compiler+' ') and ' -o obj/test_advisory_metadata ' in line]
assert len(commands)==1
command=commands[0]
for name in ['nvm_format','assembler']:
 subprocess.run([compiler,'-std=c99','-D_GNU_SOURCE','-Isrc','-Isrc/nanoisa',*flags,'-c',f'src/nanoisa/{name}.c','-o',str(work/(name+'.o'))],check=True)
command[command.index('-o')+1]=str(work/'metadata-asan')
command=[str(work/(Path(arg).name)) if arg in ['obj/nanoisa/nvm_format.o','obj/nanoisa/assembler.o'] else arg for arg in command]
subprocess.run(command+flags,check=True)
subprocess.run([str(work/'metadata-asan')],env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1'},check=True)
alloc=[shlex.split(line) for line in log.splitlines() if line.startswith(compiler+' ') and ' -o obj/test_advisory_metadata_alloc ' in line][0]
alloc[alloc.index('-o')+1]=str(work/'metadata-alloc-asan')
subprocess.run(alloc+flags,check=True)
subprocess.run([str(work/'metadata-alloc-asan')],env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1'},check=True)
