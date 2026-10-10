from pathlib import Path
import json,subprocess,shlex,tempfile
root=Path(tempfile.mkdtemp(prefix='nanolang-file-elf-link-'))
(root/'obj').mkdir();(root/'lib').mkdir()
(root/'consumer.c').write_text('extern void nl_file_source_snapshots_free(void);\nint consumer(void) { nl_file_source_snapshots_free(); return 0; }\n')
(root/'provider.c').write_text('void nl_file_source_snapshots_free(void) {}\n')
records=[]
def run(cmd):
 r=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
 records.append({'command':list(map(str,cmd)),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});return r
cc='/opt/homebrew/opt/llvm/bin/clang';ar='/opt/homebrew/opt/llvm/bin/llvm-ar';ld='/opt/homebrew/bin/ld.lld'
assert run([cc,'--target=aarch64-unknown-linux-gnu','-c','consumer.c','-o','obj/env.o']).returncode==0
assert run([cc,'--target=aarch64-unknown-linux-gnu','-c','provider.c','-o','provider.o']).returncode==0
assert run([ar,'rcs','lib/libnano_compiler_inputs.a','provider.o']).returncode==0
orders={}
for phase in ['before','after']:
 inputs=list(dict.fromkeys(shlex.split(Path('/private/tmp/nanolang-file-link-'+phase+'.txt').read_text())))
 orders[phase]=inputs
 selected=[p for p in inputs if p in ['obj/env.o','lib/libnano_compiler_inputs.a']];assert len(selected)==2
 result=run([ld,'-m','aarch64elf','--warn-backrefs','--fatal-warnings','-e','consumer',*selected,'-o',phase+'.elf'])
 if phase=='before':assert result.returncode!=0 and 'backward reference' in result.stderr,result
 else:assert result.returncode==0,result
assert set(orders['before'])==set(orders['after'])
first_archive=next(i for i,p in enumerate(orders['after']) if p.endswith('.a'))
assert all(p.endswith('.a') for p in orders['after'][first_archive:])
Path('/private/tmp/nanolang-file-elf-link-results.json').write_text(json.dumps({'root':str(root),'commands':records,'orders':orders},indent=2)+'\n')
print('PASS: old order triggers ELF backward-reference refusal; corrected Make inputs link successfully with the same membership.')
