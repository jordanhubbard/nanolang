exec(open('/private/tmp/nanolang-self-tail-proof-20261021/run.py').read().split("run('changed-object'")[0])
from tests import test_record_array_vm as support
art=pathlib.Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-self-tail-ejpl9lh1')
providers=list(dict.fromkeys([*support.PROVIDERS,'src/nanoisa/nvm_v2_convert.c','src/nanoisa/nvm_v2_module.c','src/nanoisa/nvm_format.c','src/nanoisa/nvm2c.c']))
names={pathlib.Path(p).stem for p in providers};common=[p for p in objects if pathlib.Path(p).stem not in names]
trace=child/'tests/nanoisa/record_array_alloc.c'
source=work/'old-budget.c'
source.write_text(pathlib.Path('/private/tmp/nanolang-depth-three-budget-before.c').read_text().replace('/Users/jkh/Src/nanolang-51-owned-float-design',str(child))+'\nint g_argc=0; char **g_argv=NULL;\n')
providers.remove('src/nanoisa/managed_array_shapes.c')
args=[cc,*flags,'-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME','-Itests/nanoisa','-DRA_WHITEBOX',str(source),str(trace),*[str(art/(pathlib.Path(p).stem+'.o')) for p in providers],*common,*ld,'-o',str(work/'old-budget')]
r=run_captured(args,cwd=child,timeout=180);assert r.returncode==0,r.stderr
r=run_captured([str(work/'old-budget')],cwd=child,timeout=180);(work/'old-budget.stderr').write_bytes(r.stderr);(work/'old-budget.stdout').write_bytes(r.stdout);print(r.stdout.decode(),r.stderr.decode());assert r.returncode==0,r.returncode

assert (work/"old-budget.stdout").read_text().strip()==pathlib.Path("/private/tmp/nanolang-depth-three-budget-before.log").read_text().strip()
