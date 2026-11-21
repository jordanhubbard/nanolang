from pathlib import Path
import hashlib,json,os,platform,subprocess,sys
root=Path.cwd()
def command(*args):
 return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
source=['Makefile.gnu','spec/nanoisa.yaml','scripts/gen_nanoisa_schema.py','src/nanoisa/isa.c','src/nanoisa/isa.h','src/nanoisa/generated_schema.h','tests/nanolang_translator_routes.py','tests/native_process.py']
source += [str(p) for p in sorted(Path('src_nano/nvm2c').glob('*.nano'))]
source += ['tests/'+p for p in ('nvm2c_instruction_driver.nano','nvm2c_instruction_reference.c','test_nanolang_translator_instructions.py','test_nanolang_translator_container.py','test_nanolang_translator_payloads.py','test_nanolang_translator_module.py')]
tools=['bin/nanoc_c','bin/nano_virt','bin/nano_vm','bin/nvm2c','bin/nano_aot_runtime.o','obj/nvm2c_instruction_reference','obj/nvm2c_module_reference',sys.argv[1]]
data={'component_commit':command('git','rev-parse','HEAD'),'worktree_status':command('git','status','--short'),'platform':platform.platform(),'uname':command('uname','-a'),'cc':command('cc','--version'),'python':sys.version,'cwd':str(root),'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source},'post_run_tool_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in tools},'counts':{'container':165,'payload':1585,'module':264,'instructions':3472,'instruction_explicit':721,'instruction_explicit_integer_bit_window':68,'original_products':6,'widths':256,'file_scans':256,'pending_profiles':5,'routes':5},'limits':['I hash after the run; I do not claim a before/after complete SDK closure.','My unittest products are temporary and are not archived.','I instrument generated C; I use the existing ordinary AOT runtime object.','My reused C-seed-built NanoLang driver does not establish Stage1/Stage2 equality.','I do not qualify extension admission, instruction verification, pure NanoLang C emission, installed products or the full release.']}
if sys.platform=='darwin':
 data['sdk_queries']={}
 for args in [('xcrun','--show-sdk-path'),('xcrun','--show-sdk-version'),('xcrun','--sdk','macosx','--show-sdk-version')]:
  result=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  data['sdk_queries'][' '.join(args)]={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
 data['os_version']=command('sw_vers')
Path(sys.argv[2]).write_text(json.dumps(data,indent=2)+'\n')
