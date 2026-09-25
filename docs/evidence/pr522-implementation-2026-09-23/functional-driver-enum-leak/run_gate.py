import hashlib,json,os,pathlib,shutil,subprocess,sys,time
root=pathlib.Path('/Users/jordanh/Src/nanolang-pr522-repair')
out=pathlib.Path('/tmp/pr522-enum-leak-instrumented')
label=sys.argv[1]
compiler=root/'bin/nanoc_c';backup=out/'ordinary-nanoc_c'
assert compiler.is_file() and not compiler.is_symlink()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=digest(compiler);source_hash=digest(root/'src/typechecker.c');parser_hash=digest(root/'src/parser.c');shutil.copy2(compiler,backup)
env=os.environ.copy();env.update(CC='/opt/homebrew/opt/llvm/bin/clang',NANO_CC='/opt/homebrew/opt/llvm/bin/clang',NANO_SHADOW_TIMEOUT_SECONDS='60',NANO_CALLBACK_COMPILERS='nanoc_c',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
try:
 shutil.copy2(out/'nanoc_c',compiler);t=time.monotonic()
 with (out/(label+'.log')).open('wb') as f:
  r=subprocess.run([sys.executable,'-m','unittest','-v','tests.test_nanoisa_functional_arrays','tests.test_cseed_single_letter_enums','tests.test_generic_function_values'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 (out/(label+'.json')).write_text(json.dumps({'exit':r.returncode,'seconds':time.monotonic()-t,'compiler_sha256':digest(compiler),'typechecker_sha256':source_hash,'parser_sha256':parser_hash,'env':{k:env[k] for k in ['CC','NANO_CC','NANO_SHADOW_TIMEOUT_SECONDS','ASAN_OPTIONS','UBSAN_OPTIONS']}},indent=2)+'\n')
 print(label,r.returncode,flush=True)
finally:
 shutil.copy2(backup,compiler);assert digest(compiler)==original
