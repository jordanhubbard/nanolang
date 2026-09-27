import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path('/Users/jordanh/Src/nanolang-pr522-repair')
stage=sys.argv[1]
compiler=root/'bin'/('nanoc_'+stage)
results=[]
for mode in ['ordinary','sanitized-products']:
    env={**os.environ,'NANOLANG_SELFHOST_COMPILER':str(compiler)}
    if mode=='sanitized-products':
        env.update(NANO_CC='/opt/homebrew/opt/llvm/bin/clang',NANO_CFLAGS='-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all',NANO_LDFLAGS='-fsanitize=address,undefined',ASAN_OPTIONS='detect_leaks=1:detect_stack_use_after_return=1',UBSAN_OPTIONS='halt_on_error=1')
    log=pathlib.Path('/tmp/pr522-scalar-final-'+stage+'-'+mode+'.log')
    with log.open('w') as output:
        run=subprocess.run(['python3','-m','unittest','-v','tests.test_native_module_linking','tests.test_declared_scalar_source'],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
    results.append({'mode':mode,'exit':run.returncode,'compiler_sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),'environment':{key:env.get(key) for key in ['NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','ASAN_OPTIONS','UBSAN_OPTIONS']},'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
    if run.returncode:break
pathlib.Path('/tmp/pr522-scalar-final-'+stage+'.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(run.returncode)
