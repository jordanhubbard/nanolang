"""I supervise two real standalone native invocations in the runner's process group."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tests.file_native_provider_corpus import setup

def main():
    root,work,compiler=map(Path,sys.argv[1:]);work.mkdir()
    repo,cases,alias=setup(root,work)
    (repo/'src_nano').symlink_to(root/'src_nano',target_is_directory=True)
    (repo/'modules/a/same.c').write_text('int provider_a(void){return 23;}\n')
    (repo/'modules/a/main.nano').write_text('extern fn provider_a() -> int\nshadow provider_a { assert (== (provider_a) 23) }\n')
    (repo/'modules/b/main.nano').write_text('extern fn provider_b() -> int\nshadow provider_b { assert (== (provider_b) 19) }\n')
    source=repo/'main.nano';source.write_text('module "modules/a/main.nano" as A\nmodule "modules/b/main.nano" as B\nfn main() -> int { return (+ (A.provider_a) (B.provider_b)) }\nshadow main { assert (== (main) 42) }\n')
    wrapper=work/'compiler.py'
    wrapper.write_text('#!'+sys.executable+'\nimport json,os,subprocess,sys\nfrom pathlib import Path\nargs=sys.argv[1:]\nfd=os.open(os.environ["PROVIDER_COMMAND_LOG"],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)\nos.write(fd,(json.dumps(args)+"\\n").encode());os.close(fd)\nsys.exit(53 if os.environ.get("PROVIDER_CAPTURE_FINAL")=="1" and "-c" not in args else subprocess.call(json.loads(os.environ["PROVIDER_REAL_CC"])+args))\n')
    wrapper.chmod(0o755)
    real=os.environ.get('NANO_COMPANION_CC','cc')
    import shlex
    processes=[];handles=[];outputs=[];logs=[]
    try:
        for index in (1,2):
            exe=work/f'program-{index}';log=work/f'commands-{index}.jsonl';outputs.append(exe);logs.append(log)
            out=(work/f'invocation-{index}.stdout').open('wb');err=(work/f'invocation-{index}.stderr').open('wb');handles.extend((out,err))
            env=dict(os.environ,NANO_CC=str(wrapper),CC=str(wrapper),PROVIDER_COMMAND_LOG=str(log),PROVIDER_REAL_CC=json.dumps(shlex.split(real)),NANO_CFLAGS=f'-DINVOCATION={index}')
            args=[str(compiler),str(source),'-o',str(exe),'--verbose']
            (work/f'invocation-{index}-command.json').write_text(json.dumps(dict(argv=args,cwd=str(repo),NANO_CFLAGS=env['NANO_CFLAGS']),indent=2)+'\n')
            processes.append(subprocess.Popen(args,cwd=repo,env=env,stdout=out,stderr=err))
        statuses=[p.wait(timeout=900) for p in processes]
        (work/'statuses.json').write_text(json.dumps(statuses)+'\n');assert statuses==[0,0],statuses
    finally:
        for process in processes:
            if process.poll() is None:process.kill();process.wait(timeout=10)
        for handle in handles:handle.flush();os.fsync(handle.fileno());handle.close()
    owners=[]
    for index,(exe,log) in enumerate(zip(outputs,logs),1):
        commands=[json.loads(line) for line in log.read_text().splitlines()]
        providers=[args for args in commands if '-c' in args]
        assert len(providers)==3,providers
        sources=[next(arg for arg in args if arg.endswith('.c')) for args in providers]
        assert len(sources)==len(set(sources))==3
        assert sources.count(str((repo/'modules/a/same.c').resolve()))==1
        objects=[Path(args[args.index('-o')+1]) for args in providers];owners.append(set(map(str,objects)))
        assert len({str(p.parent) for p in objects})==1
        for args in commands:assert f'-DINVOCATION={index}' in args
        for path in objects:assert not path.exists(),path
        final=commands[-1];assert final[final.index('-o')+1]==str(exe)
        for path in objects:assert final.count(str(path))==1
        completed=subprocess.run([exe],timeout=60);assert completed.returncode==42
    assert owners[0].isdisjoint(owners[1])
    # I observe the ACTUAL final driver command for two runtime paths whose
    # realpath is identical. The observer deliberately refuses the final link;
    # this control proves command identity, not executable behavior of aliases.
    (alias/'src_nano').symlink_to(root/'src_nano',target_is_directory=True)
    log=work/'runtime-alias-final.jsonl';exe=work/'runtime-alias-output';exe.write_bytes(b'output sentinel')
    env=dict(os.environ,NANO_CC=str(wrapper),CC=str(wrapper),PROVIDER_COMMAND_LOG=str(log),PROVIDER_REAL_CC=json.dumps(shlex.split(real)),PROVIDER_CAPTURE_FINAL='1',NANO_CFLAGS='')
    with (work/'runtime-alias.stdout').open('wb') as out,(work/'runtime-alias.stderr').open('wb') as err:
        result=subprocess.run([compiler,source,'-o',exe,'--root-shadows-only'],cwd=alias,env=env,stdout=out,stderr=err,timeout=300)
    (work/'runtime-alias-status.json').write_text(json.dumps(dict(returncode=result.returncode,mode='modeled final compiler refusal; actual provider compiles'))+'\n')
    assert result.returncode>0
    assert exe.read_bytes()==b'output sentinel'
    commands=[json.loads(line) for line in log.read_text().splitlines()]
    final=[args for args in commands if '-c' not in args]
    assert len(final)==1,final
    shared=str((root/'src/runtime/list_ASTFloat.c').resolve())
    assert final[0].count(shared)==1
    sources=[arg for arg in final[0] if arg.endswith('.c')]
    assert len(sources)==len(set(sources)),sources
    for args in commands:
        if '-c' in args:assert not Path(args[args.index('-o')+1]).exists()
    print('PASS standalone concurrent private provider ownership')

if __name__=='__main__':main()
