"""I qualify one frozen SDK tree; my caller supplies its exact source manifest."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

from tests.native_sdk_runner import run


def main():
    root,report,host,manifest=sys.argv[1:]
    root=Path(root).resolve();report=Path(report).resolve();report.mkdir(exist_ok=False)
    store=report/'objects';store.mkdir()
    selected=json.loads(Path(manifest).read_text())
    records=selected['records']
    def dump(name,data):(report/name).write_text(json.dumps(data,indent=2)+'\n')
    def hashfile(path,retain=False):
        path=Path(path);h=hashlib.sha256();size=0
        with path.open('rb') as stream:
            for data in iter(lambda:stream.read(1048576),b''):h.update(data);size+=len(data)
        value=dict(sha256=h.hexdigest(),bytes=size,mode=path.stat().st_mode & 0o7777)
        if retain:
            target=store/value['sha256']
            if not target.exists():
                free=shutil.disk_usage(store).free
                if free<size+2*1024**3:
                    dump('capture-capacity-refusal.json',dict(path=str(path),bytes=size,free_bytes=free,reserve_bytes=2*1024**3))
                    raise AssertionError('I preserve my capacity reserve before artifact capture')
                shutil.copyfile(path,target)
            value['archive']=str(target)
        return value
    def sources():
        result={row['path']:hashfile(root/row['path']) for row in records}
        for row in records:
            actual=result[row['path']]
            assert actual['sha256']==row['sha256'] and actual['bytes']==row['bytes'],row['path']
            expected_mode=int(row['mode'],8) & 0o7777 if isinstance(row['mode'],str) else row['mode'] & 0o7777
            assert actual['mode']==expected_mode,row['path']
        return result
    def products():
        return {str(p.relative_to(root)):hashfile(p,True) for name in ('bin','obj','lib')
                for p in sorted((root/name).rglob('*')) if p.is_file()}
    env=dict(LSAN_OPTIONS='',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
             UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    if host=='linux':
        cc=['/usr/bin/gcc'];other=['/usr/local/bin/clang','--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
        env['NMS_NATIVE_CLANG_FLAGS']=other[1]
        configs=[dict(name='gcc-ordinary',cc=cc,sanitizers=False),dict(name='gcc-sanitizers',cc=cc,sanitizers=True),
                 dict(name='clang-ordinary',cc=other,sanitizers=False),dict(name='clang-sanitizers',cc=other,sanitizers=True)]
        flags=[];links=['-lcrypto']
    else:
        cc=['/usr/bin/clang'];other=['/opt/homebrew/opt/llvm/bin/clang']
        env['PATH']='/opt/homebrew/bin:'+os.environ['PATH']
        out,_,_=run(report,'sdk-path',['/usr/bin/xcrun','--show-sdk-path'],root,env)
        env['SDKROOT']=out.decode().strip()
        env['PKG_CONFIG_PATH']='/opt/homebrew/opt/libffi/lib/pkgconfig:/opt/homebrew/opt/openssl@3/lib/pkgconfig'
        env['LIBRARY_PATH']='/opt/homebrew/opt/openssl@3/lib'
        env['DYLD_LIBRARY_PATH']='/opt/homebrew/opt/openssl@3/lib'
        configs=[dict(name='apple-ordinary',cc=cc,sanitizers=False),dict(name='homebrew-ordinary',cc=other,sanitizers=False),
                 dict(name='homebrew-sanitizers',cc=other,sanitizers=True)]
        flags=['-isysroot',env['SDKROOT'],'-I/opt/homebrew/opt/openssl@3/include'];links=['-L/opt/homebrew/opt/openssl@3/lib','-lcrypto']
    env.update(CC=shlex.join(cc),NANO_SDK_CC=shlex.join(cc),NANO_SDK_CFLAGS=shlex.join(flags),
        NANO_SDK_LDFLAGS=shlex.join(links),NANO_SDK_REPORT_DIR=str(report),NANO_SDK_EXPECT_CLEAN='1',
        NANO_SDK_PROBE_CONFIGS=json.dumps(configs),NANO_SDK_MIN_FREE_BYTES=str(2*1024**3))
    tools=[Path(cc[0]).resolve(),Path(other[0]).resolve(),Path(sys.executable).resolve()]
    for name in ('make','ar','ld','nm','pkg-config'):
        path=shutil.which(name,path=env.get('PATH',os.environ['PATH']));assert path,name;tools.append(Path(path).resolve())
    for name,args in (('ordinary-version',cc+['--version']),('other-version',other+['--version'])):
        run(report,name,args,root,env)
    queries=[('clang-asan',other+['-print-file-name='+('libclang_rt.asan.a' if host=='linux' else 'libclang_rt.asan_osx_dynamic.dylib')])]
    if host=='linux':
        queries += [('gcc-asan',cc+['-print-file-name=libasan.so']),('gcc-ubsan',cc+['-print-file-name=libubsan.so']),
                    ('gcc-cc1',cc+['-print-prog-name=cc1']),('clang-ubsan',other+['-print-file-name=libclang_rt.ubsan_standalone.a'])]
    else:queries += [('apple-owner',['/usr/bin/xcrun','--find','clang'])]
    for name,args in queries:
        out,_,_=run(report,name,args,root,env);path=Path(out.decode().strip());assert path.is_file(),(name,path);tools.append(path.resolve())
    def toolmap():return {str(p):hashfile(p) for p in sorted(set(tools))}
    free=shutil.disk_usage(root).free
    dump('capacity.json',dict(free_bytes=free,minimum_bytes=2*1024**3))
    assert free>=2*1024**3,'I retain inputs and stop below the capacity guard'
    before=sources();tb=toolmap();dump('source-before.json',before);dump('tools-before.json',tb);dump('products-before.json',products())
    dump('configuration.json',dict(source=selected,configs=configs,scope='fresh actual clean make install and full ordinary installed compiler corpus; selected two-source SDK probe instrumentation only'))
    original=root.stat();first=None
    try:
        run(report,'installed-sdk',[sys.executable,'-m','unittest','-f','-v','tests.test_native_sdk'],root,env,timeout=21600,track_descendants=True)
    except BaseException as failure:
        first=repr(failure);dump('first-failure.json',dict(error=first));raise
    finally:
        # The fixture temporarily moves ONLY this owning source tree. If a
        # supervisor kills it there, I restore only the exact recorded inode,
        # after the runner has completed bounded process-group cleanup.
        hidden=root.with_name(root.name+'-sdk-hidden')
        terminal=json.loads((report/'installed-sdk-status.json').read_text())
        cleanup_safe=terminal.get('group_absent') and terminal.get('descendants_absent') and not terminal.get('cleanup_errors')
        if not root.exists() and hidden.is_dir() and cleanup_safe:
            current=hidden.stat()
            if (current.st_dev,current.st_ino)==(original.st_dev,original.st_ino):
                hidden.rename(root);dump('source-path-recovery.json',dict(original=str(root),hidden=str(hidden),first_failure=first))
        if not root.exists():
            dump('source-path-recovery-refused.json',dict(original=str(root),hidden=str(hidden),terminal=terminal,first_failure=first))
            raise AssertionError('I retain the moved source until descendant disappearance is established')
        after=sources();ta=toolmap();dump('source-after.json',after);dump('tools-after.json',ta);dump('products-after.json',products())
        artifacts={str(p.relative_to(report)):hashfile(p,True) for p in sorted(report.rglob('*'))
                   if p.is_file() and store not in p.parents}
        dump('artifacts.json',artifacts)
        if before!=after or tb!=ta:raise AssertionError('I observed changed source/tool identity')
    dump('terminal.json',dict(status='PASS',pin=selected['pin'],installed_ordinary=True,probe_configs=configs))


if __name__=='__main__':main()
