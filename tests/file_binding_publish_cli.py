"""I exercise the actual explicit tool; forward source remains unexecuted text."""
import json
import errno
import os
from pathlib import Path
import subprocess
import sys
import time

exe=Path(sys.argv[1]);work=Path(sys.argv[2]);source=Path(sys.argv[3]);repo=Path(__file__).resolve().parents[1]
work.mkdir()
expected=(json.dumps(json.loads(source.read_bytes()),separators=(',',':'),ensure_ascii=False)+'\n').encode()
golden=(repo/'tests/fixtures/nsi_file_binding_expected.nano.txt').read_bytes()
rows=[]
def finished(name,p,out,err,expected_rc):
    status={'argv':p.args,'pid':p.pid,'returncode':None,'timeout':False}
    try:status['returncode']=p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        status['timeout']=True;p.kill()
        try:status['child_returncode']=p.wait(timeout=10)
        except subprocess.TimeoutExpired:status['cleanup_error']='child wait timeout'
    finally:
        out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno());out.close();err.close()
        (work/(name+'-status.json')).write_text(json.dumps(status,indent=2)+'\n')
    accepted=(expected_rc,) if isinstance(expected_rc,int) else expected_rc
    assert not status['timeout'] and status['returncode'] in accepted,(name,status)
    assert (work/(name+'.stdout')).read_bytes()==b'',name
    raw=(work/(name+'.stderr')).read_bytes();assert all(x<128 for x in raw),name
    report=json.loads(raw);rows.append({'name':name,**report});return report

def launch(name,args):
    out=(work/(name+'.stdout')).open('wb');err=(work/(name+'.stderr')).open('wb')
    (work/(name+'-command.json')).write_text(json.dumps({'argv':list(map(str,args))},indent=2)+'\n')
    try:p=subprocess.Popen(list(map(str,args)),stdout=out,stderr=err)
    except OSError as failure:
        out.close();err.close();(work/(name+'-status.json')).write_text(json.dumps({'launch_error':repr(failure)})+'\n');raise
    return p,out,err

def run(name,src,dest,rc=0):return finished(name,*launch(name,[exe,src,'--file-binding-dir',dest]),rc)
def complete(dest):
    assert sorted(p.name for p in dest.iterdir())==['binding.nano','interface.nsi.json']
    assert (dest/'interface.nsi.json').read_bytes()==expected
    assert (dest/'binding.nano').read_bytes()==golden
    assert dest.stat().st_mode&0o077==0
    assert all(p.stat().st_mode&0o077==0 for p in dest.iterdir())

out=work/'published';r=run('publish',source,out);assert r['published'] and r['durable'] and not r['cleanup_pending'];complete(out)
snapshot={str(p):p.read_bytes() for p in out.iterdir()};r=run('repeat',source,out,1);assert r['status']==6 and not r['published'];assert snapshot=={str(p):p.read_bytes() for p in out.iterdir()}
for kind in ('file','empty','nonempty','symlink','dangling'):
    dest=work/('existing-'+kind)
    if kind=='file':dest.write_bytes(b'sentinel')
    elif kind in ('empty','nonempty'):
        dest.mkdir()
        if kind=='nonempty':(dest/'unrelated').write_bytes(b'sentinel')
    else:dest.symlink_to(out if kind=='symlink' else work/'absent')
    before=os.lstat(dest);r=run(kind,source,dest,1);after=os.lstat(dest)
    assert r['status']==6 and not r['published'] and (before.st_dev,before.st_ino,before.st_mode)==(after.st_dev,after.st_ino,after.st_mode)
    if kind=='file':assert dest.read_bytes()==b'sentinel'
    if kind=='nonempty':assert (dest/'unrelated').read_bytes()==b'sentinel'
    if kind in ('symlink','dangling'):assert dest.is_symlink()
parent=work/'parent-link';parent.symlink_to(work,target_is_directory=True);r=run('parent-symlink',source,parent/'refused',1);assert not r['published'] and not (work/'refused').exists()
r=run('parent-symlink-slashes',source,str(parent)+'//refused-extra',1);assert not r['published'] and not (work/'refused-extra').exists()
max_component=work/('v'*255);r=run('maximum-component',source,max_component);assert r['published'];complete(max_component)
for kind in ('symlink','directory','fifo','invalid','oversize'):
    src=work/('input-'+kind);dest=work/('refused-'+kind)
    if kind=='symlink':src.symlink_to(source)
    elif kind=='directory':src.mkdir()
    elif kind=='fifo':os.mkfifo(src)
    elif kind=='invalid':src.write_bytes(b'{}')
    else:src.write_bytes(b' '*(1048576+1))
    r=run('input-'+kind,src,dest,1);assert not r['published'] and not dest.exists()
existing_input_link=work/'input-existing-symlink';existing_input_link.symlink_to(source.resolve());assert existing_input_link.exists()
r=run('input-existing-symlink',existing_input_link,work/'refused-existing-symlink',1);assert not r['published'] and not (work/'refused-existing-symlink').exists()
quoted=work/"quoted ' $(touch injected); utf8-\u00e9"
r=run('quoted-path',source,quoted);assert r['directory'].encode('latin1')==os.fsencode(quoted);complete(quoted);assert not (repo/'injected').exists()
odd=work/os.fsdecode(b"quoted ' $(touch injected); \xff\xc3")
parent_identity=work.stat()
probe={'component_hex':os.fsencode(odd.name).hex(),'parent_dev':parent_identity.st_dev,'parent_ino':parent_identity.st_ino,'mkdir_errno':None,'rmdir_errno':None}
try:
    try:os.mkdir(odd,0o700);probe['mkdir_errno']=0
    except OSError as failure:probe['mkdir_errno']=failure.errno
    if probe['mkdir_errno']==0:
        try:os.rmdir(odd);probe['rmdir_errno']=0
        except OSError as failure:probe['rmdir_errno']=failure.errno;raise
finally:(work/'raw-byte-filesystem-probe.json').write_text(json.dumps(probe,indent=2)+'\n')
assert probe['mkdir_errno'] in (0,errno.EILSEQ),probe
r=run('byte-path',source,odd,0 if probe['mkdir_errno']==0 else 1)
assert r['directory'].encode('latin1')==os.fsencode(odd)
if probe['mkdir_errno']==0:assert r['published'] and r['durable'];complete(odd)
else:
    assert r['status']==5 and r['failed_stage']==7 and r['first_errno']==errno.EILSEQ
    assert not r['published'] and not r['durable'] and not r['cleanup_pending'] and r['cleanup_errno']==0
    assert not os.path.lexists(odd)
assert not (repo/'injected').exists()
longname='x'*4096;r=run('long-input',longname,work/'long-refused',1);assert r['status']==2 and not r['published']
r=finished('usage',*launch('usage',[exe]),2);assert not r['published']
# All children inherit the outer retained runner group. Its timeout cleanup kills
# the complete group; I retain each child's file-backed output/status separately.
concurrent=[];dest=work/'raced'
try:
    for i in range(8):concurrent.append((f'race-{i}',*launch(f'race-{i}',[exe,source,'--file-binding-dir',dest])))
    results=[]
    for name,p,stdout,stderr in concurrent:
        results.append(finished(name,p,stdout,stderr,(0,1)))
    assert sum(r['published'] for r in results)==1
    assert all((r['status']==0 and r['durable']) if r['published'] else r['status']==6 for r in results)
finally:
    for name,p,stdout,stderr in concurrent:
        if p.poll() is None:
            p.kill()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        status_file=work/(name+'-status.json')
        if not status_file.exists():status_file.write_text(json.dumps({'returncode':p.poll(),'cancelled_after_prior_terminal':True})+'\n')
        if not stdout.closed:stdout.flush();os.fsync(stdout.fileno());stdout.close()
        if not stderr.closed:stderr.flush();os.fsync(stderr.fileno());stderr.close()
complete(dest)
# A read-only stderr descriptor rejects the final JSON report after commit.
# I retain the actual process status and inspect committed bytes independently.
report_dest=work/'report-failure';report_status={'returncode':None,'timeout':False}
with (work/'report-failure.stdout').open('wb') as out,open(os.devnull,'rb') as err:
    report_args=list(map(str,[exe,source,'--file-binding-dir',report_dest]))
    (work/'report-failure-command.json').write_text(json.dumps({'argv':report_args,'stderr':'read-only /dev/null'})+'\n')
    p=None
    try:
        p=subprocess.Popen(report_args,stdout=out,stderr=err)
        try:report_status['returncode']=p.wait(timeout=60)
        except subprocess.TimeoutExpired:report_status['timeout']=True
    except OSError as failure:report_status['launch_error']=repr(failure)
    finally:
        if p is not None and p.poll() is None:
            p.kill()
            try:report_status['child_returncode']=p.wait(timeout=10)
            except subprocess.TimeoutExpired:report_status['cleanup_error']='child wait timeout'
        out.flush();os.fsync(out.fileno());(work/'report-failure-status.json').write_text(json.dumps(report_status,indent=2)+'\n')
assert report_status['returncode']==1 and not report_status['timeout'];complete(report_dest)
assert not list(work.glob('.nsi-file-binding-*'))
(work/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS actual publisher CLI',len(rows),'reports and8 competing publishers')
