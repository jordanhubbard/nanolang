"""I restore only the actual previously qualified compiler and its inputs."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, platform, subprocess, zipfile

REVISION = 'ddd3b5a4ff35678d1906c07c08fc6bbc39b041de'
ARCHIVE_SHA256 = '13f942d8c634b38cbe38b3c9f8d81ed41e6308ebd86f220d42159baad49c5b43'

def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('I refuse an unsafe retained path.')
    return path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--admission', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and platform.machine() == 'x86_64'
    sha = lambda value: hashlib.sha256(value).hexdigest()
    with a.archive.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda:stream.read(2*1024*1024), b''): h.update(block)
    assert h.hexdigest() == ARCHIVE_SHA256 and a.archive.stat().st_size == 125166776
    workspace = a.workspace.resolve()
    with zipfile.ZipFile(a.archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and z.testzip() is None
        for name in names: safe_name(name)
        prefix = 'nanolang/nanolang/'
        q = json.loads(z.read(prefix+'bin/nanoc_bootstrap.json'))
        assert q['root'] == str(workspace)
        assert q['phase'] == 'stage2' and q['raw_stage1_stage2_equal']
        assert q['stage_timeout_seconds'] == 1800 and q['shadow_timeout_seconds'] == 30
        assert len(q['steps']) == 17 and all(s['exit_code'] == 0 for s in q['steps'])
        wanted = {'bin/nanoc_stage2','bin/nanoc_bootstrap.json','modules/index.json'}
        wanted.update('bin/'+n for n in q['tools'])
        for name in names:
            if not name.startswith(prefix) or name.endswith('/'): continue
            relative = name[len(prefix):]
            if relative not in wanted and not relative.startswith('obj/module_cache/'): continue
            safe_name(relative)
            path = workspace/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(z.read(name))
            if path.parent == workspace/'bin' and path.suffix not in ('.json','.o','.so'):
                path.chmod(0o755)
        sources = list(q['sources'])
        request = ''.join(REVISION+':'+name+'\n' for name in sources).encode()
        data = subprocess.check_output(['git','cat-file','--batch'], cwd=a.repo, input=request)
        offset = 0
        for name in sources:
            parts = safe_name(name).parts
            assert parts[0] not in ('product','observer','.git','evidence')
            end = data.index(b'\n',offset);header=data[offset:end].split()
            if header[-1] == b'missing':
                assert name == 'modules/index.json'
                blob = z.read(prefix+name);offset=end+1
            else:
                assert header[1] == b'blob'
                size=int(header[-1]);blob=data[end+1:end+1+size];offset=end+size+2
            assert sha(blob) == q['sources'][name]
            path=workspace/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
        assert offset == len(data)
        for name,digest in q['tools'].items(): assert sha((workspace/'bin'/name).read_bytes()) == digest
        for name,digest in q['hosts'].items():
            assert Path(name).is_relative_to(workspace)
            assert sha(Path(name).read_bytes()) == digest
        native=workspace/'bin/nanoc_stage2';image=native.read_bytes()
        assert sha(image) == q['artifacts']['nanoc_stage2']
        assert image[:6] == b'\x7fELF\x02\x01' and int.from_bytes(image[18:20],'little') == 62
        record=dict(qualification_execution=False,whole_release=False,source_provider=REVISION,
            archive_sha256=ARCHIVE_SHA256,original_run=46119624230,original_job=140901037401,
            original_artifact=15254880555,actual_native=str(native),actual_native_sha256=sha(image),
            actual_hosts=q['hosts'],source_inputs={str(workspace/n):h for n,h in q['sources'].items()},
            tools={str(workspace/'bin'/n):h for n,h in q['tools'].items()},
            receipt=str(workspace/'bin/nanoc_bootstrap.json'),receipt_sha256=sha((workspace/'bin/nanoc_bootstrap.json').read_bytes()),
            subject_stage_alias_created=False)
        a.admission.write_text(json.dumps(record,indent=2)+'\n')

if __name__ == '__main__': main()
