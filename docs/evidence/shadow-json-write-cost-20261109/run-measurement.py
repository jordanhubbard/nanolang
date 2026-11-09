"""I compile current formatter bodies and retain a fixed native I/O measurement."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(output):
    plan = json.loads((HERE/'plan.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT/plan['source_file']
    current = source.read_bytes()
    assert current == subprocess.check_output(['git', 'show', plan['source_pin']+':'+plan['source_file']], cwd=ROOT)
    text = current.decode()
    helpers = text[text.index(plan['helper_start']):text.index(plan['helper_end_before'])]
    (output/'actual_trace_helpers.inc').write_text(helpers)
    assert (output/'actual_trace_helpers.inc').read_bytes() == helpers.encode()
    fixture = ROOT/plan['trace_fixture']
    text = json.loads(gzip.decompress(fixture.read_bytes()))[plan['trace_attempt']]
    raw_events = [m.group() for m in re.finditer(r'\{"event":"shadow-(?:start|complete)"[^\n]*?\}', text)]
    events = [json.loads(raw) for raw in raw_events]
    assert len(events) == plan['trace_events']
    starts = [e for e in events if e['event']=='shadow-start']
    ends = [e for e in events if e['event']=='shadow-complete']
    assert len(starts) == len(ends) == plan['selected_shadows']
    assert [(e['name'],e['source'],e['line']) for e in starts] == [(e['name'],e['source'],e['line']) for e in ends]
    assert all(e['failures']==0 and math.isfinite(e['elapsed_seconds']) for e in events)
    expected = ('\n'.join(raw_events)+'\n').encode()
    (output/'expected.jsonl').write_bytes(expected)
    def literal(value):
        return '"'+''.join(chr(c) if 32<=c<127 and c not in (34,92) else '\\%03o'%c for c in value.encode())+'"'
    header = 'typedef struct {const char *event,*name,*source;int line;double elapsed;int failures;} TraceEvent;\nstatic const TraceEvent events[] = {\n'
    for event in events:
        header += '{'+','.join([*(literal(event[k]) for k in ('event','name','source')),str(event['line']),repr(float(event['elapsed_seconds'])),str(event['failures'])])+'},\n'
    (output/'trace-events.h').write_text(header+'};\n')
    shutil.copy2(HERE/'native-witness.c', output/'native-witness.c')
    inputs = [source,HERE/'native-witness.c',HERE/'plan.json',Path(__file__).resolve(),fixture]
    before = {str(p):sha(p) for p in inputs}
    cc = shlex.split(os.environ.get('CC') or 'cc')
    compiler = Path(shutil.which(cc[0]) or cc[0]).resolve(strict=True)
    compiler_before = sha(compiler)
    version = subprocess.check_output(cc+['--version'],text=True,timeout=10)
    operations = []
    def command(label, argv, *, stderr_file=None):
        with (output/(label+'.stdout')).open('wb') as out, (output/(label+'.stderr')).open('wb') as err:
            p = subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=out,stderr=err if stderr_file is None else stderr_file,timeout=30)
        operations.append({'label':label,'argv':list(map(str,argv)),'returncode':p.returncode})
        (output/'operations.json').write_text(json.dumps(operations,indent=2)+'\n')
        if p.returncode:
            raise RuntimeError('I retain the failed native operation '+label+'; I do not continue measurement.')
        return (output/(label+'.stdout')).read_text()
    flags = ['-std=c99','-O3','-Wall','-Wextra','-Werror','-UNDEBUG']
    binaries = {}
    for name,extra in [('sanitized',['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']),('plain',[]),('counted',['-DCOUNT_STDIO'])]:
        binary = output/name
        command('build-'+name,cc+flags+extra+[output/'native-witness.c','-o',binary])
        assert binary.read_bytes()[:4] in (b'\xcf\xfa\xed\xfe',b'\x7fELF')
        binaries[name] = {'path':str(binary),'sha256_before_execution':sha(binary),'magic':binary.read_bytes()[:4].hex()}
        command('controls-'+name,[binary])
    samples = []
    # I use no I/O-count wrappers in timed samples.
    for i in range(plan['samples']):
        label = f'sample-{i+1:02}'
        with (output/(label+'.jsonl')).open('wb') as stream:
            measurement = json.loads(command(label,[output/'plain','--replay'],stderr_file=stream))
        assert not measurement['counted'] and measurement['events']==plan['trace_events']
        actual = output/(label+'.jsonl')
        assert actual.read_bytes() == expected
        samples.append({**measurement,'index':i+1,'bytes':actual.stat().st_size,'sha256':sha(actual)})
        (output/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    with (output/'counted.jsonl').open('wb') as stream:
        counts = json.loads(command('counted',[output/'counted','--replay'],stderr_file=stream))
    assert counts['counted'] and counts['events']==plan['trace_events']
    assert (output/'counted.jsonl').read_bytes()==expected
    after = {str(p):sha(p) for p in inputs}
    assert before==after and sha(compiler)==compiler_before
    for binary in binaries.values():
        binary['sha256_after_execution']=sha(binary['path'])
        assert binary['sha256_after_execution']==binary['sha256_before_execution']
    values = sorted(s['wall_seconds'] for s in samples)
    def percentile(p):
        offset = (len(values)-1)*p; lo = int(offset); hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(offset-lo)
    report = {'plan':plan,'before':before,'after':after,'helper_sha256':sha(output/'actual_trace_helpers.inc'),
        'compiler':{'argv':cc,'observed_executable':str(compiler),'sha256':compiler_before,'version':version},
        'products':binaries,'expected_trace_bytes':len(expected),'expected_trace_sha256':hashlib.sha256(expected).hexdigest(),
        'samples':samples,'counted_separate_replay':counts,
        'distribution':{'count':len(values),'median':statistics.median(values),'p25':percentile(.25),'p75':percentile(.75),'iqr':percentile(.75)-percentile(.25),'min':min(values),'max':max(values)},
        'scope':'I mechanically copy current bodies into an isolated native witness. Timed images use real libc I/O without count wrappers; a separate image counts calls. I check actual bytes after every replay and all input/tool/image hashes before/after. This is regular-file micro-measurement, not whole-compiler timing, complete toolchain closure or release admission.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'distribution':report['distribution'],'counts':counts,'bytes_per_replay':len(expected)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();run(args.output.resolve())
