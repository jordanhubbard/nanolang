"""I compare copied current and private span formatters in a fixed alternating replay."""
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
    inputs = [source,HERE/'native-witness.c',HERE/'plan.json',HERE/'original.inc',HERE/'candidate.inc',Path(__file__).resolve(),fixture]
    assert helpers == (HERE/'original.inc').read_text()
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
    for variant in ('original','candidate'):
        for mode,extra in [('sanitized',['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']),('plain',[]),('counted',['-DCOUNT_STDIO'])]:
            name=variant+'-'+mode
            binary=output/name
            selected=['-DTRACE_HELPERS='+json.dumps(str(HERE/(variant+'.inc')))]
            if variant=='candidate': selected+=['-DSPAN_CANDIDATE']
            command('build-'+name,cc+flags+extra+selected+[output/'native-witness.c','-o',binary])
            data=binary.read_bytes()
            assert data[:4] in (b'\xcf\xfa\xed\xfe',b'\x7fELF')
            if mode=='sanitized': assert b'__asan_init' in data and b'__ubsan_handle' in data
            binaries[name]={'path':str(binary),'sha256_before_execution':sha(binary),'magic':data[:4].hex()}
            command('controls-'+name,[binary])
    samples=[]
    assert len(plan['order'])==40 and plan['pairs']==20
    for i,variant in enumerate(plan['order']):
        label=f'sample-{i+1:02}-{variant}'
        with (output/(label+'.jsonl')).open('wb') as stream:
            measurement=json.loads(command(label,[output/(variant+'-plain'),'--replay'],stderr_file=stream))
        assert not measurement['counted'] and measurement['events']==plan['trace_events']
        actual=output/(label+'.jsonl')
        assert actual.read_bytes()==expected
        samples.append({**measurement,'index':i+1,'pair':i//2+1,'variant':variant,'bytes':actual.stat().st_size,'sha256':sha(actual)})
        (output/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    counts={}
    for variant in ('original','candidate'):
        label='counted-'+variant
        with (output/(label+'.jsonl')).open('wb') as stream:
            counts[variant]=json.loads(command(label,[output/(variant+'-counted'),'--replay'],stderr_file=stream))
        assert counts[variant]['counted'] and counts[variant]['events']==plan['trace_events']
        assert (output/(label+'.jsonl')).read_bytes()==expected
    after = {str(p):sha(p) for p in inputs}
    assert before==after and sha(compiler)==compiler_before
    for binary in binaries.values():
        binary['sha256_after_execution']=sha(binary['path'])
        assert binary['sha256_after_execution']==binary['sha256_before_execution']
    def distribution(variant):
        values=sorted(s['wall_seconds'] for s in samples if s['variant']==variant)
        assert len(values)==20
        def percentile(p):
            offset=(len(values)-1)*p;lo=int(offset);hi=min(lo+1,len(values)-1)
            return values[lo]+(values[hi]-values[lo])*(offset-lo)
        return {'count':len(values),'median':statistics.median(values),'p25':percentile(.25),'p75':percentile(.75),'min':min(values),'max':max(values)}
    pairs=[]
    for i in range(20):
        pair={sample['variant']:sample['wall_seconds'] for sample in samples[2*i:2*i+2]}
        assert set(pair)=={'original','candidate'}
        pairs.append({'index':i+1,**pair,'difference_seconds':pair['candidate']-pair['original']})
    report={'plan':plan,'before':before,'after':after,'compiler':{'argv':cc,'observed_executable':str(compiler),'sha256':compiler_before,'version':version},'products':binaries,'expected_trace_bytes':len(expected),'expected_trace_sha256':hashlib.sha256(expected).hexdigest(),'samples':samples,'pairs':pairs,'counted_separate_replays':counts,'distribution':{v:distribution(v) for v in ('original','candidate')},'production_changed':False,'whole_factory_gain_established':False,'scope':'I compare isolated unbuffered regular-file output with actual libc I/O. All 40 replay bytes match. Separate counted images do not supply timings. Native controls test byte output, boundary flush and failed-stream visibility, not equivalence under every partial-write fault, pipes, terminals, whole compiler or complete SDK closure.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'distribution':report['distribution'],'counts':counts,'bytes_per_replay':len(expected)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();run(args.output.resolve())
