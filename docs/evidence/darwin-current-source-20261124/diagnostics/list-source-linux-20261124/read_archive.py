"""I verify three Linux provider profiles and retain the original shadow timeout."""
from pathlib import Path
import sys,json,subprocess,hashlib
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'edbdb38eab6c9679f36a3c04ad7e4af38768f8972f1882c28e2ebe96204fe6f6')
q=js('journal/result.json');f=js('journal/source-followup-recovered/result.json')
assert not m['full_source_accepted'] and not m['whole_release']
assert m['source_pin']==q['source_pin']==f['source_pin']=='5bf50072752f6e88bad274058663e6e7effc1f89'
assert q['tools']==q['tools_after']==f['tools_before']==f['tools_after']
assert f['terminal'] and not f.get('accepted') and not f['ordinary_ownership_enabled'] and not f['whole_release']
assert [p['mode'] for p in q['profiles']]==[p['mode'] for p in f['private_profiles']]==['normal','san','production']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for p,private in zip(q['profiles'],f['private_profiles']):
    mode=p['mode'];assert not any(p['initial_outputs'].values())
    assert p['before']==p['after_build']==p['after_focused']==private['before']==private['after']
    assert p['build']['exit']==p['focused']['exit']==0 and not p['build']['timed_out']
    assert p['build']['timeout']==1800 and p['focused']['outer_timeout']==600
    assert p['build']['args'][-4:]==['bin/nanoc_c','bin/nano_vm','bin/nvm2c','bin/nanoisa']
    assert 'test-module-list-preparation' in p['focused']['args'] and 'test-module-rejection-rollback' in p['focused']['args']
    for name,h in p['before'].items():
        assert rows[mode+'/'+name]['sha256']==h
        proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert head[1]==b'blob';raw=proc.stdout.read(int(head[2]));assert proc.stdout.read(1)==b'\n';assert hashlib.sha256(raw).hexdigest()==h
    for name,info in p['outputs'].items():assert rows[mode+'/'+name]['sha256']==info['sha256'] and rows[mode+'/'+name]['bytes']==info['bytes']
    assert rows[mode+'/bin/nanoc_c']['sha256']==p['image_sha256']
    for name in ['nanoc_c','nano_vm','nvm2c','nanoisa','nanolang_private']:
        head=data(mode+'/bin/'+name)[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
    tests=private['tests'];assert (tests['tests'],tests['failures'],tests['errors'],tests['skips'],tests['calls'])==(41,0,0,0,62)
    assert rows[mode+'/bin/nanolang_private']['sha256']==private['image_sha256']
    calls=js('journal/source-followup-recovered/'+mode+'-private-cases/calls.json');assert len(calls)==62 and all(c['exit']==0 and c['timeout']==30 for c in calls)
proc.stdin.close();assert proc.wait()==0
for name,h in q['tools'].items():assert rows['tool/'+name.lstrip('/')]['sha256']==h
assert rows['normal/bin/nanoc_private']['sha256']==f['private_compiler_sha256']
assert len(f['source_stages'])==1;stage=f['source_stages'][0];command=stage['command']
assert stage['name']=='ordinary' and stage['after']==q['profiles'][0]['before']
assert command['exit']==1 and not command['timed_out'] and command['timeout']==600 and command['seconds']<600
assert (stage['starts'],stage['completions'])==(1071,1070) and not stage['failed_shadows']
events=[]
for line in data('journal/source-followup-recovered/ordinary-source.stderr').splitlines():
    try:events.append(json.loads(line))
    except ValueError:pass
starts=[e for e in events if e.get('event')=='shadow-start'];ends=[e for e in events if e.get('event')=='shadow-complete'];assert len(starts)==1071 and len(ends)==1070 and all(e['failures']==0 for e in ends)
assert starts[-1]['name']=='nisa_emit_flow'
supervision=[e for e in events if e.get('event')=='cseed-shadow-phase' and e.get('phase')=='supervisor-status'];assert len(supervision)==1
assert supervision[0]['deadline_seconds']==10 and supervision[0]['parent_timeout'] and supervision[0]['signal']==14 and not supervision[0]['completed']
assert 'journal/source-followup-recovered/ordinary-source/seed' not in rows
assert not any(c['label']=='private-source' for c in f['commands'])
assert js('journal/followup-setup-failure.json')['native_commands_started']==0
assert data('journal/source-followup-recovered/observe-repl.py')==data('journal/observe-repl.py')
print('PASS integrity: %d records; fresh normal/SAN/O3+focused/private41 pass. Ordinary source FAIL at10-second supervisor deadline after1070 completions; private source NOT RUN. Release remains open.'%len(rows))
