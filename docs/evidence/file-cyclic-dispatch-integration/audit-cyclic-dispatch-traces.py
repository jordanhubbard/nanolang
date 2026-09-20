import pathlib,json,hashlib,re
r=pathlib.Path('/home/jkh/Src/nanolang-file-cyclic-dispatch-plan/docs/evidence/file-cyclic-dispatch');store=json.loads((r/'artifact-store.json').read_text());configs=[]
def blob(row):return pathlib.Path(store[row['sha256']]['path']).read_bytes()
for group,phases in [('6a65-linux',['ordinary','clang-ordinary','sanitizer','clang-sanitizer']),('100c-darwin',['ordinary','clang-ordinary','sanitizer'])]:
 for phase in phases:
  m=json.loads((r/group/(phase+'-artifacts.json')).read_text())
  def one(suffix):
   rows=[v for n,v in m.items() if n.endswith('/'+suffix)];assert len(rows)==1,(group,phase,suffix,len(rows));return blob(rows[0])
  nested=0
  for n,v in m.items():
   if '/nano-file-cyclic-dispatch-' in n and n.endswith('-status.json'):
    s=json.loads(blob(v));assert s['returncode']==0 and s['leader_reaped'] and s['group_disappeared'] and not s['timeout'] and not s['errors'] and not s['cleanup_signals'],(n,s);nested+=1
  modes=[]
  for mode in ['instrumented','linked']:
   stdout=one(mode+'-capture-run-stdout.log');assert b'PASS cyclic VM capture: 23 exact modules' in stdout
   traces=[l for l in stdout.splitlines() if l.startswith(b'TRACE ')];assert len(traces)>=28
   for opt in ['O0','O2']:
    replay=one(mode+'-'+opt+'-run-stdout.log');assert [l for l in replay.splitlines() if l.startswith(b'TRACE ')]==traces;assert b'PASS cyclic native replay:' in replay
   gen=json.loads(one(mode+'-generated-sha256.json'));assert len(gen)==22
   for name,h in gen.items():
    assert name in m and m[name]['sha256']==h;code=blob(m[name]);assert b'nf_function_' in code and b'nf_label_0:' in code and b'nvm_file_runtime_cyclic_enter(c)' in code
    assert b'nvm_file_vm_cyclic_execute' not in code and b'fvm_step' not in code
   modes.append({'mode':mode,'traces':len(traces),'modules':23,'generated':22})
  for kind in ['plain','abi_revision','abi_size','later_variant','reference','edge','dead_label']:
   for opt in ['O0','O2']:
    syms=one('linked-'+kind+'-'+opt+'-symbols-stdout.log');assert not re.search(rb'\b_?(nvm_file_vm_cyclic_execute|vm_execute|vm_core_execute|nvm_file_vm_execute)\b',syms)
  iso=json.loads(one('linked-isolated-objects.json'));assert not any(pathlib.Path(x).name in ['vm.o','file_vm_cyclic_private.o','file_public_vm.o'] for x in iso)
  configs.append({'group':group,'phase':phase,'modes':modes,'nested_success_statuses':nested,'isolated_pairs':14})
identity=json.loads((r/'nanolang-file-cyclic-dispatch-generated-identity.json').read_text());found=[]
for group in ['100c-linux','6a65-linux']:
 m=json.loads((r/group/'ordinary-artifacts.json').read_text());rows=[blob(v) for n,v in m.items() if n.endswith('/instrumented-generated-sha256.json')];assert len(rows)==1
 actual={pathlib.Path(n).name:h for n,h in json.loads(rows[0]).items()};assert actual==identity['sha256_by_name'];found.append(actual)
assert found[0]==found[1]
pathlib.Path('/tmp/cyclic-dispatch-trace-audit.json').write_text(json.dumps({'configs':configs,'generated_first_corrected_equal':22},indent=2)+'\n');print(json.dumps(configs),flush=True)
