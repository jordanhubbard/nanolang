"""I audit retained comparison bytes without executing retained native products."""
import gzip, hashlib, io, json, re, statistics, subprocess, sys, zipfile
from pathlib import Path
archive, destination, run_id, artifact_id = sys.argv[1:]
out = Path(destination); out.mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(archive)
assert z.testzip() is None
prefix = 'cseed-source-comparison/'
read = lambda name: json.loads(z.read(prefix+name))
sha = lambda b: hashlib.sha256(b).hexdigest()
def write(name, obj):
    (out/name).write_text(json.dumps(obj, indent=2)+'\n')
def packed(name, obj):
    (out/name).write_bytes(gzip.compress(json.dumps(obj).encode(), mtime=0))
s = read('summary.json'); plan = read('plan.json')
assert s['plan'] == plan
assert plan['baseline'] == '48016fec4bc56daffa2d90a60de36399c4f9b453'
assert plan['candidate'] == '6c97df51bd8c98e5730bf10da5193b8059d8e55a'
assert plan['order'] == ['original','modified','modified','original','original','modified']
assert plan['shadow_timeout_seconds'] == 10 and plan['native_build_budget'] == 600
assert s['comparison_valid'] and s['all_planned_samples_finished']
assert len(s['samples']) == 6
providers = read('providers-before.json'); assert providers == read('providers-after.json')
shared = read('shared-inputs.json')
# I independently read each recorded source from both exact Git revisions.
for rev in (plan['baseline'], plan['candidate']):
    proc = subprocess.Popen(['git','cat-file','--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for name, digest in shared.items():
        proc.stdin.write((rev+':'+name+'\n').encode()); proc.stdin.flush()
        header = proc.stdout.readline().split(); assert len(header) == 3 and header[1] == b'blob'
        data = proc.stdout.read(int(header[2])); assert proc.stdout.read(1) == b'\n'
        assert sha(data) == digest, (rev, name)
    proc.stdin.close(); assert proc.wait() == 0
products = []
for variant in ('original','modified'):
    data = z.read(prefix+'providers/'+variant+'/nanoc_c')
    assert sha(data) == providers[variant]['bin/nanoc_c']
    assert data[:4] in (b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf')
    products.append({'path':'providers/'+variant+'/nanoc_c','sha256':sha(data),'magic':data[:4].hex()})
selection = None
selections = []
for i, sample in enumerate(s['samples'], 1):
    variant = plan['order'][i-1]; assert sample['index'] == i and sample['variant'] == variant
    root = plan['worktrees'][variant]; base = f'samples/{i:02}-{variant}/'
    inp = read(base+'inputs.json'); assert inp['before'] == inp['after']
    for path, digest in inp['before'].items():
        if path.startswith(root+'/bin/'):
            assert providers[variant][path[len(root)+1:]] == digest
        else:
            assert shared[path[len(root)+1:]] == digest
    events = [json.loads(m.group()) for m in re.finditer(r'\{"event":"shadow-(?:start|complete)"[^\n]*?\}', z.read(prefix+base+'source-build.stderr').decode())]
    starts = [e for e in events if e['event']=='shadow-start']; ends = [e for e in events if e['event']=='shadow-complete']
    identity = lambda e: [e['name'], e['source'].replace(root+'/', ''), e['line']]
    selected = list(map(identity, starts))
    selections.append(selected)
    assert list(map(identity, ends)) == selected[:len(ends)]
    trace = sample['trace']
    assert selected == trace['selection_prefix']
    assert len(starts) == trace['starts'] and len(ends) == trace['completes']
    assert sum(e['failures'] for e in ends) == trace['failures'] == 0
    assert abs(sum(e['elapsed_seconds'] for e in ends)-trace['summed_body_seconds']) < 1e-10
    if selection is None or len(selected) > len(selection): selection = selected
    assert selected == selection[:len(selected)]
    if sample['creation_succeeded']:
        assert len(starts) == len(ends) == 1286 and sample['process']['returncode'] == 0
        data = z.read(prefix+base+'compiler')
        assert sha(data) == inp['product_before_run'] == inp['product_after_run'] == sample['product_sha256']
        assert data[:4] in (b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf')
        products.append({'path':base+'compiler','sha256':sha(data),'magic':data[:4].hex()})
assert all(a == selection[:len(a)] for a in selections)
pairs = []
for i in range(0,6,2):
    group = {a['variant']:a for a in s['samples'][i:i+2]}
    a,b = [group[v]['trace']['summed_body_seconds'] for v in ('original','modified')]
    comparable = all(a['creation_succeeded'] for a in group.values())
    wa,wb = [group[v]['process']['wall_seconds'] for v in ('original','modified')]
    pairs.append({'pair':i//2+1,'original_body_seconds':a,'modified_body_seconds':b,
                  'both_creations_complete':comparable,
                  'modified_body_change_percent':(b/a-1)*100 if comparable else None,
                  'original_factory_wall_seconds':wa,'modified_factory_wall_seconds':wb,
                  'modified_factory_wall_change_percent':(wb/wa-1)*100 if comparable else None})
def distribution(values):
    values = sorted(values)
    def percentile(p):
        point = (len(values)-1)*p
        lo = int(point); hi = min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(point-lo)
    return {'count':len(values),'median':statistics.median(values),
            'p25':percentile(.25),'p75':percentile(.75),'iqr':percentile(.75)-percentile(.25),
            'min':min(values),'max':max(values)} if values else None
analysis = {'scope':'I compare only complete paired selections. Partial timeout body sums are censored, not faster/slower complete workloads. I retain all failed attempts. Whole-factory wall times include other phases and remain separate from shadow-body times.',
            'pairs':pairs,'all_creations_passed':s['all_creations_passed'],
            'shared_git_verified_inputs':len(shared),'selection_count':len(selection),
            'completed_sample_distributions':{v:{
                'body_seconds':distribution([a['trace']['summed_body_seconds'] for a in s['samples'] if a['variant']==v and a['creation_succeeded']]),
                'factory_wall_seconds':distribution([a['process']['wall_seconds'] for a in s['samples'] if a['variant']==v and a['creation_succeeded']])}
                for v in ('original','modified')}}
write('summary.json',s); write('analysis.json',analysis); write('verified-native-products.json',products)
write('artifact.json',{'run_id':run_id,'artifact_id':artifact_id,'zip_bytes':Path(archive).stat().st_size,'zip_sha256':sha(Path(archive).read_bytes()),'full_zip_crc_checked':True})
packed('inventory.json.gz', {n:{'bytes':z.getinfo(n).file_size,'sha256':sha(z.read(n))} for n in z.namelist()})
texts = {}
for n in z.namelist():
    if n.endswith(('.json','.jsonl','.log','.stderr','.stdout','.c')):
        texts[n] = z.read(n).decode()
packed('texts.json.gz',texts)
print(json.dumps(analysis,indent=2))
