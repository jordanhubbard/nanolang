"""I verify my retained C dispatch controls without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='9b71b34ed60fa0774920546ceb19daec686ba8c1dd487c5936c1911849e109db'
manifest=json.loads(raw)
assert not manifest['qualification'] and not manifest['whole_release']
with tempfile.TemporaryDirectory(prefix='namespace-admission-reader-', dir=args.workspace) as temporary:
    root = Path(temporary)
    used, seen = set(), set()
    for record in manifest['files']:
        relative = Path(record['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert str(relative) not in seen
        seen.add(str(relative))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with target.open('wb') as stream:
            for key in record['chunks']:
                assert re.fullmatch('[a-f0-9]{64}', key)
                packed = (args.archive / 'objects' / (key + '.gz')).read_bytes()
                metadata = manifest['chunks'][key]
                assert sha(packed) == metadata['gzip_sha256']
                block = gzip.decompress(packed)
                assert sha(block) == key and len(block) == metadata['bytes']
                digest.update(block)
                size += len(block)
                stream.write(block)
                used.add(key)
        assert size == record['size'] and digest.hexdigest() == record['sha256']
    assert used == set(manifest['chunks'])
    load=lambda name:json.loads((root/name).read_text())
    for name in ['broad','provider-controls','sanitized','original-emitter']:
        d=load(name+'/result.json')
        assert d['exit']==0 and d.get('timeout') is not True and d['guards_match']
        assert d['source_before']==d['source_after']
        if 'tools_before' in d: assert d['tools_before']==d['tools_after']
    d=load('original-emitter/result.json')
    assert d['timeout'] is False and d['seconds']<300
    assert d['argv'][1].endswith('/tests/selfhost_shadow_emitter.nano')
    def cases(name):
        rows=[json.loads(line) for line in (root/name).read_text().splitlines() if line.startswith('{"case"')]
        assert len(rows)==13 and len({row['case'] for row in rows})==13
        return {row['case']:row for row in rows}
    baseline=cases('baseline-final/run.stdout');candidate=cases('provider-controls/stdout')
    assert candidate==cases('sanitized/stdout')
    for name,row in baseline.items():
        other=candidate[name]
        assert row['value']==other['value'] and row['side_effects']==other['side_effects']
    for name in ['string-length','byte-access','absolute-value','array-length','integer-text','empty-print']:
        assert baseline[name]['evaluator_lookups']==2 and candidate[name]['evaluator_lookups']==1
    assert baseline['argument-order-once']['evaluator_lookups']==4
    assert candidate['argument-order-once']['evaluator_lookups']==2
    for name in ['ordinary-function','declared-builder','declared-push','bstring-prefixed-function','ordinary-builtin-suffix','bstring-provider']:
        assert baseline[name]['evaluator_lookups']==candidate[name]['evaluator_lookups']
    d=load('mutation-context-override/result.json')
    assert d['mutation_caught'] and d['product_source_unchanged']
    assert [(row['label'],row['exit']) for row in d['operations']]==[('compile',0),('link',0),('run',1)]
    assert 'runtime error: member access within misaligned address' in (root/'mutation-context-override/run.stderr').read_text()
    assert not load('mutation-context-only-not-distinguishing/result.json')['mutation_caught']
    source=(root/'candidate-source/src/eval.c').read_text()
    assert 'case EVAL_CALL_array_push:' in source and 'Function *named_func = NULL;' in source
    assert 'test-eval-builtin-resolution' in (root/'candidate-source/.github/workflows/c-repair-original-gates.yml').read_text()
print('I verified thirteen paired semantic cases, lookup observations, sanitizer/negative controls and the original local emitter pass; full qualification remains open.')
