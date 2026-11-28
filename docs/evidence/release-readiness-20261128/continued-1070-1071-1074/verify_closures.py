"""I verify administrative consolidation closures and preserved source branches."""
from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--closures',type=Path,default=Path(__file__).with_name('closures.json'));a=p.parse_args();raw=a.closures.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='3857fe68af5119f77f6dc440d7593576b4144c848a3a63b752580f8a235a3e50'
q=json.loads(raw);assert not any(q[k] for k in ['release_gate_passed','all_69_requirements_complete','product_acceptance_granted','main_merge_granted','stale_label_available'])
proof=json.loads(Path(__file__).with_name('source-proof.json').read_text());heads={r['pr']:r['head'] for r in proof['items']};branches={1070:'fix/5.1-capture-requalification',1071:'fix/5.1-declare-wasmtime',1074:'fix/5.1-reproducible-version'}
assert len(q['items'])==3
for row in q['items']:
 n=row['pr'];assert row['head']==heads[n]==row['terminal']['headRefOid'];assert row['terminal']['state']=='CLOSED' and row['terminal']['mergedAt'] is None
 assert q['preserved_branch_refs']['refs/heads/'+branches[n]]==row['head']
 assert row['comment_url'].startswith('https://github.com/jordanhubbard/nanolang/pull/'+str(n)+'#issuecomment-')
 body=Path(__file__).with_name('pr-'+str(n)+'-consolidation.md').read_text();assert body.startswith('Stale for release v5.1.0: closing because I consolidated this proposal into #1168;') and row['head'] in body and 'grants no product acceptance' in body
print('I verify three annotated consolidation closures, unchanged exact heads and preserved branches. My release gate remains false.')
