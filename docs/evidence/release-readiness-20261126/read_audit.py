"""I verify the full scope inventory; optional Git replay executes no products."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path);args=p.parse_args();base=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
manifest=json.loads((base/'manifest.json').read_bytes())
for row in manifest['files']:
 raw=(base/row['path']).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256'],row['path']
load=lambda name:json.loads((base/name).read_bytes())
report=load('report.json');issues=load('open-issues.json');prs=load('open-prs.json');review=load('scope-review.json');counts=load('counts.json')['data']['repository']
assert len(issues)==counts['issues']['totalCount']==report['open_issues']==134
assert len(prs)==counts['pullRequests']['totalCount']==report['open_prs']==22
assert len({r['number'] for r in issues})==len(issues) and len({r['number'] for r in prs})==len(prs)
contract=(base/'scope.md').read_bytes();assert len(re.findall(rb'^- \*\*[^\n]+',contract,re.M))==report['original_headlines']==67
assert len(re.findall(rb'^\*\*Required \(',contract,re.M))==report['additional_headlines']==2
assert report['headline_entries']==report['original_headlines']+report['additional_headlines']==69 and report['scope_unchanged']
assert b'not a 5.1 release gate' in contract and b'This changes no other row.' in contract
rows={(kind,r['number']):r for kind,items in [('issue',issues),('pr',prs)] for r in items};seen=set();parents={n:load(f'parent-{n}.json') for n in (995,1255)}
assert len(review)==len(rows)
for item in review:
 key=item['kind'],item['number'];assert key in rows and key not in seen;seen.add(key);row=rows[key]
 assert item['scope_reviewed'] and not item['acceptance_verified'] and item['action']=='retain-open'
 assert item['url']==row['url'] and item['title']==row['title'] and item['body_sha256']==sha(row['body'].encode())
 assert item['labels']==[v['name'] for v in row['labels']] and item['milestone']==row['milestone']
 assert item['evidence'] in row['body'] or item['evidence']==row['title'] or item['evidence']==str(row['milestone'])
 if not item['release_scoped']:
  if item['number']==1089:assert item['basis']=='explicit-existing-nongate' and 'not a new 5.1 release gate' in item['evidence']
  elif item['number']==1195:assert item['basis']=='separate-fleet-migration' and 'fleet task' in item['evidence']
  else:raise AssertionError(item)
 elif item['basis']=='linked-release-work':
  assert str(item['parent']) in item['evidence'] and (('issue',item['parent']) in rows or ('pr',item['parent']) in rows or item['parent'] in parents)
 else:assert re.search(r'(?<!\d)v?5\.1(?:\.0)?(?!\d)',item['evidence']) or '5.1' in str(row['milestone'])
scoped_issues=[r for r in review if r['kind']=='issue' and r['release_scoped']];scoped_prs=[r for r in review if r['kind']=='pr' and r['release_scoped']]
assert len(scoped_issues)==report['release_scoped_issues']==132 and len(scoped_prs)==report['release_scoped_prs']==22
assert sorted(r['number'] for r in review if not r['release_scoped'])==sorted(report['outside_gate_issues'])==[1089,1195]
# Every linked scope path must terminate in explicit release text, with no cycle.
by_number={r['number']:r for r in review}
for item in review:
 if not item['release_scoped']:continue
 visited=set();at=item
 while at['basis']=='linked-release-work':
  parent=at['parent'];assert parent not in visited;visited.add(parent)
  if parent in parents:
   assert re.search(r'5\.1',parents[parent]['title']+'\n'+parents[parent]['body']);break
  at=by_number[parent];assert at['release_scoped']
refs=dict(line.split()[::-1] for line in (base/'live-refs.txt').read_text().splitlines());bases={r['number']:r for r in load('pr-bases.json')};approvals=0;drafts=0;mismatches=[]
for row in prs:
 n=row['number'];item=by_number[n];b=bases[n]
 assert item['head']==row['headRefOid']==b['headRefOid']==refs['refs/heads/'+row['headRefName']]
 assert item['base_ref']==b['baseRefName'] and item['reported_base']==b['baseRefOid']
 if b['baseRefOid']!=refs['refs/heads/'+b['baseRefName']]:mismatches.append(n)
 states=load(f'reviews/{n}.json');assert states['endpoint']==f'repos/jordanhubbard/nanolang/pulls/{n}/reviews' and states['page_count']>=1
 assert item['review_states']==[r['state'] for r in states['reviews']]
 approvals+=sum(r['state']=='APPROVED' for r in states['reviews']);drafts+=row['isDraft']
assert approvals==report['approved_reviews']==0 and drafts==report['draft_prs']==13
assert mismatches==report['reported_base_mismatches']==[1176,1122,1052]
ancestry=load('ancestry.json');assert len(ancestry)==len(prs)*2
assert refs['refs/heads/main']==report['observed_main'] and refs['refs/heads/feat/5.1-service-module-integration']==report['observed_integration']==report['observed_source']
for row in ancestry:
 target=report['observed_'+row['target']];assert row['head']==by_number[row['pr']]['head'] and row['target_commit']==target
 assert row['argv']==['git','merge-base','--is-ancestor',row['head'],target] and row['exit'] in (0,1) and row['is_ancestor']==(row['exit']==0)
 assert not row['stdout'] and not row['stderr']
 if args.repo:
  observed=subprocess.run(row['argv'],cwd=args.repo,capture_output=True);assert observed.returncode==row['exit'] and not observed.stdout and not observed.stderr
assert sum(r['target']=='integration' and r['is_ancestor'] for r in ancestry)==report['integration_ancestor_heads']==18
assert sum(r['target']=='main' and r['is_ancestor'] for r in ancestry)==report['main_ancestor_heads']==0
if args.repo:
 observed=subprocess.check_output(['git','show',report['observed_source']+':docs/RELEASE_5.1_SCOPE.md'],cwd=args.repo);assert observed==contract
terminal=load('parent-1255-terminal.json');assert terminal['number']==1255 and terminal['state']=='CLOSED' and terminal['stateReason']=='COMPLETED'
assert 'c3e710d870456820f72296340fcea69801b42057' in terminal['comments'][-1]['body']
assert not report['readiness_pass'] and not report['release_complete'] and report['merged']==report['closed']==report['scope_moved']==[]
print(json.dumps(dict(verified=True,headline_entries=69,open_issues=134,release_scoped_issues=132,outside_gate_issues=[1089,1195],open_prs=22,approved_reviews=0,readiness_pass=False,release_complete=False,git_ancestry_replayed=bool(args.repo))))
