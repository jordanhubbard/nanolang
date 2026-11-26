from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess
repo=Path('/var/tmp/nanolang-darwin-publication-qkk55gi3');out=repo/'docs/evidence/release-readiness-20261126';assert not out.exists();out.mkdir();(out/'reviews').mkdir()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
run=lambda *args:subprocess.check_output(list(args),cwd=repo,text=True).strip()
pin=run('git','rev-parse','HEAD');assert pin=='4a1d4e24eeeb134addfc362411ef3bec844e743e'
source_files={
 'open-issues.json':'/var/tmp/nanolang-release-5.1-open-issues-20261126-current.json',
 'open-prs.json':'/var/tmp/nanolang-release-5.1-open-prs-20261126-current.json',
 'pr-bases.json':'/var/tmp/nanolang-release-5.1-pr-bases-20261126-current.json',
 'counts.json':'/var/tmp/nanolang-release-5.1-counts-20261126-current.json',
 'parent-1255.json':'/var/tmp/nanolang-release-5.1-parent-1255-20261126-current.json',
 'parent-1255-terminal.json':'/var/tmp/nanolang-release-5.1-parent-1255-terminal-20261126-current.json',
 'parent-995.json':'/var/tmp/nanolang-release-5.1-parent-995-20261126-current.json',
 'live-refs.txt':'/var/tmp/nanolang-release-5.1-live-refs-20261126.txt',
}
retrieved={}
for name,path in source_files.items():
 p=Path(path);shutil.copy2(p,out/name);retrieved[name]=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()
contract=subprocess.check_output(['git','show',pin+':docs/RELEASE_5.1_SCOPE.md'],cwd=repo);(out/'scope.md').write_bytes(contract)
issues=json.loads((out/'open-issues.json').read_bytes());prs=json.loads((out/'open-prs.json').read_bytes());bases={r['number']:r for r in json.loads((out/'pr-bases.json').read_bytes())};parents=[json.loads((out/f'parent-{n}.json').read_bytes()) for n in (1255,995)]
all_rows={r['number']:r for r in [*issues,*prs,*parents]}
links={1256:1255,1252:976,1251:976,1250:976,1228:1116,1209:976,1140:1116,1127:976,1085:976,1084:976,1082:976,1077:976,1065:976,1056:1039,1055:1039,1054:1039,1050:976,1048:976,1046:976,1044:1039,1042:1039,1041:1039,1004:976,1003:976,
       1141:1139,1132:1133,1126:1116,1122:976,1120:976,1074:1039,1072:1054,1071:1054,1069:995,1066:1065}
pattern=re.compile(r'(?<!\d)v?5\.1(?:\.0)?(?!\d)')
review=[]
for kind,items in [('issue',issues),('pr',prs)]:
 for row in items:
  n=row['number'];body=row['body'];direct=bool(pattern.search(row['title']+'\n'+body)) or any('5.1' in l['name'] for l in row['labels']) or '5.1' in str(row['milestone'])
  item=dict(kind=kind,number=n,url=row['url'],title=row['title'],body_sha256=sha(body.encode()),labels=[l['name'] for l in row['labels']],milestone=row['milestone'],scope_reviewed=True,acceptance_verified=False,action='retain-open')
  if kind=='issue' and n==1089:
   quote=next(p for p in body.split('\n\n') if 'not a new 5.1 release gate' in p);item.update(release_scoped=False,basis='explicit-existing-nongate',evidence=quote)
  elif kind=='issue' and n==1195:item.update(release_scoped=False,basis='separate-fleet-migration',evidence=body.split('\n\n')[0])
  elif direct:
   paragraph=next((p for p in (row['title']+'\n\n'+body).split('\n\n') if pattern.search(p)),None)
   item.update(release_scoped=True,basis='release-text-or-work-branch',evidence=paragraph or str(row['milestone']))
  else:
   parent=links[n];assert str(parent) in body and parent in all_rows
   paragraph=next(p for p in body.split('\n\n') if str(parent) in p)
   item.update(release_scoped=True,basis='linked-release-work',parent=parent,evidence=paragraph)
  if kind=='pr':
   pages=json.loads(Path(f'/var/tmp/nanolang-release-5.1-reviews-20261126/{n}.json').read_bytes());states=[]
   for page in pages:
    for r in page:states.append({key:r[key] for key in ['id','state','commit_id','submitted_at']}|{'user':r['user']['login']})
   (out/f'reviews/{n}.json').write_text(json.dumps(dict(endpoint=f'repos/jordanhubbard/nanolang/pulls/{n}/reviews',page_count=len(pages),reviews=states),indent=2)+'\n')
   item.update(head=row['headRefOid'],head_ref=row['headRefName'],draft=row['isDraft'],base_ref=bases[n]['baseRefName'],reported_base=bases[n]['baseRefOid'],review_decision=row['reviewDecision'],review_states=[r['state'] for r in states])
  review.append(item)
refs=dict(line.split()[::-1] for line in (out/'live-refs.txt').read_text().splitlines());integration=refs['refs/heads/feat/5.1-service-module-integration'];main=refs['refs/heads/main'];assert integration==pin
ancestry=[]
for row in prs:
 for label,target in [('integration',integration),('main',main)]:
  command=['git','merge-base','--is-ancestor',row['headRefOid'],target];r=subprocess.run(command,cwd=repo,capture_output=True);assert r.returncode in (0,1)
  ancestry.append(dict(pr=row['number'],target=label,target_commit=target,head=row['headRefOid'],argv=command,exit=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),is_ancestor=r.returncode==0))
(out/'ancestry.json').write_text(json.dumps(ancestry,indent=2)+'\n')
(out/'scope-review.json').write_text(json.dumps(review,indent=2)+'\n')
report=dict(observed_source=pin,observed_main=main,observed_integration=integration,scope_unchanged=True,headline_entries=69,
 original_headlines=len(re.findall(rb'^- \*\*[^\n]+',contract,re.M)),additional_headlines=len(re.findall(rb'^\*\*Required \(',contract,re.M)),
 open_issues=len(issues),release_scoped_issues=sum(r['kind']=='issue' and r['release_scoped'] for r in review),outside_gate_issues=[r['number'] for r in review if r['kind']=='issue' and not r['release_scoped']],
 open_prs=len(prs),release_scoped_prs=sum(r['kind']=='pr' and r['release_scoped'] for r in review),draft_prs=sum(r['isDraft'] for r in prs),
 approved_reviews=sum(state=='APPROVED' for r in review if r['kind']=='pr' for state in r['review_states']),
 integration_ancestor_heads=sum(r['target']=='integration' and r['is_ancestor'] for r in ancestry),main_ancestor_heads=sum(r['target']=='main' and r['is_ancestor'] for r in ancestry),
 reported_base_mismatches=[r['number'] for r in review if r['kind']=='pr' and r['reported_base']!=refs['refs/heads/'+r['base_ref']]],
 readiness_pass=False,release_complete=False,merged=[],closed=[],scope_moved=[],retrieved_at_utc=retrieved)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'prepare_audit.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(report,indent=2))
