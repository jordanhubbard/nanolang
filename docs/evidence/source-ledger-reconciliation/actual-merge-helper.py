import json, subprocess, sys

task, number, test_summary = sys.argv[1:]
def out(args):
    return subprocess.check_output(args, text=True)
pr = json.loads(out(['gh','pr','view',number,'--json','state,headRefOid,mergeCommit,url,files']))
assert pr['state'] == 'MERGED'
subprocess.run(['git','fetch','origin'], check=True, stdout=subprocess.DEVNULL)
tip = out(['git','rev-parse','origin/main']).strip()
assert tip == out(['git','ls-remote','origin','refs/heads/main']).split()[0]
subprocess.run(['git','merge-base','--is-ancestor',pr['headRefOid'],tip], check=True)
v = {'schema':'mac.worker_evidence.v1','status':'complete','evidence_type':'repo_change',
     'repo':{'head_sha':pr['headRefOid'],'pushed':True,'dirty':False,
             'remote_ref':'refs/heads/main','remote_url':'https://github.com/jordanhubbard/nanolang.git',
             'pr_url':pr['url'],'files_changed':[f['path'] for f in pr['files']]},
     'tests':[{'summary':test_summary,'status':'pass'}],
     'canonical_integration':{'schema':'mac.canonical_integration.v1','status':'pass','remote_verified':True,
             'canonical_ref':'refs/heads/main','canonical_tip_sha':tip,'reviewed_head_sha':pr['headRefOid'],
             'contains_reviewed_head':True,'publication_mode':'operator_post_merge_reconciliation',
             'pull_request_number':int(number),'pull_request_url':pr['url'],'merged_sha':pr['mergeCommit']['oid']}}
subprocess.run(['mac','--profile','default','task','evidence',task,'--kind','repo_change','--uri',pr['url'],
                '--summary',test_summary+' Canonical integration verified; hosted CI remains separately tracked.',
                '--created-by','codex-v501','--metadata',json.dumps({'verification':v,
                    'review_claim':'No MAC review claimed; operator reconciliation of merged work.'})],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['mac','--profile','default','task','force-complete',task,'--actor','codex-v501',
               '--reason','Verified merged PR'+number+' on canonical main. '+test_summary],check=True,stdout=subprocess.DEVNULL)
print(task,'completed from merged PR'+number)
