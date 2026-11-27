"""I verify the exact restored roadmap against retained bytes and Git history."""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent);args=p.parse_args()
d=json.loads((args.archive/'identities.json').read_text());sha=lambda x:hashlib.sha256(x).hexdigest()
git=lambda rev:subprocess.check_output(['git','show',rev+':docs/ROADMAP.md'],cwd=args.repo)
old=git(d['parent_commit']);before=git(d['before_commit']);cut=git(d['deletion_commit']);after=gzip.decompress((args.archive/'restored-roadmap.md.gz').read_bytes())
missing=b''.join(old.splitlines(keepends=True)[49:])
for name,value in [('parent',old),('before',before),('missing',missing),('after',after)]:
 assert {'bytes':len(value),'lines':len(value.splitlines()),'sha256':sha(value)}==d[name]
assert old.splitlines(keepends=True)[:49]==cut.splitlines(keepends=True)[:49]
assert len(cut.splitlines())==57 and d['missing']['lines']==18670
entry=d['entry'].encode();anchor=d['anchor'].encode();without=after.replace(entry,b'',1);i=without.index(anchor);assert without[i-len(missing):i]==missing
assert without[:i-len(missing)]+without[i:]==before
for source in [old,before]:
 lines=source.splitlines(keepends=True);i=0
 for line in after.splitlines(keepends=True):
  if i<len(lines) and line==lines[i]:i+=1
 assert i==len(lines)
assert after.count(b'Under #1268')==1
print('I verified all18,670 restored lines, every pre-deletion line and every newer row/status against exact Git bytes.')
