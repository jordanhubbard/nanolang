from pathlib import Path
import sys,json
p=Path(sys.argv[1]); rows=[]
for line in (p/'full-compiler.log').read_text().splitlines():
 if not line.startswith('LIFETIME '):continue
 f=line.split();rows.append(dict(phase=f[1],name=f[2],**{k:int(v) for k,v in (x.split('=') for x in f[3:])}))
first,last=rows[0],rows[-1];delta={k:last[k]-first[k] for k in first if isinstance(first[k],int)}
pairs=[];pending=None
for row in rows:
 if row['phase']=='begin':assert pending is None;pending=row
 else:
  assert pending and row['name']==pending['name']
  pairs.append(dict(name=row['name'],**{k:row[k]-pending[k] for k in delta}));pending=None
summary=dict(markers=len(rows),completed=len(pairs),first=first,last=last,last_unmatched=pending,measured_delta=delta,top_completed=sorted(pairs,key=lambda r:r['ns'],reverse=True)[:15],limits=['observer overhead included','last unmatched interval not measured','nested view timings inclusive nonadditive','allocation counts cover named domains only'])
(p/'timing-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
