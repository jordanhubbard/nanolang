from pathlib import Path
import subprocess,json
p=Path(__file__).parent;rows=json.loads((p/'budget-commands.json').read_text())
for r in rows:
 q=subprocess.run(r['command'],capture_output=True,timeout=180);(p/(r['name']+'.stdout')).write_bytes(q.stdout);(p/(r['name']+'.stderr')).write_bytes(q.stderr);assert q.returncode==0 and not q.stderr,(r['name'],q.returncode,q.stderr)
assert (p/'old-six-budget.stdout').read_bytes()==Path('docs/evidence/map-paired-publication-20261022/old-six-budget.stdout').read_bytes()
print('PASS current source six old budget rows byte-identical')
