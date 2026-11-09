"""I retain a second private candidate without overwriting my first refusal."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
before=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('faults',HERE/'run-faults.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.SOURCE=HERE/'fallback-candidate'
output=Path(sys.argv[1]).resolve()
module.run(output)
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==before
p=output/'report.json';report=json.loads(p.read_text())
report['inputs_before'][str(Path(__file__).resolve())]=before
report['inputs_after'][str(Path(__file__).resolve())]=before
p.write_text(json.dumps(report,indent=2)+'\n')
