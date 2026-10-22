from pathlib import Path
import os,json,shlex,sys,unittest
p=Path(__file__).parent;s=json.loads((p/'focused.json').read_text());sel=s['retained_selection'];command=next(x['args'] for x in s['commands'] if x['name']=='focused-switch-build');start=command.index('tests/nanoisa/record_array_alloc.c')+1;end=len(command)-2-len(sel['ldflags']);objects=[x for x in command[start:end] if not Path(x).name.startswith("combined-")]
os.environ.update(RECORD_ARRAY_VM_CC=shlex.join(sel['cc']),RECORD_ARRAY_VM_CFLAGS=shlex.join(x for x in sel['flags'] if x!='-DNANO_NO_COMPUTED_GOTO'),RECORD_ARRAY_VM_OBJECTS=shlex.join(objects),RECORD_ARRAY_VM_LDFLAGS=shlex.join(sel['ldflags']),RECORD_MAP_STRING_MODULES=shlex.join(str(p/(n+'-selected.nvm')) for n in ['seed','canonical','c-shadow','raw-shadow']))
sys.path.insert(0,str(Path.cwd()));result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_record_map_string'));sys.exit(not result.wasSuccessful())
