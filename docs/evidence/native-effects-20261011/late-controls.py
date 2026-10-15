from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()))
from tests.test_native_effects import NativeEffectBoundaries
cases={
'entry_argument_count': '''.function main 0 1 0 int 1
HANDLER_PUSH op arm 0 1
HANDLER_POP 1
HANDLER_PUSH op arm 0 0
PERFORM op 0
PUSH_I64 7
EQ
ASSERT
HANDLER_POP 1
PUSH_I64 0
RET
arm:
PUSH_I64 7
EFFECT_RESUME
.end
''',
'halt_in_arm': '''.function owner 0 0 0 int 1
HANDLER_PUSH op arm 0 0
PERFORM op 0
RET
arm:
PUSH_I64 0
HALT
.end
.function main 0 0 0 int 1
CALL owner
POP
PUSH_I64 99
RET
.end
'''}
results={}
for name,body in cases.items():
 try: NativeEffectBoundaries().exercise(body);results[name]='passed'
 except Exception as e: results[name]=str(e)
Path('/private/tmp/nl51-native-effects/late-controls.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
