from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()))
from tests.test_native_effects import NativeEffectBoundaries
body='''.function owner 0 0 0 int 1
PUSH_I64 0
HALT
.end
.function main 0 0 0 int 1
CALL owner
POP
PUSH_I64 99
RET
.end
'''
try:
 NativeEffectBoundaries().exercise(body)
 result={'result':'passed'}
except AssertionError as error:
 result={'result':'failed','error':str(error)}
Path('/private/tmp/nl51-native-effects/ordinary-halt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
