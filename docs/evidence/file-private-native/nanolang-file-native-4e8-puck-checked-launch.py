import json,subprocess,os
from pathlib import Path
report=Path('/tmp/nanolang-file-native-4e8-puck-launch');report.mkdir(exist_ok=False)
commands=[]
def run(label,args,**kw):
 commands.append({'label':label,'argv':args});(report/'commands.json').write_text(json.dumps(commands,indent=2))
 with (report/(label+'.log')).open('w') as log:
  p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,**kw)
 (report/(label+'-status.txt')).write_text(str(p.returncode)+'\n');p.check_returncode()
run('source-map-transfer',['scp','-q','/tmp/nanolang-file-native-4e8-linux/normal-source-before.json','puck.local:/tmp/nanolang-file-native-4e8-source-map.json'],timeout=60)
code='''from pathlib import Path
import hashlib,tarfile,shutil,json,inspect
assert 'filter' in inspect.signature(tarfile.TarFile.extractall).parameters
archive=Path('/tmp/nanolang-file-native-4e8-source.tar.gz');assert hashlib.sha256(archive.read_bytes()).hexdigest()=='e67142efe9259d180295a07d40cd0b6b643eccc5d3596eb605949033ee543cbe'
root=Path('/tmp/nanolang-file-native-4e8-corrected');root.mkdir(exist_ok=False)
with tarfile.open(archive) as t:t.extractall(root,filter='data')
source=json.loads(Path('/tmp/nanolang-file-native-4e8-source-map.json').read_text())
for name,sha in source.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
shutil.copyfile('/tmp/nanolang-file-native-4e8-tracked',root/'.qualification-tracked')
(root/'.qualification-pin').write_text('4e8ec0d67\\n')
assert not (root/'obj').exists();print('Verified source archive and',len(source),'source hashes; fresh provider closure')
'''
(report/'extraction.py').write_text(code)
run('extraction',['ssh','puck.local','/opt/homebrew/bin/python3','-'],input=code,text=True,timeout=120)
command='env PATH=/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang CARRIER_SAN_CC=/opt/homebrew/opt/llvm/bin/clang CARRIER_PHASES=setup,normal,sanitizer,vm,frames,opcodes,wrapper SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk LIBRARY_PATH=/opt/homebrew/opt/openssl@3/lib NANO_FILE_RUNTIME_CFLAGS=-I/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/include/ffi LSAN_OPTIONS= /opt/homebrew/bin/python3 /tmp/nanolang-file-native-acff-driver.py /tmp/nanolang-file-native-4e8-corrected /tmp/nanolang-file-native-4e8-puck-corrected'
run('driver',['ssh','puck.local',command])
