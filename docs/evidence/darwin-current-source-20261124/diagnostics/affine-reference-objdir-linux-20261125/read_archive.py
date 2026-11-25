"""I verify the private-directory reference links and their complete C reports."""
from pathlib import Path
import hashlib,shlex,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'62ef9918120131591e64877ad14de8ae81a8ac3e93fb47fd87107556357f0542')
sha=lambda b:hashlib.sha256(b).hexdigest()
a=js('controls/actual-before-result.json');b=js('controls/actual-after-result.json');r=js('controls/reports-result.json');selection=js('controls/result.json')
assert m['c_reference_links_only'] and not m['full_five_routes'] and not m['whole_release']
assert a['terminal'] and b['terminal'] and a['all_guards_match'] and b['all_guards_match'] and r['terminal'] and r['accepted']
assert len(a['commands'])==1 and a['commands'][0]['exit']==2 and b'multiple definition' in data('controls/actual-before.stderr')
assert selection['exit']!=0 and b'multiple definition' in data('controls/duplicate.stderr')
assert selection['make_expansions'][0]['before']==selection['make_expansions'][0]['after']
assert 'private-obj/nanoisa/affine_bytecode.o' in selection['make_expansions'][1]['before'] and 'private-obj/nanoisa/affine_bytecode.o' not in selection['make_expansions'][1]['after']
assert len(b['commands'])==4 and all(c['exit']==0 and c['timeout']==60 for c in b['commands'])
assert a['objects_before']==a['objects_after']==b['objects_before']==b['objects_after']==r['objects_before']==r['objects_after']
assert a['compiler_before']==a['compiler_after']==b['compiler_before']==b['compiler_after']
assert b['source_before']==b['source_after']==r['source_before']==r['source_after']
def mapped(name):
 p=Path(name)
 for label,root in sorted(m['roots'].items(),key=lambda x:len(x[1]),reverse=True):
  if p.is_relative_to(root):return label+'/'+str(p.relative_to(root))
 return 'external/'+str(p).lstrip('/')
for n,h in {**b['objects_before'],**b['source_before']}.items():assert rows[mapped(n)]['sha256']==h,n
assert rows['external/usr/bin/cc']['sha256']==b['compiler_before']
before=data('controls/Makefile.before');after=data('controls/Makefile.after');assert sha(before)==a['makefile_before'] and sha(after)==b['makefile_after']
lines=before.splitlines(keepends=True);changed=0
for i,line in enumerate(lines):
 if b'tests/nvm2c_affine_' in line and b'$(filter-out obj/nanoisa/affine_bytecode.o,' in line:
  lines[i]=line.replace(b'$(filter-out obj/nanoisa/affine_bytecode.o,',b'$(filter-out $(OBJ_DIR)/nanoisa/affine_bytecode.o,');changed+=1
assert changed==4 and b''.join(lines)==after
for c in b['commands']:
 label=c['label'];assert data('controls/'+label+'.stderr')==b''
 emitted=data('controls/'+label+'.stdout').decode().splitlines();assert len(emitted)==1
 argv=shlex.split(emitted[0]);assert argv[0]=='/usr/bin/cc'
 assert not any(n.endswith('/nanoisa/affine_bytecode.o') for n in argv)
 for flag in ['-Wall','-Wextra','-Werror','-std=c99','-O3','-fPIC']:assert flag in argv
 for n in argv:
  if n.endswith(('.o','.a')):assert '--assume-old='+n in c['argv']
assert len(r['commands'])==693 and all(c['exit']==0 for c in r['commands'])
observed={}
for i,c in enumerate(r['commands'],1):
 assert data('controls/report-%04d.stderr'%i)==b''
 kind=Path(c['argv'][0]).name.removeprefix('nvm2c_affine_').removesuffix('_reference');paths=c['argv'][1:]
 group='original' if Path(paths[0]).name.startswith('original-') else 'globals'
 assert c['timeout']==(600 if group=='original' else 60)
 assert len(paths)<=(8 if group=='original' else 32)
 if group=='globals':assert len(paths)==1 or sum(rows[mapped(n)]['bytes'] for n in paths)<=8192
 observed.setdefault((kind,group),[]).append(data('controls/report-%04d.stdout'%i))
assert len(r['groups'])==5
for g in r['groups']:
 raw=b''.join(observed[g['kind'],g['group']]);assert raw==data('controls/'+g['kind']+'-'+g['group']+'.reports')==data('oracle/'+g['group']+'.reports')
 assert sha(raw)==g['sha256'] and len(raw.decode().splitlines())==g['count']==(1001 if g['group']=='original' else 3688)
 assert rows['controls/private-obj/nvm2c_affine_'+g['kind']+'_reference']['sha256']==g['image_sha256']
# I bind both Makefile states and selected source bytes to Git independently.
assert subprocess.check_output(['git','show','be8047814d:Makefile.gnu'],cwd=base)==before
assert subprocess.check_output(['git','show','3408c58c65:Makefile.gnu'],cwd=base)==after
repo_prefix='/var/tmp/nanolang-darwin-publication-qkk55gi3/'
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in b['source_before'].items():
 assert n.startswith(repo_prefix);proc.stdin.write(('3408c58c65:'+n[len(repo_prefix):]+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,n
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; actual Make red/four green links, all693 report commands, original1001 reports on all four references plus new3688 globals, exact source/tool/object guards and default/private selections. No full custom build, allocation-target, NanoLang/paired/CFG/cutover/release acceptance.'%len(rows))
