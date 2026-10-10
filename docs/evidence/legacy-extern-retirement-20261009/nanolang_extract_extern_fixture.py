from pathlib import Path
import re,json,hashlib
root=Path('/private/tmp/nanolang-match-guards-20261009')
s=(root/'src_nano/transpiler.nano').read_text()
# I identify complete top-level function/shadow bodies, respecting strings/comments.
def body_end(start):
 i=s.index('{',start);depth=0;mode='code'
 while i<len(s):
  c=s[i];n=s[i:i+2]
  if mode=='string':
   if c=='\\':i+=2;continue
   if c=='"':mode='code'
  elif mode=='block':
   if n=='*/':mode='code';i+=2;continue
  elif mode=='line':
   if c=='\n':mode='code'
  elif n=='/*':mode='block';i+=2;continue
  elif n=='//' or c=='#':mode='line'
  elif c=='"':mode='string'
  elif c=='{':depth+=1
  elif c=='}':
   depth-=1
   if depth==0:return i+1
  i+=1
 raise ValueError(start)
functions={};shadows={}
for m in re.finditer(r'^(fn|shadow) (\w+)',s,re.M):
 target=functions if m[1]=='fn' else shadows
 target[m[2]]=s[m.start():body_end(m.start())]
selected=set();pending=['is_runtime_declared_extern','generate_function_prototypes','type_to_c']
while pending:
 name=pending.pop()
 if name in selected:continue
 selected.add(name)
 text=functions[name]+'\n'+shadows.get(name,'')
 pending.extend(n for n in re.findall(r'\(\s*(\w+)',text) if n in functions and n not in selected)
print('Selected functions:',sorted(selected))
# These are the only referenced module globals in this helper closure.
globals_=re.findall(r'^let mut (\w+):[^\n]*',s,re.M)
text='\n\n'.join(functions[n]+'\n\n'+shadows.get(n,'') for n in functions if n in selected)
global_lines=[]
for name in globals_:
 if re.search(r'\b'+name+r'\b',text):global_lines.append(re.search(r'^let mut '+name+r':[^\n]*',s,re.M)[0])
fixture='\n'.join(global_lines)+'\n\n'+text+'\n'
original=(root/'tests/transpiler_externs.nano').read_text().replace('import "src_nano/transpiler.nano"\n','')
imports='import "src_nano/compiler/ir.nano"\nimport "src_nano/compiler/module_bindings.nano"\n'
source=imports+fixture+'\n'+original
out=Path('/private/tmp/nanolang-extern-fixture-probe');out.mkdir(exist_ok=True)
(out/'driver.nano').write_text(source)
(out/'helpers.nano.txt').write_text(fixture)
(out/'provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'functions':{n:hashlib.sha256(functions[n].encode()).hexdigest() for n in sorted(selected)},'globals':global_lines},indent=2)+'\n')
print('Fixture bytes:',len(source),'globals:',global_lines)
