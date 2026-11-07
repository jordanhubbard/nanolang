from pathlib import Path
import hashlib,json,os,sys,time
sys.path.insert(0,str(Path.cwd()))
from tests.native_process import run_captured
from tests.test_record_map_native import RecordMapNative
root=Path('/Users/jkh/Src/nanolang-51-harness');out=Path(__file__).parent
producers=[('seed',root/'bin/nano_virt'),('c-hosted',Path('/private/tmp/nanolang-generic-record-field-20261021/canonical')),('stage1',root/'bin/nanoc_stage1'),('stage2',root/'bin/nanoc_stage2')]
tools={n:root/'bin'/n for n in ['nano_vm','nanoisa','nvm2c']};cc=Path('/opt/homebrew/opt/llvm/bin/clang')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=Path('tests/nanoisa/fixtures/map_record_fields.nano').read_bytes();old=b'(remember alias "second" 2)';new=b'(remember alias (+ "sec" "ond") 2)';assert original.count(old)==1
variant=original.replace(old,new);assert [l for l in original.splitlines() if b'assert' in l]==[l for l in variant.splitlines() if b'assert' in l]
for n,s in [('unchanged',original),('concat',variant)]: (out/(n+'.nano')).write_bytes(s)
paths=[p for _,p in producers]+list(tools.values())+[cc];receipt={'base':'24feed248','before':{str(p):sha(p) for p in paths},'sources':{n:sha(out/(n+'.nano')) for n in ['unchanged','concat']},'commands':[],'cases':[]}
def save(): (out/'raw-baseline.json').write_text(json.dumps(receipt,indent=2)+'\n')
def run(name,args,build=False):
 p=run_captured(list(map(str,args)),cwd=Path.cwd(),timeout=180,build=build);(out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr);receipt['commands'].append(dict(name=name,argv=list(map(str,args)),status=p.returncode));save();return p
for n in ['unchanged','concat']:
 for provider,compiler in producers:
  label=n+'-'+provider;m=out/(label+'.nvm');case=dict(source=n,producer=provider);receipt['cases'].append(case)
  p=run(label+'-compile',[compiler,out/(n+'.nano'),'--emit-nvm','-o',m]);case['compile']=p.returncode
  if p.returncode:save();continue
  for phase,args in [('verify',[tools['nano_vm'],'--verify-only',m]),('vm',[tools['nano_vm'],m]),('dump',[tools['nanoisa'],'dump',m]),('native-translate',[tools['nvm2c'],m,'-o',out/(label+'.c')])]:
   p=run(label+'-'+phase,args);case[phase]=p.returncode
   if p.returncode:raise SystemExit(label+' '+phase+' failed; retained')
  dump=(out/(label+'-dump.stdout')).read_bytes();case['concat_opcodes']=dump.count(b'STR_CONCAT');case['struct_opcodes']=dump.count(b'STRUCT_LITERAL')+dump.count(b'STRUCT_NEW');case['authority_present']=b'.ownership ' in dump or b'.layouts ' in dump
  for forced in [False,True]:
   c=out/(label+'.c')
   if forced:c=out/(label+'-forced.c');c.write_text(RecordMapNative.force_collection((out/(label+'.c')).read_text()))
   for opt in ['0','2']:
    tag=label+('-forced' if forced else '')+'-O'+opt;exe=out/tag
    p=run(tag+'-build',[cc,'-std=c11','-O'+opt,'-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',exe],True);assert p.returncode==0,p.stderr
    p=run(tag+'-run',[exe]);assert p.returncode==0 and not p.stderr,(tag,p.returncode,p.stderr)
  save()
receipt['after']={str(p):sha(p) for p in paths};assert receipt['before']==receipt['after'];save();print(json.dumps(receipt['cases'],indent=2))
