"""I qualify positive checker events without granting executable borrow authority."""
import ast
from contextlib import contextmanager
from collections import Counter
import os
from pathlib import Path
import shlex
import shutil
import struct
import tempfile
import unittest
from tests.native_process import run_captured, native_compile_timeout

ROOT=Path(__file__).resolve().parents[1]
DRIVERS=[Path(p) for p in os.environ.get('NANOLANG_BORROW_FACT_COMPILERS',os.pathsep.join(str(ROOT/'bin'/name) for name in ('nano_virt','nanoc_stage1','nanoc_stage2'))).split(os.pathsep) if p]
KEY=b'nano.borrow-place.v1'

class CheckedBorrowMetadata(unittest.TestCase):
    def command(self,args,build=False):
        result=run_captured(list(map(str,args)),cwd=ROOT,text=True,timeout=native_compile_timeout() if build else 180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result.stdout

    @contextmanager
    def retain_failure(self,directory):
        try:
            yield
        except BaseException:
            target=Path(os.environ.get('NANOLANG_BORROW_FACT_FAILURE_DIR',ROOT/'.test_output/checked-borrow-failures'))
            target.mkdir(parents=True,exist_ok=True)
            saved=target/Path(directory).name
            shutil.copytree(directory,saved,dirs_exist_ok=True)
            raise

    def decode(self,canonical):
        lines=canonical.splitlines()
        strings=[ast.literal_eval(line[8:]).encode('latin1') for line in lines if line.startswith('.string ')]
        functions=[shlex.split(line)[1] for line in lines if line.startswith('.function ')]
        keys={i for i,s in enumerate(strings) if s==KEY}
        entries=[line for line in lines if line.startswith('.metadata ') and int(line.split()[1]) in keys]
        facts=[];sites=set()
        for entry in entries:
            data=strings[int(entry.split()[2])]
            self.assertGreaterEqual(len(data),40)
            fn,cl,cc,al,ac,ordinal,mode,coords,reserved,np,nr,nj=struct.unpack('<6IBBH3I',data[:40])
            self.assertEqual(reserved,0);self.assertEqual(len(data),40+np+nr+nj)
            self.assertLess(fn,len(functions));self.assertTrue(all(n>0 for n in (cl,cc,al,ac,np,nr)))
            self.assertIn(mode,(1,2));self.assertIn(coords,(1,2))
            path=data[40:40+np].decode();root=data[40+np:40+np+nr].decode();projection=data[40+np+nr:].decode()
            site=(fn,cl,cc,al,ac,ordinal,coords,path)
            self.assertNotIn(site,sites);sites.add(site)
            facts.append((functions[fn],mode,coords,root,projection,ordinal,path))
        return lines,strings,entries,facts

    def test_checked_field_reads_keep_projection_and_root_category(self):
        cases=[('source_borrow_shared.nano', [('read',2,'view','.value')]),
               ('source_borrow_exclusive.nano', [('update',3,'view','.value'),('update',3,'view','.active'),('main',1,'owner','.active')]),
               ('source_borrow_nested.nano', [('main',1,'root','.branch.left.value'),('main',1,'root','.branch.right.value')])]
        for driver in DRIVERS:
            for fixture,required in cases:
                with self.subTest(driver=driver,fixture=fixture),tempfile.TemporaryDirectory(prefix='nano-read-facts-') as directory,self.retain_failure(directory):
                    work=Path(directory);module=work/'source.nvm';source=ROOT/'tests/nanoisa/fixtures'/fixture
                    self.command([driver,source,'--emit-nvm','-o',module])
                    canonical=self.command([ROOT/'bin/nanoisa','dump',module]);(work/'original.nasm').write_text(canonical)
                    lines=canonical.splitlines()
                    strings=[ast.literal_eval(line[8:]).encode('latin1') for line in lines if line.startswith('.string ')]
                    functions=[shlex.split(line)[1] for line in lines if line.startswith('.function ')]
                    entries=[line for line in lines if line.startswith('.metadata ') and strings[int(line.split()[1])]==b'nano.field-read.v1']
                    facts=[];sites=set()
                    for entry in entries:
                        data=strings[int(entry.split()[2])]
                        fn,line,column,category,coords,reserved,np,nr,nj=struct.unpack('<3IBBH3I',data[:28])
                        self.assertEqual(len(data),28+np+nr+nj);self.assertEqual(reserved,0)
                        self.assertGreater(line,0);self.assertGreater(column,0);self.assertLess(fn,len(functions))
                        self.assertIn(category,(1,2,3));self.assertIn(coords,(1,2));self.assertTrue(np and nr and nj)
                        path=data[28:28+np].decode();root=data[28+np:28+np+nr].decode();projection=data[28+np+nr:].decode()
                        self.assertTrue(projection.startswith('.'));self.assertEqual(Path(path).resolve(),source.resolve())
                        site=(fn,line,column,coords,path);self.assertNotIn(site,sites);sites.add(site)
                        facts.append((functions[fn],category,root,projection))
                    for fact in required:self.assertIn(fact,facts)
                    if fixture=='source_borrow_shared.nano':
                        self.assertEqual([fact for fact in facts if fact[0]=='read'],required)
                    elif fixture=='source_borrow_exclusive.nano':
                        self.assertEqual(Counter(fact for fact in facts if fact[0]=='update'),Counter({('update',3,'view','.value'):2,('update',3,'view','.active'):1}))
                    else:
                        self.assertEqual(Counter(fact for fact in facts if fact[0]=='main'),Counter({('main',1,'root','.branch.left.value'):2,('main',1,'root','.branch.right.value'):2}))
                    self.assertFalse(any(path in ('.branch','.branch.left','.branch.right') for *_,path in facts))
                    assembly=work/'roundtrip.nasm';rebuilt=work/'roundtrip.nvm';assembly.write_text(canonical)
                    self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt]);self.assertEqual(module.read_bytes(),rebuilt.read_bytes())
                    self.command([ROOT/'bin/nano_vm',module])
                    native=work/'program.c';binary=work/'program';self.command([ROOT/'bin/nvm2c',module,'-o',native]);original=native.read_bytes()
                    self.command(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',native,'-lm','-o',binary],build=True)
                    self.command([binary])
                    for label,variant in [('stripped',[line for line in lines if line not in entries]),('duplicate',lines+entries)]:
                        assembly.write_text('\n'.join(variant)+'\n');self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt])
                        self.command([ROOT/'bin/nano_vm',rebuilt]);self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                        self.assertEqual(original,native.read_bytes(),label)

    def test_actual_root_and_nested_events(self):
        cases=[('source_borrow_exclusive.nano',[(2,'owner','',0)]),
               ('source_borrow_nested_shared.nano',[(1,'root','.branch.left',0),(1,'root','.branch.left',1),
                  (1,'root','.branch.left',0),(1,'root','.branch.right',1),
                  (1,'root','.branch.right',0),(1,'root','.branch.left',1)])]
        for driver in DRIVERS:
            for fixture,expected in cases:
                with self.subTest(driver=driver,fixture=fixture),tempfile.TemporaryDirectory(prefix='nano-borrow-facts-') as directory,self.retain_failure(directory):
                    work=Path(directory);module=work/'source.nvm';source=ROOT/'tests/nanoisa/fixtures'/fixture
                    self.command([driver,source,'--emit-nvm','-o',module])
                    canonical=self.command([ROOT/'bin/nanoisa','dump',module]);(work/'original.nasm').write_text(canonical)
                    lines,strings,entries,facts=self.decode(canonical)
                    self.assertEqual([(mode,root,path,ordinal) for fn,mode,coords,root,path,ordinal,_ in facts],expected)
                    self.assertTrue(all(fn=='main' for fn,*_ in facts))
                    self.assertTrue(all(Path(path).resolve()==source.resolve() for *_,path in facts))
                    expected_coordinates=1 if driver.name=='nano_virt' else 2
                    self.assertTrue(all(fact[2]==expected_coordinates for fact in facts))
                    assembly=work/'roundtrip.nasm';rebuilt=work/'roundtrip.nvm';assembly.write_text(canonical)
                    self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt]);self.assertEqual(module.read_bytes(),rebuilt.read_bytes())
                    self.command([ROOT/'bin/nano_vm',module])
                    native=work/'program.c';binary=work/'program'
                    self.command([ROOT/'bin/nvm2c',module,'-o',native]);original=native.read_bytes()
                    self.command(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',native,'-lm','-o',binary],build=True)
                    self.command([binary])
                    payload=bytearray(strings[int(entries[0].split()[2])]);payload[24]=0
                    forged=lines.copy();position=next(i for i,line in enumerate(forged) if line.startswith('.metadata '))
                    forged.insert(position,'.string "'+''.join(f'\\x{byte:02x}' for byte in payload)+'"')
                    forged[forged.index(entries[0])]=f'.metadata {entries[0].split()[1]} {len(strings)}'
                    for variant in ([line for line in lines if line not in entries],lines+[entries[0]],forged):
                        assembly.write_text('\n'.join(variant)+'\n');self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt])
                        self.command([ROOT/'bin/nano_vm',rebuilt]);self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native]);self.assertEqual(native.read_bytes(),original)

    def test_four_raw_checked_contract_neighbors(self):
        from tests.test_source_borrow_emission import affine_comparison
        raw=Path(os.environ.get('NANOLANG_BORROW_RAW_EMITTER',ROOT/'bin/nanoisa_emit'))
        seed=next((driver for driver in DRIVERS if driver.name=='nano_virt'),ROOT/'bin/nano_virt')
        for fixture in ('source_borrow_shared.nano','source_borrow_exclusive.nano','source_borrow_order.nano','source_borrow_bool.nano'):
            with self.subTest(fixture=fixture),tempfile.TemporaryDirectory(prefix='nano-borrow-neighbor-') as directory,self.retain_failure(directory):
                work=Path(directory);source=ROOT/'tests/nanoisa/fixtures'/fixture
                baseline=work/'seed.nvm';self.command([seed,source,'--emit-nvm','--strip-debug','-o',baseline])
                expected=self.command([ROOT/'bin/nanoisa','dump',baseline])
                assembly=work/'raw.nasm';raw_module=work/'raw.nvm'
                self.command([raw,source,'-o',assembly]);self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',raw_module])
                actual=self.command([ROOT/'bin/nanoisa','dump',raw_module])
                self.assertEqual(affine_comparison(actual,0),affine_comparison(expected,1))
                modules=[baseline,raw_module]
                for index,driver in enumerate(DRIVERS):
                    if driver==seed:continue
                    module=work/f'checked-{index}.nvm'
                    self.command([driver,source,'--emit-nvm','-o',module]);modules.append(module)
                for index,module in enumerate(modules):
                    self.command([ROOT/'bin/nano_vm',module])
                    native=work/f'program-{index}.c';binary=work/f'program-{index}'
                    self.command([ROOT/'bin/nvm2c',module,'-o',native])
                    cc=shlex.split(os.environ.get('NANO_NATIVE_TEST_CC',os.environ.get('CC','cc')))
                    self.command(cc+['-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',native,'-lm','-o',binary],build=True)
                    self.command([binary])

    def test_actual_checker_context_raw_recheck_and_descriptor_lifetime(self):
        # I execute the changed Nano checker/bridge, not a fabricated fact payload.
        helper = r'''import "src_nano/parser.nano"
from "src_nano/compiler/lexer.nano" import tokenize_string
from "src_nano/compiler/checked_borrow_facts.nano" import borrow_fact_lookup, field_read_fact_lookup
from "src_nano/typecheck.nano" import typecheck_for_immediate_emission
from "src_nano/compiler/nanoisa_codegen.nano" import nanoisa_check_and_emit
from "src_nano/compiler/nanoisa_borrows.nano" import nanoisa_borrow_emit, nanoisa_borrow_lower
fn main() -> int {
 let source:string="resource struct R { value:int } fn read(view:&R)->int{return view.value} shadow read{let owner:R=R{value:7} assert (== (read &owner) 7) let R{value}=owner} fn main()->int{let owner:R=R{value:9} assert (== (read &owner) 9) let R{value}=owner return 0} shadow main{assert true}"
 let diagnostics:List<CompilerDiagnostic> =(list_CompilerDiagnostic_new)
 let tokens:List<LexerToken> =(tokenize_string source "ledger-context.nano" diagnostics)
 let parser:Parser=(parse_program tokens (list_LexerToken_length tokens) "ledger-context.nano")
 let checked:CheckedAdvisoryOutput=(typecheck_for_immediate_emission parser "ledger-context.nano" 0)
 assert (not checked.phase.had_error)
 assert (== (array_length checked.borrow_context) 1)
 let events:array<BorrowFactEvent> =(at checked.borrow_context 0).events
 assert (> (array_length events) 0)
 let mut found:int=0
 let mut i:int=0
 while (< i (parser_get_function_count parser)) {
  let function:ASTFunction=(parser_get_function parser i)
  let mut j:int=0
  while (< j (array_length events)) {
   let event:BorrowFactEvent=(at events j)
   if (>= (borrow_fact_lookup parser checked.borrow_context function event.call event.argument event.ordinal) 0) {set found (+ found 1)}
   set j (+ j 1)
  }
  set i (+ i 1)
 }
 assert (> found 0)
 let reads:array<FieldReadFactEvent> =(at checked.borrow_context 0).reads
 assert (> (array_length reads) 0)
 let mut found_reads:int=0
 set i 0
 while (< i (parser_get_function_count parser)) {
  let function:ASTFunction=(parser_get_function parser i)
  let mut j:int=0
  while (< j (array_length reads)) {
   let event:FieldReadFactEvent=(at reads j)
   if (>= (field_read_fact_lookup parser checked.borrow_context function event.field) 0) {
    set found_reads (+ found_reads 1)
    let original:ASTFieldAccess=(parser_get_field_access parser event.field)
    (list_ASTFieldAccess_set parser.field_accesses event.field ASTFieldAccess {node_type:original.node_type,line:original.line,column:original.column,object:original.object,object_type:original.object_type,field_name:"wrong"})
    assert (== (field_read_fact_lookup parser checked.borrow_context function event.field) (- 0 1))
    (list_ASTFieldAccess_set parser.field_accesses event.field ASTFieldAccess {node_type:original.node_type,line:original.line,column:original.column,object:(- 0 1),object_type:original.object_type,field_name:original.field_name})
    assert (== (field_read_fact_lookup parser checked.borrow_context function event.field) (- 0 1))
    (list_ASTFieldAccess_set parser.field_accesses event.field original)
    assert (>= (field_read_fact_lookup parser checked.borrow_context function event.field) 0)
   }
   set j (+ j 1)
  }
  set i (+ i 1)
 }
 assert (> found_reads 0)
 let first:NanoisaBorrowLowering=(nanoisa_borrow_lower parser false 0 true)
 assert (> (array_length first.sites) 0)
 let saved:NanoisaBorrowSite=(at first.sites 0)
 assert (> (array_length first.reads) 0)
 let saved_read:NanoisaFieldReadSite=(at first.reads 0)
 let second:NanoisaBorrowLowering=(nanoisa_borrow_lower parser true 0 true)
 assert (!= second.text "")
 assert (== (at first.sites 0).call saved.call)
 assert (== (at first.reads 0).field saved_read.field)
 assert (not (str_contains first.text ".field_read"))
 assert (== (at first.sources saved.emitted) saved.source)
 assert (not (str_contains first.text ".borrow_place"))
 assert (not (str_contains (nanoisa_borrow_emit parser false 0) ".borrow_place"))
 let emitted:NanoisaCheckedText=(nanoisa_check_and_emit parser "ledger-context.nano" 0 true)
 assert (not emitted.phase.had_error)
 assert (== emitted.lowering_stage 0)
 assert (str_contains emitted.program ".borrow_place")
 assert (str_contains emitted.program ".field_read")
 assert (!= emitted.shadows "")
 set i 0
 while (< i (parser_get_function_count parser)) {
  let function:ASTFunction=(parser_get_function parser i)
  let mut j:int=0
  while (< j (array_length events)) {
   let event:BorrowFactEvent=(at events j)
   assert (== (borrow_fact_lookup parser checked.borrow_context function event.call event.argument event.ordinal) (- 0 1))
   set j (+ j 1)
  }
  set i (+ i 1)
 }
 set i 0
 while (< i (parser_get_function_count parser)) {
  let function:ASTFunction=(parser_get_function parser i)
  let mut j:int=0
  while (< j (array_length reads)) {
   assert (== (field_read_fact_lookup parser checked.borrow_context function (at reads j).field) (- 0 1))
   set j (+ j 1)
  }
  set i (+ i 1)
 }
 assert (not (str_contains (nanoisa_borrow_emit parser false 0) ".borrow_place"))
 assert (not (str_contains (nanoisa_borrow_emit parser false 0) ".field_read"))
 return 0
}
shadow main { assert true }
'''
        with tempfile.TemporaryDirectory(prefix='nano-borrow-context-') as directory,self.retain_failure(directory):
            work=Path(directory);source=work/'context.nano';binary=work/'context'
            source.write_text(helper)
            self.command([ROOT/'bin/nanoc_c',source,'-o',binary],build=True)
            self.command([binary])

    def test_multi_helper_import_profile_remains_refused(self):
        for driver in DRIVERS:
            for reverse in (False,True):
                with self.subTest(driver=driver,reverse=reverse),tempfile.TemporaryDirectory(prefix='nano-borrow-imports-') as directory,self.retain_failure(directory):
                    work=Path(directory)
                    for name,record,value in [('left','Left',7),('right','Right',9)]:
                        (work/(name+'.nano')).write_text(f"""resource struct {record} {{ value:int }}
fn read(view:&{record})->int{{return view.value}}
shadow read {{assert true}}
pub fn run()->int{{let owner:{record}={record}{{value:{value}}} assert (== (read &owner) {value}) let {record}{{value}}=owner return value}}
shadow run {{assert (== (run) {value})}}
""")
                    imports=['module "left.nano" as left','module "right.nano" as right']
                    if reverse:imports.reverse()
                    source=work/'main.nano';module=work/'main.nvm'
                    source.write_text('\n'.join(imports)+'\nfn main()->int{assert (== (left.run) 7) assert (== (right.run) 9) return 0}\nshadow main{assert (== (main) 0)}\n')
                    module.write_bytes(b'prior artifact')
                    result=run_captured([str(driver),str(source),'--emit-nvm','-o',str(module)],cwd=ROOT,text=True,timeout=180)
                    (work/'refusal.log').write_text(result.stdout+result.stderr)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('exactly main and one borrowed helper',result.stdout+result.stderr)
                    self.assertEqual(module.read_bytes(),b'prior artifact')

    def test_imported_helper_existing_producer_boundary(self):
        fixture=(ROOT/'tests/nanoisa/fixtures/source_borrow_exclusive.nano').read_text()
        boundary=fixture.index('fn main()')
        for driver in DRIVERS:
            with self.subTest(driver=driver),tempfile.TemporaryDirectory(prefix='nano-borrow-import-helper-') as directory,self.retain_failure(directory):
                work=Path(directory);source=work/'main.nano';module=work/'main.nvm'
                (work/'owner.nano').write_text(fixture[:boundary].replace('resource struct','pub resource struct').replace('fn update','pub fn update'))
                source.write_text('from "owner.nano" import Counter, update\n'+fixture[boundary:])
                module.write_bytes(b'prior artifact')
                result=run_captured([str(driver),str(source),'--emit-nvm','-o',str(module)],cwd=ROOT,text=True,timeout=180)
                if driver.name!='nano_virt':
                    # I preserve the independently reproduced baseline Nano profile refusal.
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('exact declared borrowed helper',result.stdout+result.stderr)
                    self.assertEqual(module.read_bytes(),b'prior artifact')
                    continue
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                canonical=self.command([ROOT/'bin/nanoisa','dump',module]);(work/'original.nasm').write_text(canonical)
                _,_,_,facts=self.decode(canonical)
                self.assertEqual(len(facts),1)
                self.assertEqual(facts[0][:6],('main',2,1,'owner','',0))
                self.assertEqual(Path(facts[0][6]).resolve(),source.resolve())
                self.command([ROOT/'bin/nano_vm',module])

    def test_later_argument_and_runtime_shadow_failure_preserve_output(self):
        overlap='''resource struct R{value:int}
fn inspect(a:&mut R,b:&R)->int{return (+ a.value b.value)}
fn main()->int{let mut owner:R=R{value:1} let n:int=(inspect &mut owner &owner) let R{value}=owner return 0}
'''
        shadow='''resource struct R{value:int}
fn read(a:&R)->int{return a.value}
shadow read {assert false}
fn main()->int{let owner:R=R{value:1} assert (== (read &owner) 1) let R{value}=owner return 0}
'''
        for driver in DRIVERS:
            for source,diagnostic in ((overlap, "borrow"), (shadow, "shadow")):
                with self.subTest(driver=driver,source=source),tempfile.TemporaryDirectory(prefix='nano-borrow-refusal-') as directory,self.retain_failure(directory):
                    work=Path(directory);src=work/'bad.nano';output=work/'prior.nvm';src.write_text(source);output.write_bytes(b'prior artifact')
                    result=run_captured([str(driver),str(src),'--emit-nvm','-o',str(output)],cwd=ROOT,text=True,timeout=180)
                    (work/'refusal.log').write_text(result.stdout+result.stderr)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn(diagnostic,(result.stdout+result.stderr).lower())
                    self.assertEqual(output.read_bytes(),b'prior artifact')

if __name__=='__main__':unittest.main()
