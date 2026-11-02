"""I qualify actual source-produced flat-array authority, not hand-authored bytecode."""
import ctypes
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import tempfile
import unittest
from tests.native_process import run_captured

ROOT = Path(__file__).resolve().parents[1]

class FlatArraySourcePublication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler_work=Path(tempfile.mkdtemp(prefix='nano-flat-source-producers-'))
        # I explicitly prepare the real query library even when fresh artifact
        # producers are supplied: --emit-nvm alone need not build a native module.
        probe=cls.compiler_work/'query-provider.nano'
        binary=cls.compiler_work/'query-provider'
        probe.write_text('module "modules/nanoisa/nanoisa.nano" as Isa\n'
            'fn main()->int { assert (== (Isa.array_publication "not assembly") -1) return 0 }\n'
            'shadow main { assert (== (main) 0) }\n')
        build=run_captured([ROOT/'bin/nanoc_c',probe,'-o',binary],cwd=ROOT,
            text=True,timeout=180,build=True)
        (cls.compiler_work/'query-provider-build.log').write_text(build.stdout+build.stderr)
        if build.returncode:
            raise AssertionError(build.stdout+build.stderr)
        run=run_captured([binary],cwd=ROOT,text=True,timeout=30)
        if run.returncode:
            raise AssertionError(run.stdout+run.stderr)
        cls.query_library=Path(os.environ.get('NANO_FLAT_ARRAY_QUERY_LIBRARY',str(ROOT/'bin/libnanoisa_query.so')))
        if 'NANO_FLAT_ARRAY_QUERY_LIBRARY' not in os.environ:
            build=run_captured(['make','-j1','bin/libnanoisa_query.so'],cwd=ROOT,
                text=True,timeout=180,build=True)
            (cls.compiler_work/'query-library-build.log').write_text(build.stdout+build.stderr)
            if build.returncode:
                raise AssertionError(build.stdout+build.stderr)
        if not cls.query_library.is_file():
            raise AssertionError('I require the actual query shared library')
        configured=os.environ.get('NANO_FLAT_ARRAY_COMPILERS')
        if configured:
            cls.compilers=configured.split(',')
        else:
            emitter=cls.compiler_work/'emitter'
            build=subprocess.run([str(ROOT/'bin/nanoc_c'), 'src_nano/nanoisa_emit.nano', '-o',str(emitter)],
                                 cwd=ROOT,text=True,capture_output=True,timeout=600)
            (cls.compiler_work/'emitter-build.log').write_text(build.stdout+build.stderr)
            if build.returncode:
                raise AssertionError(build.stdout+build.stderr)
            cls.compilers=[str(ROOT/'bin/nano_virt'),str(emitter)]
        print('I retain producer build evidence at',cls.compiler_work,flush=True)

    def run_command(self, *args, ok=True):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, text=True,
                                capture_output=True, timeout=180)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def producers(self):
        return self.compilers

    def exercise(self, source, elements, run=True, modules=None, layout_count=1, array_layout=0, expected_opcode=None):
        for compiler in self.producers():
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-flat-source-') as directory:
                work=Path(directory); program=work/'source.nano'; module=work/'source.nvm'
                for name,text in (modules or {}).items():
                    (work/name).write_text(text)
                program.write_text(source)
                self.run_command(compiler, program, '--emit-nvm', '-o', module)
                self.run_command(ROOT/'bin/nano_vm', '--verify-only', module)
                dump=self.run_command(ROOT/'bin/nanoisa', 'dump', module).stdout
                if expected_opcode:
                    self.assertIn('\n  '+expected_opcode+' ',dump)
                raw=lambda key: bytes.fromhex(''.join(re.findall(r'^\.'+key+r' "([0-9a-f]*)"$',dump,re.M)))
                layouts, facts=raw('layouts'),raw('ownership')
                if elements is None:
                    self.assertEqual((layouts,facts),(b'',b''))
                else:
                    self.assertEqual(struct.unpack_from('<II',facts),(3,layout_count))
                    prefix=8+((layout_count+3)&~3)
                    self.assertEqual(facts[8:prefix],b'\x01'*layout_count+b'\0'*(prefix-8-layout_count))
                    count=struct.unpack_from('<I',facts,prefix)[0]; at=prefix+4
                    for _ in range(count):
                        locals_,_=struct.unpack_from('<HH',facts,at);at+=4+8*(locals_+1)
                    self.assertEqual(struct.unpack_from('<IIIHH',facts,at),(4,0,1,2,1));at+=16
                    size=struct.unpack_from('<I',facts,at)[0];at+=4
                    self.assertEqual(len(facts)-at,size)
                    types=struct.unpack_from('<I',facts,at)[0];at+=4
                    rows=[];wire_rows=[]
                    for _ in range(types):
                        tag,ref=struct.unpack_from('<B3xI',facts,at);at+=8
                        wire_rows.append((tag,ref))
                        if tag==7:
                            self.assertLess(ref,len(rows))
                            self.assertIn(rows[ref],(1,2,3,4,5,7,8,9,10,11))
                            rows.append({1:7,5:8,2:9,3:10,4:11,7:12,9:13,10:14,11:15,8:16}[rows[ref]])
                        else:
                            self.assertEqual(ref,0xffffffff);rows.append(tag)
                    required=set(elements)
                    for code,middle in ((12,7),(13,9),(14,10),(15,11),(16,8)):
                        if code in required: required.add(middle)
                    if 7 in required: required.add(1)
                    if 8 in required: required.add(5)
                    for code in (9,10,11):
                        if code in required: required.add(code-7)
                    ordered=sorted(required)
                    expected_rows=[(7,ordered.index(1)) if element==7 else
                                   (7,ordered.index(5)) if element==8 else
                                   (7,ordered.index({12:7,13:9,14:10,15:11,16:8}[element])) if element>=12 else
                                   (7,ordered.index(element-7)) if element>=9 else
                                   (element,0xffffffff) for element in ordered]
                    self.assertEqual(wire_rows,expected_rows)
                    bindings=struct.unpack_from('<I',facts,at)[0];at+=4
                    actual=[]
                    for field in range(bindings):
                        layout,ordinal,reserved,element=struct.unpack_from('<IHHI',facts,at);at+=12
                        expected_layout,expected_field=array_layout[field] if isinstance(array_layout,list) else (array_layout,field)
                        self.assertEqual((layout,ordinal,reserved),(expected_layout,expected_field,0));actual.append(rows[element])
                    self.assertEqual(actual,elements);self.assertEqual(at,len(facts))
                    # I retain exact old flat/INT wire bytes, including reserved
                    # padding, while STRING adds its separate referent row.
                    expected_wire=struct.pack('<I',len(expected_rows))
                    expected_wire+=b''.join(struct.pack('<B3xI',tag,ref) for tag,ref in expected_rows)
                    expected_wire+=struct.pack('<I',len(elements))
                    expected_wire+=b''.join(struct.pack('<IHHI',*(array_layout[field] if isinstance(array_layout,list) else (array_layout,field)),0,ordered.index(element))
                                            for field,element in enumerate(elements))
                    self.assertEqual(facts[-len(expected_wire):],expected_wire)
                    library_path=self.query_library
                    library=ctypes.CDLL(str(library_path));query=library.nl_nanoisa_array_publication
                    query.argtypes=[ctypes.c_char_p];query.restype=ctypes.c_int64
                    self.assertEqual(query(dump.encode()),1)
                    self.assertIn('AGG_PACK',dump);self.assertNotIn('STRUCT_LITERAL',dump)
                    assembly=work/'roundtrip.nasm';roundtrip=work/'roundtrip.nvm';assembly.write_text(dump)
                    self.run_command(ROOT/'bin/nanoisa','asm',assembly,'-o',roundtrip)
                    self.assertEqual(self.run_command(ROOT/'bin/nanoisa','dump',roundtrip).stdout,dump)
                    # Wrong scalar element remains a structurally valid envelope but must fail actual flow.
                    forged=bytearray(facts); payload=prefix+4
                    for _ in range(count):
                        locals_,_=struct.unpack_from('<HH',forged,payload);payload+=4+8*(locals_+1)
                    first_type=payload+24
                    forged[first_type]=5 if forged[first_type]!=5 else 1
                    stripped=re.sub(r'^\.ownership "[0-9a-f]*"\n','',dump,flags=re.M)
                    bad=work/'forged.nasm';bad.write_text(stripped+'\n.ownership "'+forged.hex()+'"\n')
                    prior=work/'preserved.nvm';prior.write_bytes(b'prior-output')
                    rejected=self.run_command(ROOT/'bin/nanoisa','asm',bad,'-o',prior,ok=False)
                    self.assertNotEqual(rejected.returncode,0);self.assertEqual(prior.read_bytes(),b'prior-output')
                    # Well-formed but unproved element flow is UNRESOLVED: no optional publication.
                    self.assertEqual(query(bad.read_bytes()),0)
                    malformed=bytearray(facts);struct.pack_into('<I',malformed,first_type+4,0)
                    malformed_text=stripped+'\n.ownership "'+malformed.hex()+'"\n'
                    self.assertEqual(query(malformed_text.encode()),-1)
                    binding_count_at=first_type+types*8
                    binding_rows=binding_count_at+4
                    missing=bytearray(facts[:binding_rows]);struct.pack_into('<I',missing,binding_count_at,0)
                    struct.pack_into('<I',missing,payload+16,len(missing)-(payload+20))
                    duplicate=bytearray(facts+facts[binding_rows:binding_rows+12])
                    struct.pack_into('<I',duplicate,binding_count_at,bindings+1)
                    struct.pack_into('<I',duplicate,payload+16,len(duplicate)-(payload+20))
                    for invalid in (missing,duplicate):
                        invalid_text=stripped+'\n.ownership "'+invalid.hex()+'"\n'
                        self.assertEqual(query(invalid_text.encode()),-1)
                    self.assertEqual(query(dump.encode()),1)
                if run:
                    self.run_command(ROOT/'bin/nano_vm',module)
                    native=work/'native.c';binary=work/'native'
                    self.run_command(ROOT/'bin/nvm2c',module,'-o',native)
                    self.run_command(*shlex.split(os.environ.get('CC','cc')),'-std=c11','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',native,'-lm','-o',binary)
                    self.run_command(binary)

    def test_query_status_and_host_interfaces(self):
        self.assertTrue(self.query_library.is_file())
        library=ctypes.CDLL(str(self.query_library));query=library.nl_nanoisa_array_publication
        query.argtypes=[ctypes.c_char_p];query.restype=ctypes.c_int64
        ordinary=b'.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n'
        for source,expected in [(b'not assembly',-1),(None,-1),(ordinary,0),(b' '*((16*1024*1024)+1),0),(ordinary,0)]:
            self.assertEqual(query(source),expected)
        with tempfile.TemporaryDirectory(prefix='nano-flat-query-host-') as directory:
            work=Path(directory);source=work/'query.nano';module=work/'query.nvm'
            source.write_text('module "modules/nanoisa/nanoisa.nano" as Isa\n'
                              'fn main() -> int { assert (== (Isa.array_publication "not assembly") -1) return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            self.run_command(ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module)
            self.run_command(ROOT/'bin/nano_vm',module)
            native=work/'query.c';binary=work/'query'
            self.run_command(ROOT/'bin/nvm2c',module,'-o',native)
            self.run_command(*shlex.split(os.environ.get('CC','cc')),'-std=c11','-O1','-fsanitize=address,undefined',native,'-lm','-o',binary)
            self.run_command(binary)

    def test_existing_array_result_without_array_fields(self):
        source='struct Plain { value: int }\nfn values() -> array<int> { return [7, 11] }\nshadow values { assert (== (array_length (values)) 2) }\nfn main() -> int { assert (== (array_length (values)) 2) return 0 }\nshadow main { assert (== (main) 0) }\n'
        for compiler in self.producers():
            with self.subTest(compiler=compiler),tempfile.TemporaryDirectory(prefix='nano-flat-prior-') as directory:
                work=Path(directory);program=work/'main.nano';module=work/'main.nvm';program.write_text(source)
                self.run_command(compiler,program,'--emit-nvm','-o',module)
                self.run_command(ROOT/'bin/nano_vm','--verify-only',module)
                self.run_command(ROOT/'bin/nano_vm',module)
                dump=self.run_command(ROOT/'bin/nanoisa','dump',module).stdout
                facts=''.join(re.findall(r'^\.ownership "([0-9a-f]*)"$',dump,re.M))
                if Path(compiler).name=='nano_virt':
                    self.assertEqual(facts,'010000000100000001000000020000000000000007000000ffffffff0000000001000000ffffffff')
                else:
                    self.assertEqual(facts,'')

    def test_five_scalar_elements(self):
        for name,value,tag in [('int','7',1),('u8','octet_value',2),('float','7.5',3),('bool','true',4),('string','"seven"',5)]:
            with self.subTest(element=name):
                self.exercise(f'''struct Box {{ values: array<{name}> }}
fn main() -> int {{ let octet_value: u8 = 7
 let box: Box = Box {{ values: [{value}] }}
 assert (== (at box.values 0) {value}) return 0 }}
shadow main {{ assert (== (main) 0) }}
''',[tag])

    def test_shared_and_empty_fields(self):
        self.exercise('''struct Box { first: array<int>, second: array<int>, empty: array<int> }
fn main() -> int { let values: array<int> = [7]
 let box: Box = Box { first: values, second: values, empty: [] }
 (array_set values 0 11)
 assert (== (at box.first 0) 11) assert (== (at box.second 0) 11)
 assert (== (array_length box.empty) 0) return 0 }
shadow main { assert (== (main) 0) }
''',[1,1,1])

    def test_nested_int_arrays_publish_exact_authority(self):
        self.exercise('''struct Box { values: array<array<int>> }
fn main() -> int { let box: Box = Box { values: [[7]] }
 assert (== (at (at box.values 0) 0) 7) return 0 }
shadow main { assert (== (main) 0) }
''',[7])

    def test_nested_aliases_empty_fields_and_generic_forwarder(self):
        self.exercise('''struct Box { values: array<array<int>>, empty: array<array<int>> }
fn identity(value:T)->T { return value }
shadow identity { assert (== (identity 7) 7) }
fn main()->int {
 let child: array<int> = [7]
 let outer: array<array<int>> = [child, child]
 let box: Box = (identity Box { values: outer, empty: [] })
 (array_set child 0 11)
 assert (== (at (at box.values 0) 0) 11)
 assert (== (at (at box.values 1) 0) 11)
 assert (== (array_length box.empty) 0)
 return 0 }
shadow main { assert (== (main) 0) }
''',[7,7])

    def test_nested_string_alias_refill_and_mixed_leaves(self):
        self.exercise('''struct Box { rows:array<array<string>>, numbers:array<array<int>> }
fn identity(value:T)->T { return value }
shadow identity { assert (== (identity 7) 7) }
fn main()->int {
 let child:array<string> =["before"]
 let box:Box=(identity Box {rows:[child,child],numbers:[[7]]})
 (array_set child 0 (+ "af" "ter"))
 assert (== (at (at box.rows 1) 0) "after")
 let empty:array<string> = []
 (array_set box.rows 0 empty)
 let filled:array<string> =(array_push (at box.rows 0) "refilled")
 assert (== (at filled 0) "refilled")
 assert (== (at (at box.rows 1) 0) "after")
 assert (== (at (at box.numbers 0) 0) 7) return 0 }
shadow main { assert (== (main) 0) }
''',[8,7])

    def test_nested_string_forward_record_dag(self):
        self.exercise('''struct Root { child:Box }
struct Box { rows:array<array<string>> }
fn main()->int { let root:Root=Root {child:Box {rows:[["kept"]]}}
 assert (== (at (at root.child.rows 0) 0) "kept") return 0 }
shadow main { assert (== (main) 0) }
''',[8],layout_count=2,array_layout=1)

    def test_nested_string_bound_import(self):
        previous=self.compilers
        self.compilers=os.environ.get('NANO_FLAT_ARRAY_BOUND_COMPILERS',str(ROOT/'bin/nano_virt')).split(',')
        try:
            self.exercise('''module "types.nano" as types
fn main()->int { let box:types.Box=types.Box {rows:[["kept"]]}
 assert (== (at (at box.rows 0) 0) "kept") return 0 }
shadow main { assert (== (main) 0) }
''',[8],modules={'types.nano':'pub struct Box {rows:array<array<string>>}\n'})
        finally:
            self.compilers=previous

    def test_nested_forward_record_dag(self):
        self.exercise('''struct Root { child: Box }
struct Box { values: array<array<int>> }
fn main()->int { let root:Root = Root { child: Box { values:[[7]] } }
 assert (== (at (at root.child.values 0) 0) 7) return 0 }
shadow main { assert (== (main) 0) }
''',[7],layout_count=2,array_layout=1)

    def test_nested_bound_import(self):
        previous=self.compilers
        self.compilers=os.environ.get('NANO_FLAT_ARRAY_BOUND_COMPILERS',str(ROOT/'bin/nano_virt')).split(',')
        try:
            self.exercise('''module "types.nano" as types
fn main()->int { let box:types.Box=types.Box { values:[[7]] }
 assert (== (at (at box.values 0) 0) 7) return 0 }
shadow main { assert (== (main) 0) }
''',[7],modules={'types.nano':'pub struct Box { values: array<array<int>> }\n'})
        finally:
            self.compilers=previous

    def test_nested_primitive_exact_leaves(self):
        for leaf,value,code in [('u8','300',9),('float','7.25',10),('bool','true',11)]:
            expected='44' if leaf=='u8' else value
            with self.subTest(leaf=leaf):
                self.exercise(f"""struct Box {{ values:array<array<{leaf}>> }}
fn main()->int {{ let box:Box=Box {{values:[[{value}],[]]}}
 assert (== (at (at box.values 0) 0) {expected})
 assert (== (array_length (at box.values 1)) 0) return 0 }}
shadow main {{ assert (== (main) 0) }}
""",[code])

    def test_nested_primitive_aliases_and_generic_forwarder(self):
        for leaf,first,second,code in [('u8','7','11',9),('float','7.25','11.5',10),('bool','true','false',11)]:
            with self.subTest(leaf=leaf):
                self.exercise(f"""struct Box {{values:array<array<{leaf}>>,empty:array<array<{leaf}>>}}
fn identity(value:T)->T {{ return value }}
shadow identity {{ assert (== (identity 7) 7) }}
fn main()->int {{ let first:{leaf}={first} let second:{leaf}={second}
 let child:array<{leaf}> = [first]
 let outer:array<array<{leaf}>> = [child,child]
 let box:Box=(identity Box {{values:outer,empty:[]}})
 (array_set child 0 second)
 assert (== (at (at box.values 0) 0) second)
 assert (== (at (at box.values 1) 0) second)
 let empty:array<{leaf}> = []
 (array_set box.values 0 empty)
 assert (== (array_length (at box.values 0)) 0)
 assert (== (at (at box.values 1) 0) second)
 assert (== (array_length box.empty) 0) return 0 }}
shadow main {{ assert (== (main) 0) }}
""",[code,code])

    def test_nested_primitive_bound_imports(self):
        previous=self.compilers
        self.compilers=os.environ.get('NANO_FLAT_ARRAY_BOUND_COMPILERS',str(ROOT/'bin/nano_virt')).split(',')
        try:
            for leaf,value,expected,code in [('u8','300','44',9),('float','7.25','7.25',10),('bool','true','true',11)]:
                with self.subTest(leaf=leaf):
                    self.exercise(f"""module "types.nano" as types
fn main()->int {{ let box:types.Box=types.Box {{values:[[{value}]]}}
 assert (== (at (at box.values 0) 0) {expected}) return 0 }}
shadow main {{ assert (== (main) 0) }}
""",[code],modules={'types.nano':f'pub struct Box {{values:array<array<{leaf}>>}}\n'})
        finally:
            self.compilers=previous

    def test_previously_unpublished_bool_source(self):
        self.exercise("""struct Entries { values: array<array<bool>> }
fn main() -> int { let entries: Entries = Entries { values: [[false, true]] }
 assert (at (at entries.values 0) 1) return 0 }
shadow main { assert (== (main) 0) }
""",[11])

    def test_unchanged_nested_float_contexts_publish(self):
        self.exercise((ROOT/'tests/nanoisa/fixtures/nested_array_contexts.nano').read_text(),
                      [10,7],layout_count=2,array_layout=[(0,0),(1,0)])

    def test_nested_u8_contextual_wrap_boundaries(self):
        self.exercise("""struct Box {values:array<array<u8>>}
fn main()->int {let box:Box=Box {values:[[-1,256,300]]}
 assert (== (at (at box.values 0) 0) 255)
 assert (== (at (at box.values 0) 1) 0)
 assert (== (at (at box.values 0) 2) 44) return 0}
shadow main {assert (== (main) 0)}
""",[9])

    def test_nested_five_leaves_coexist(self):
        self.exercise("""struct Box { ints:array<array<int>>, strings:array<array<string>>,
 octets:array<array<u8>>, floats:array<array<float>>, flags:array<array<bool>> }
fn main()->int { let box:Box=Box {ints:[[7]],strings:[["seven"]],octets:[[300]],floats:[[7.25]],flags:[[true]]}
 assert (== (at (at box.ints 0) 0) 7)
 assert (== (at (at box.strings 0) 0) "seven")
 assert (== (at (at box.octets 0) 0) 44)
 assert (== (at (at box.floats 0) 0) 7.25)
 assert (at (at box.flags 0) 0) return 0 }
shadow main { assert (== (main) 0) }
""",[7,8,9,10,11])

    def test_depth_three_exact_leaves(self):
        for leaf,value,expected,code in [('int','7','7',12),('u8','300','44',13),
                ('float','7.25','7.25',14),('bool','true','true',15),('string','"seven"','"seven"',16)]:
            with self.subTest(leaf=leaf):
                self.exercise(f"""struct Box {{ values:array<array<array<{leaf}>>> }}
fn main()->int {{ let box:Box=Box {{values:[[[{value}],[]],[]]}}
 assert (== (at (at (at box.values 0) 0) 0) {expected})
 assert (== (array_length (at (at box.values 0) 1)) 0)
 assert (== (array_length (at box.values 1)) 0) return 0 }}
shadow main {{assert (== (main) 0)}}
""",[code])

    def test_unchanged_depth_three_bool_anchor(self):
        self.exercise("""struct Entries { values: array<array<array<bool>>> }
fn main() -> int { let entries: Entries = Entries { values: [[[false, true]]] }
 assert (at (at (at entries.values 0) 0) 1) return 0 }
shadow main { assert (== (main) 0) }
""",[15])

    def test_depth_three_aliases_and_typed_empty(self):
        self.exercise("""struct Box {values:array<array<array<string>>>, empty:array<array<array<string>>>}
fn identity(value:T)->T {return value}
shadow identity {assert (== (identity 7) 7)}
fn main()->int {
 let child:array<string> = [(+ "se" "ven")]
 let middle:array<array<string>> = [child,child]
 let outer:array<array<array<string>>> = [middle,middle]
 let box:Box=(identity Box {values:outer,empty:[]})
 (array_set child 0 (+ "ele" "ven"))
 assert (== (at (at (at box.values 1) 1) 0) "eleven")
 let empty:array<array<string>> = []
 (array_set box.values 0 empty)
 assert (== (array_length (at box.values 0)) 0)
 assert (== (at (at (at box.values 1) 0) 0) "eleven")
 assert (== (array_length box.empty) 0) return 0 }
shadow main {assert (== (main) 0)}
""",[16,16])

    def test_depth_three_mixed_depth_tables(self):
        fields=[];values=[];checks=[];codes=[]
        for leaf,value,expected,old,new in [('int','7','7',7,12),('u8','300','44',9,13),
                ('float','7.25','7.25',10,14),('bool','true','true',11,15),('string','"seven"','"seven"',8,16)]:
            for depth,code in ((2,old),(3,new)):
                name=leaf+str(depth);fields.append(name+':'+('array<'*depth)+leaf+('>'*depth))
                values.append(name+':'+('['*depth)+value+(']'*depth));codes.append(code)
                expression='box.'+name
                for _ in range(depth): expression='(at '+expression+' 0)'
                checks.append('assert (== '+expression+' '+expected+')')
        self.exercise('struct Box {'+','.join(fields)+'}\nfn main()->int {let box:Box=Box {'+
            ','.join(values)+'}\n'+'\n'.join(checks)+' return 0}\nshadow main {assert (== (main) 0)}\n',codes)

    def test_depth_three_bound_imports(self):
        previous=self.compilers
        self.compilers=os.environ.get('NANO_FLAT_ARRAY_BOUND_COMPILERS',str(ROOT/'bin/nano_virt')).split(',')
        try:
            self.exercise("""module "types.nano" as types
fn main()->int {let box:types.Box=types.Box {values:[[[true]]]}
 assert (at (at (at box.values 0) 0) 0) return 0}
shadow main {assert (== (main) 0)}
""",[15],modules={'types.nano':'pub struct Box {values:array<array<array<bool>>>}\n'})
        finally:
            self.compilers=previous

    def test_unsupported_nested_descriptors_stay_optional(self):
        for declared,value in [('array<array<array<array<int>>>>','[[[[7]]]]')]:
            with self.subTest(declared=declared):
                self.exercise('struct Box { values:'+declared+' }\nfn main()->int { let box:Box=Box { values:'+value+' } return 0 }\nshadow main { assert true }\n',None,run=False)
        self.exercise('extern fn get_argc()->int\nstruct Box { values:array<array<int>> }\nfn main()->int { let box:Box=Box { values:[[7]] } unsafe { return (get_argc) } }\nshadow main { assert true }\n',None,run=False)

    def test_forward_record_dag(self):
        self.exercise('struct Root { child: Box }\nstruct Box { values: array<int> }\nfn main() -> int { let root: Root = Root { child: Box { values: [7] } }\n assert (== (at root.child.values 0) 7) return 0 }\nshadow main { assert (== (main) 0) }\n',[1],layout_count=2,array_layout=1)

    def test_bound_imports(self):
        previous=self.compilers
        self.compilers=os.environ.get('NANO_FLAT_ARRAY_BOUND_COMPILERS',str(ROOT/'bin/nano_virt')).split(',')
        try:
            self.exercise('module "types.nano" as types\nfn main() -> int { let row: types.Entry = types.Entry { values: [7, 11] }\n assert (== (at row.values 1) 11) return 0 }\nshadow main { assert (== (main) 0) }\n',[1],modules={'types.nano':'pub struct Entry { values: array<int> }\n'})
        finally:
            self.compilers=previous

    def test_large_closed_module_omits_optional_envelope(self):
        functions='\n'.join(f'fn chain_{i}() -> int {{ return '+(f'(chain_{i+1})' if i<259 else '0')+' }' for i in range(260))
        self.exercise('struct Box { values: array<int> }\n'+functions+\
            '\nfn main() -> int { let box: Box = Box { values: [7] } assert (== (at box.values 0) 7) return (chain_0) }\nshadow main { assert true }\n',None,run=False)

    def test_foreign_module_omits_optional_envelope(self):
        self.exercise('extern fn get_argc() -> int\nstruct Box { values: array<int> }\nfn main() -> int { let box: Box = Box { values: [7] }\n assert (== (at box.values 0) 7) unsafe { return (get_argc) } }\nshadow main { assert true }\n',None,run=False)


    def test_concrete_generic_array_record_arguments_and_results(self):
        self.exercise('''struct Box { values: array<int> }
fn identity(value: T) -> T { let copy: T = value return copy }
shadow identity { assert (== (identity 7) 7) }
fn forward(value: T) -> T { let copy: T = (identity value) let result: T = (identity copy) return result }
shadow forward { assert (== (forward true) true) }
fn main() -> int {
 let values: array<int> = [7, 11]
 let box: Box = (forward Box { values: values })
 let copy: Box = (identity box)
 assert (== (at copy.values 1) 11)
 return 0
}
shadow main { assert (== (main) 0) }
''',[1])
    def test_generic_direct_tail_calls_publish_proved_array_profile(self):
        self.exercise('''struct Box { values: array<int> }
fn identity(value: T) -> T { let copy: T = value return copy }
shadow identity { assert (== (identity 7) 7) }
fn forward(value: T) -> T { let copy: T = (identity value) return (identity copy) }
shadow forward { assert (== (forward true) true) }
fn main() -> int {
 let values: array<int> = [7, 11]
 let box: Box = (forward Box { values: values })
 let copy: Box = (identity box)
 assert (== (at copy.values 1) 11)
 return 0
}
shadow main { assert (== (main) 0) }
''',[1],expected_opcode='TAIL_CALL')

    def test_self_tail_calls_publish_proved_array_authority(self):
        self.exercise('''struct Box { values: array<int> }
fn repeat(value: Box, remaining: int) -> Box {
 if (<= remaining 0) { return value }
 return (repeat value (- remaining 1))
}
shadow repeat {
 let box: Box = (repeat Box { values: [7] } 2)
 assert (== (at box.values 0) 7)
}
fn main() -> int {
 let box: Box = (repeat Box { values: [7, 11] } 3)
 assert (== (at box.values 1) 11)
 return 0
}
shadow main { assert (== (main) 0) }
''',[1],expected_opcode='TAIL_CALL')

if __name__=='__main__':
    unittest.main()
