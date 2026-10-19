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
                    rows=[]
                    for _ in range(types):
                        tag,ref=struct.unpack_from('<B3xI',facts,at);at+=8
                        self.assertEqual(ref,0xffffffff);rows.append(tag)
                    bindings=struct.unpack_from('<I',facts,at)[0];at+=4
                    actual=[]
                    for field in range(bindings):
                        layout,ordinal,reserved,element=struct.unpack_from('<IHHI',facts,at);at+=12
                        self.assertEqual((layout,ordinal,reserved),(array_layout,field,0));actual.append(rows[element])
                    self.assertEqual(actual,elements);self.assertEqual(at,len(facts))
                    library_path=next((ROOT/'modules/nanoisa/.build/current').glob('libnanoisa.*'))
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
        libraries=list((ROOT/'modules/nanoisa/.build/current').glob('libnanoisa.*'))
        self.assertEqual(len(libraries),1)
        library=ctypes.CDLL(str(libraries[0]));query=library.nl_nanoisa_array_publication
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

    def test_nested_arrays_remain_unpublished(self):
        self.exercise('''struct Box { values: array<array<int>> }
fn main() -> int { let box: Box = Box { values: [[7]] }
 assert (== (at (at box.values 0) 0) 7) return 0 }
shadow main { assert (== (main) 0) }
''',None)

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

    def test_recursive_tail_calls_keep_optional_array_authority_absent(self):
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
''',None,expected_opcode='TAIL_CALL')

if __name__=='__main__':
    unittest.main()
