"""I qualify retained checker uses without granting execution authority."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import shutil
import struct
import tempfile
import unittest
from tests.native_process import run_captured, native_compile_timeout

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [Path(p) for p in os.environ.get('NANOLANG_AFFINE_USE_COMPILERS',
    str(ROOT/'bin/nano_virt')).split(os.pathsep) if p]
KEY = b'nano.affine-use.v1'

class AffineUseMetadata(unittest.TestCase):
    def command(self, args, expected=0, build=False, env=None):
        result = run_captured(list(map(str,args)), cwd=ROOT, env=env, text=True,
            timeout=native_compile_timeout() if build else 180)
        self.assertEqual(result.returncode, expected, result.stdout+result.stderr)
        return result.stdout

    def test_actual_observe_consume_and_advisory_independence(self):
        source = '''resource struct R { value:int }
fn take(x:R)->int { let prior:int=x.value let R{value}=x return (+ prior value) }
fn other(x:R)->int { let R{value}=x return value }
fn main()->int { let a:R=R{value:7} let b:R=R{value:9}
assert (== (take a) 14) assert (== (other b) 9) return 0 }
'''
        for driver in DRIVERS:
            with self.subTest(driver=driver), tempfile.TemporaryDirectory(prefix='nano-affine-use-') as directory:
                work=Path(directory); src=work/'case.nano'; module=work/'case.nvm'
                src.write_text(source)
                self.command([driver,src,'--emit-nvm','-o',module])
                canonical=self.command([ROOT/'bin/nanoisa','dump',module])
                lines=canonical.splitlines()
                strings=[ast.literal_eval(line[8:]).encode('latin1') for line in lines if line.startswith('.string ')]
                functions=[shlex.split(line)[1] for line in lines if line.startswith('.function ')]
                keys={i for i,value in enumerate(strings) if value==KEY}
                entries=[line for line in lines if line.startswith('.metadata ') and int(line.split()[1]) in keys]
                self.assertTrue(entries)
                observed={}
                sites=set()
                for entry in entries:
                    value=strings[int(entry.split()[2])]
                    self.assertGreaterEqual(len(value),24)
                    fn,line,column,state,coords,reserved,path_length,name_length=struct.unpack('<IIIBBHII',value[:24])
                    self.assertEqual(reserved,0); self.assertIn(state,(1,2));self.assertIn(coords,(1,2))
                    self.assertGreater(line,0);self.assertGreater(column,0)
                    self.assertEqual(len(value),24+path_length+name_length)
                    name=value[24+path_length:].decode()
                    self.assertTrue(name)
                    site=(fn,line,column,coords,value[24:24+path_length])
                    self.assertNotIn(site,sites);sites.add(site)
                    observed.setdefault(functions[fn],set()).add((name,state))
                self.assertEqual(observed['take'],{('x',1),('x',2)})
                self.assertEqual(observed['other'],{('x',2)})
                self.assertEqual(observed['main'],{('a',2),('b',2)})
                assembly=work/'case.nasm'; rebuilt=work/'roundtrip.nvm'
                assembly.write_text(canonical)
                self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt])
                self.assertEqual(module.read_bytes(),rebuilt.read_bytes())
                self.command([ROOT/'bin/nano_vm',module])
                native=work/'case.c'; binary=work/'case'
                self.command([ROOT/'bin/nvm2c',module,'-o',native])
                original=native.read_bytes()
                self.command(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined',str(native),'-lm','-o',str(binary)],build=True)
                self.command([binary])
                forged=lines.copy(); entry=entries[0]; payload=bytearray(strings[int(entry.split()[2])]);payload[12]=0
                payload=bytes(payload)
                if payload in strings: replacement=strings.index(payload)
                else:
                    replacement=len(strings)
                    position=next(i for i,line in enumerate(forged) if line.startswith('.metadata '))
                    forged.insert(position,'.string "'+''.join(f'\\x{b:02x}' for b in payload)+'"')
                forged[forged.index(entry)]=f'.metadata {entry.split()[1]} {replacement}'
                for variant in ([line for line in lines if line not in entries],lines+[entries[0]],forged):
                    assembly.write_text('\n'.join(variant)+'\n')
                    self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt])
                    self.command([ROOT/'bin/nano_vm',rebuilt])
                    self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
                    self.assertEqual(original,native.read_bytes())

    def test_borrowed_resource_roots_observe_without_consuming(self):
        for fixture, function, count in (('source_borrow_shared.nano', 'read', 1),
                                         ('source_borrow_exclusive.nano', 'update', 3)):
            for driver in DRIVERS:
                with self.subTest(fixture=fixture, driver=driver), tempfile.TemporaryDirectory(prefix='nano-affine-borrowed-') as directory:
                    work=Path(directory); module=work/'source.nvm'
                    source=ROOT/'tests/nanoisa/fixtures'/fixture
                    self.command([driver,source,'--emit-nvm','-o',module])
                    canonical=self.command([ROOT/'bin/nanoisa','dump',module])
                    lines=canonical.splitlines()
                    strings=[ast.literal_eval(line[8:]).encode('latin1') for line in lines if line.startswith('.string ')]
                    functions=[shlex.split(line)[1] for line in lines if line.startswith('.function ')]
                    keys={i for i,value in enumerate(strings) if value==KEY}
                    entries=[line for line in lines if line.startswith('.metadata ') and int(line.split()[1]) in keys]
                    roots=[]
                    for entry in entries:
                        value=strings[int(entry.split()[2])]
                        fn,line,column,state,coords,reserved,path_length,name_length=struct.unpack('<IIIBBHII',value[:24])
                        self.assertEqual(len(value),24+path_length+name_length)
                        self.assertEqual(reserved,0)
                        name=value[24+path_length:].decode()
                        if functions[fn]==function:
                            roots.append((name,state,line,column,coords))
                    self.assertEqual(len(roots), count, roots)
                    self.assertTrue(all(name=='view' and state==1 for name,state,_,_,_ in roots), roots)
                    self.assertTrue(all(line>0 and column>0 and coords in (1,2)
                                        for _,_,line,column,coords in roots))
                    self.command([ROOT/'bin/nano_vm',module])
                    native=work/'source.c';binary=work/'source'
                    self.command([ROOT/'bin/nvm2c',module,'-o',native])
                    original=native.read_bytes()
                    self.command(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O2','-Wall','-Wextra','-Werror',
                        '-fsanitize=address,undefined','-fno-sanitize-recover=all',native,'-lm','-o',binary],build=True)
                    self.command([binary])
                    assembly=work/'stripped.nasm';stripped=work/'stripped.nvm'
                    assembly.write_text('\n'.join(line for line in lines if line not in entries)+'\n')
                    self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',stripped])
                    self.command([ROOT/'bin/nano_vm',stripped])
                    self.command([ROOT/'bin/nvm2c',stripped,'-o',native])
                    self.assertEqual(original,native.read_bytes())

    def test_borrowed_resource_misuse_preserves_output(self):
        for driver in DRIVERS:
            for annotation,body in (('&Counter', 'set view.value 8 return view.value'),
                                    ('&mut Counter', 'let Counter { value } = view return value'),
                                    ('&Unknown', 'return view.value'),
                                    ('&&Counter', 'return view.value'),
                                    ('&mut &Counter', 'return view.value')):
                with self.subTest(driver=driver, annotation=annotation), tempfile.TemporaryDirectory(prefix='nano-affine-borrow-refusal-') as directory:
                    work=Path(directory);source=work/'bad.nano';output=work/'prior.nvm'
                    source.write_text('resource struct Counter { value:int }\nfn invalid(view:'+annotation+
                        ')->int{'+body+'}\nshadow invalid { assert true }\nfn main()->int{return 0}\n')
                    output.write_bytes(b'prior qualified output')
                    result=run_captured([driver,source,'--emit-nvm','-o',output],cwd=ROOT,text=True,timeout=180)
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(),b'prior qualified output')

    def test_invalid_resource_uses_preserve_output(self):
        prelude='resource struct R{value:int} fn take(x:R)->int{let R{value}=x return value} '
        for driver in DRIVERS:
            for body in ['let x:R=R{value:1} let n:int=(take x) return (take x)',
                         'let x:R=R{value:1} let n:int=(take x) return x.value']:
                with self.subTest(driver=driver,body=body), tempfile.TemporaryDirectory(prefix='nano-affine-use-refusal-') as directory:
                    work=Path(directory); src=work/'bad.nano'; output=work/'prior.nvm'
                    src.write_text(prelude+'fn main()->int{'+body+'}')
                    output.write_bytes(b'prior qualified output')
                    result=run_captured([driver,src,'--emit-nvm','-o',output],cwd=ROOT,text=True,timeout=180)
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(),b'prior qualified output')

    @contextmanager
    def retain_site_failure(self, directory):
        try:
            yield
        except BaseException:
            root=Path(os.environ.get('NANOLANG_AFFINE_USE_FAILURE_DIR',ROOT/'.test_output/affine-use-failures'))
            root.mkdir(parents=True,exist_ok=True)
            shutil.copytree(directory,root/Path(directory).name,dirs_exist_ok=True)
            raise

    def imported_site_product(self, driver, work, source):
        module=work/'sites.nvm'
        self.command([driver,source,'--emit-nvm','-o',module])
        canonical=self.command([ROOT/'bin/nanoisa','dump',module])
        lines=canonical.splitlines()
        strings=[ast.literal_eval(line[8:]).encode('latin1') for line in lines if line.startswith('.string ')]
        functions=[shlex.split(line)[1] for line in lines if line.startswith('.function ')]
        keys={i for i,value in enumerate(strings) if value==KEY}
        entries=[line for line in lines if line.startswith('.metadata ') and int(line.split()[1]) in keys]
        facts=[]; sites=set()
        for entry in entries:
            value=strings[int(entry.split()[2])]
            self.assertGreaterEqual(len(value),24)
            fn,line,column,state,coords,reserved,path_length,name_length=struct.unpack('<IIIBBHII',value[:24])
            self.assertLess(fn,len(functions)); self.assertEqual(reserved,0)
            self.assertEqual(len(value),24+path_length+name_length)
            self.assertGreater(line,0); self.assertGreater(column,0)
            self.assertEqual(coords,1 if driver.name=='nano_virt' else 2)
            self.assertIn(state,(1,2))
            path=Path(value[24:24+path_length].decode())
            binding=value[24+path_length:].decode()
            site=(fn,path,line,column,coords)
            self.assertNotIn(site,sites); sites.add(site)
            facts.append((fn,path,line,column,state,binding))
        self.assertTrue(facts)
        assembly=work/'roundtrip.nasm'; rebuilt=work/'roundtrip.nvm'
        assembly.write_text(canonical)
        self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt])
        self.assertEqual(module.read_bytes(),rebuilt.read_bytes())
        self.command([ROOT/'bin/nano_vm',module])
        native=work/'sites.c'; binary=work/'sites'
        self.command([ROOT/'bin/nvm2c',module,'-o',native])
        original=native.read_bytes()
        self.command(shlex.split(os.environ.get('CC','cc'))+[
            '-std=c11','-O2','-Wall','-Wextra','-Werror',
            '-fsanitize=address,undefined','-fno-sanitize-recover=all',
            native,'-lm','-o',binary],build=True)
        self.command([binary])
        assembly.write_text('\n'.join(line for line in lines if line not in entries)+'\n')
        self.command([ROOT/'bin/nanoisa','asm',assembly,'-o',rebuilt])
        self.command([ROOT/'bin/nano_vm',rebuilt])
        self.command([ROOT/'bin/nvm2c',rebuilt,'-o',native])
        self.assertEqual(original,native.read_bytes())
        return functions,facts

    def test_imported_same_spelled_checked_sites(self):
        for driver in DRIVERS:
            for reverse in (False,True):
                with self.subTest(driver=driver,reverse=reverse), tempfile.TemporaryDirectory(prefix='nano-affine-site-imports-') as directory, self.retain_site_failure(directory):
                    work=Path(directory)
                    for filename,number in (('left.nano',7),('right.nano',9)):
                        (work/filename).write_text('resource struct Handle { payload:int }\n'
                            'fn take(value:Handle)->int { let observed:int=value.payload let Handle{payload}=value return (+ observed payload) }\n'
                            f'shadow take {{ assert (== (take Handle{{payload:{number}}}) {number*2}) }}\n'
                            f'pub fn run()->int {{ let value:Handle=Handle{{payload:{number}}} return (take value) }}\n'
                            f'shadow run {{ assert (== (run) {number*2}) }}\n')
                    imports=['module "left.nano" as left','module "right.nano" as right']
                    if reverse: imports.reverse()
                    source=work/'main.nano'
                    source.write_text('\n'.join(imports)+'\nfn main()->int{assert (== (left.run) 14) assert (== (right.run) 18) return 0}\nshadow main {assert (== (main) 0)}\n')
                    functions,facts=self.imported_site_product(driver,work,source)
                    groups={}; owners={}
                    for fn,path,line,column,state,binding in facts:
                        self.assertEqual(binding,'value')
                        groups.setdefault(fn,[]).append(state)
                        if driver.name=='nano_virt':
                            self.assertIn(path.name,('left.nano','right.nano'))
                            self.assertEqual(path.resolve(),(work/path.name).resolve())
                            owners.setdefault(path.name,set()).add(fn)
                        else:
                            self.assertEqual(path.resolve(),source.resolve())
                    self.assertEqual(sorted(sorted(states) for states in groups.values()),[[1,2],[1,2],[2],[2]])
                    self.assertEqual(len(groups),4)
                    self.assertEqual(len(set(functions)),len(functions))
                    if driver.name=='nano_virt':
                        self.assertEqual(set(owners),{'left.nano','right.nano'})
                        self.assertEqual(len(owners['left.nano']),2)
                        self.assertEqual(len(owners['right.nano']),2)
                        self.assertTrue(owners['left.nano'].isdisjoint(owners['right.nano']))

    def test_branch_local_checked_sites(self):
        source='''resource struct Handle { payload:int }
fn choose(flag:bool)->int {
 if flag {
  let value:Handle=Handle{payload:7}
  let observed:int=value.payload
  let Handle{payload}=value
  return (+ observed payload)
 } else {
  let value:Handle=Handle{payload:9}
  let observed:int=value.payload
  let Handle{payload}=value
  return (+ observed payload)
 }
}
shadow choose {assert (== (choose true) 14) assert (== (choose false) 18)}
fn main()->int{assert (== (choose true) 14) assert (== (choose false) 18) return 0}
shadow main {assert (== (main) 0)}
'''
        for driver in DRIVERS:
            with self.subTest(driver=driver), tempfile.TemporaryDirectory(prefix='nano-affine-site-branches-') as directory, self.retain_site_failure(directory):
                work=Path(directory); program=work/'branches.nano'; program.write_text(source)
                functions,facts=self.imported_site_product(driver,work,program)
                self.assertEqual(len(facts),4)
                self.assertEqual({functions[fn] for fn,_,_,_,_,_ in facts},{'choose'})
                self.assertEqual({path.resolve() for _,path,_,_,_,_ in facts},{program.resolve()})
                self.assertEqual({binding for _,_,_,_,_,binding in facts},{'value'})
                ordered=sorted((line,column,state) for _,_,line,column,state,_ in facts)
                self.assertEqual([state for _,_,state in ordered],[1,2,1,2])

    def test_imported_and_branch_move_refusals(self):
        for driver in DRIVERS:
            for imported in (False,True):
                with self.subTest(driver=driver,imported=imported), tempfile.TemporaryDirectory(prefix='nano-affine-site-refusal-') as directory, self.retain_site_failure(directory):
                    work=Path(directory); program=work/'main.nano'; output=work/'prior.nvm'
                    bad='resource struct Handle { payload:int }\npub fn broken(flag:bool)->int { if flag { let value:Handle=Handle{payload:7} let Handle{payload}=value return value.payload } return 0 }\n'
                    if imported:
                        (work/'owner.nano').write_text(bad)
                        program.write_text('module "owner.nano" as owner\nfn main()->int{return (owner.broken true)}\n')
                    else: program.write_text(bad+'fn main()->int{return (broken true)}\n')
                    output.write_bytes(b'prior qualified output')
                    result=run_captured([driver,program,'--emit-nvm','-o',output],cwd=ROOT,text=True,timeout=180)
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertRegex(result.stdout+result.stderr,r'(?i)(moved value|after.{0,30}(mov|consum)|use.{0,30}moved)')
                    self.assertEqual(output.read_bytes(),b'prior qualified output')

    def test_resource_synthetic_shadow_entry(self):
        source='resource struct Handle{payload:int}\nfn take(value:Handle)->int{let Handle{payload}=value return payload}\nshadow take {assert (== (take Handle{payload:7}) 7)}\nfn main()->int{return (- (take Handle{payload:7}) 7)}\n'
        for driver in DRIVERS:
            with self.subTest(driver=driver), tempfile.TemporaryDirectory(prefix='nano-affine-site-shadow-') as directory, self.retain_site_failure(directory):
                work=Path(directory); program=work/'main.nano'; output=work/'prior.nvm'
                program.write_text(source)
                self.command([driver,program,'--emit-nvm','-o',output])
                self.command([ROOT/'bin/nano_vm',output])
                program.write_text(source.replace('shadow take {assert (== (take Handle{payload:7}) 7)}','shadow take {assert false}'))
                output.write_bytes(b'prior qualified output')
                result=run_captured([driver,program,'--emit-nvm','-o',output],cwd=ROOT,text=True,timeout=180)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertRegex(result.stdout+result.stderr,r'(?i)shadow')
                self.assertEqual(output.read_bytes(),b'prior qualified output')

    def test_actual_checker_epoch_and_raw_boundary(self):
        with tempfile.TemporaryDirectory(prefix='nano-affine-use-checker-') as directory:
            work=Path(directory); source=work/'checker.nano'; binary=work/'checker'
            source.write_text((ROOT/'tests/nanoisa/fixtures/affine_use_checker_facts.nano.txt').read_text())
            compiler=Path(os.environ.get('NANOLANG_AFFINE_USE_NATIVE_COMPILER',ROOT/'bin/nanoc_c'))
            self.command([compiler,source,'-o',binary],build=True,
                env={**os.environ,'NANO_SHADOW_TIMEOUT_SECONDS':'300'})
            self.command([binary])

if __name__=='__main__': unittest.main()
