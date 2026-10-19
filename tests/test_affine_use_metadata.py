"""I qualify retained checker uses without granting execution authority."""
import ast
import os
from pathlib import Path
import shlex
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

    def test_actual_checker_epoch_and_raw_boundary(self):
        with tempfile.TemporaryDirectory(prefix='nano-affine-use-checker-') as directory:
            work=Path(directory); source=work/'checker.nano'; binary=work/'checker'
            source.write_text((ROOT/'tests/nanoisa/fixtures/affine_use_checker_facts.nano.txt').read_text())
            compiler=Path(os.environ.get('NANOLANG_AFFINE_USE_NATIVE_COMPILER',ROOT/'bin/nanoc_c'))
            self.command([compiler,source,'-o',binary],build=True,
                env={**os.environ,'NANO_SHADOW_TIMEOUT_SECONDS':'300'})
            self.command([binary])

if __name__=='__main__': unittest.main()
