"""I qualify exact source bit transport through fresh compiler producers."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class Binary64SourceTransport(unittest.TestCase):
    def command(self,*args):
        p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=180)
        self.assertEqual(p.returncode,0,f'{args}\n{p.stdout}\n{p.stderr}')
        return p
    def test_source_producers_and_legacy_c(self):
        source=ROOT/'tests/nanoisa/fixtures/binary64_bits.nano'
        with tempfile.TemporaryDirectory(prefix='binary64-source-') as tmp:
            work=Path(tmp)
            for name in ('nano_virt','nanoc_stage1','nanoc_stage2'):
                with self.subTest(producer=name):
                    module=work/(name+'.nvm')
                    self.command(ROOT/'bin'/name,source,'--emit-nvm','-o',module)
                    assembly=self.command(ROOT/'bin/nanoisa','disasm',module).stdout
                    self.assertIn('F64_FROM_BITS',assembly)
                    self.assertIn('F64_TO_BITS',assembly)
                    self.command(ROOT/'bin/nano_vm','--verify-only',module)
                    self.command(ROOT/'bin/nano_vm',module)
                    c=work/(name+'.c');exe=work/name
                    self.command(ROOT/'bin/nvm2c',module,'-o',c)
                    self.command(os.environ.get('CC','cc'),'-std=c11','-O2','-Wall','-Wextra','-Werror',
                                 '-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',exe)
                    self.command(exe)
            for name in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
                with self.subTest(legacy=name):
                    exe=work/(name+'-legacy')
                    self.command(ROOT/'bin'/name,source,'-o',exe)
                    self.command(exe)
    def test_reserved_declarations_and_bound_callable_refusals(self):
        cases = (
            'fn float_from_bits(x:int)->int{return x} shadow float_from_bits {assert true} fn main()->int{return 0}',
            'fn float_to_bits(x:int)->int{return x} shadow float_to_bits {assert true} fn main()->int{return 0}',
            'fn identity(x:int)->int{return x} shadow identity {assert true} fn main()->int{let float_from_bits:fn(int)->int = identity return (float_from_bits 1)}',
            'fn identity(x:float)->int{return 7} shadow identity {assert true} fn main()->int{let float_to_bits:fn(float)->int = identity return (float_to_bits 1.0)}',
        )
        with tempfile.TemporaryDirectory(prefix='binary64-name-refusal-') as tmp:
            work=Path(tmp);source=work/'names.nano';output=work/'retained'
            for body in cases:
                source.write_text(body+'\nshadow main { assert true }\n')
                for name in ('nanoc_c','nano_virt','nanoc_stage1','nanoc_stage2'):
                    with self.subTest(compiler=name,source=body):
                        output.write_bytes(b'retained')
                        args=[ROOT/'bin'/name,source,'-o',output]
                        if name!='nanoc_c':args.insert(2,'--emit-nvm')
                        p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=180)
                        self.assertGreater(p.returncode,0,f'{args}\n{p.stdout}\n{p.stderr}')
                        self.assertRegex(p.stdout+p.stderr,'(?i)(built.?in|intrinsic)')
                        self.assertEqual(output.read_bytes(),b'retained')

    def test_source_exact_type_and_arity_refusals(self):
        cases = ('(float_from_bits true)', '(float_from_bits 1.0)',
                 '(float_to_bits 1)', '(float_to_bits false)',
                 '(float_from_bits)', '(float_to_bits 1.0 2.0)')
        with tempfile.TemporaryDirectory(prefix='binary64-source-refusal-') as tmp:
            work=Path(tmp); source=work/'wrong.nano'; output=work/'retained'
            for expression in cases:
                source.write_text('fn main() -> int { let value: int = '+expression+' return 0 }\nshadow main { assert true }\n')
                for name in ('nanoc_c','nano_virt','nanoc_stage1','nanoc_stage2'):
                    with self.subTest(compiler=name,expression=expression):
                        output.write_bytes(b'retained')
                        args=[ROOT/'bin'/name,source,'-o',output]
                        if name!='nanoc_c': args.insert(2,'--emit-nvm')
                        p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=180)
                        self.assertGreater(p.returncode,0,f'{args}\n{p.stdout}\n{p.stderr}')
                        self.assertEqual(output.read_bytes(),b'retained')

if __name__=='__main__':unittest.main()
