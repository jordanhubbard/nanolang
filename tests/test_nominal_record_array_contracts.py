"""I check nominal array identity at declared C-frontend boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
DECL='struct First { a: int } struct Second { a: int } struct Holder { values: array<First> } union Values { Some { values: array<First> } }\n'
END='\nshadow main { assert (== (main) 0) }\n'
class NominalRecordArrays(unittest.TestCase):
    def compile(self, compiler, source, output):
        cmd=[ROOT/'bin'/compiler,source,'-o',output]
        if compiler=='nano_virt':cmd.append('--emit-nvm')
        return subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=120)
    def test_wrong_nominal_contexts(self):
        programs=[
          ("fn main() -> int { let value: Values = Values.Some { values: [Second { a: 2 }] } return 0 }", 'union payload value to match its complete concrete destination'),
          ('fn main() -> int { let values: array<First> = (array_new 2 Second { a: 2 }) return 0 }', 'declared nominal record type'),
          ('fn main() -> int { let wrong: array<Second> = [Second { a: 2 }] let values: array<First> = (array_slice wrong 0 1) return 0 }', 'declared nominal record type'),
          ('fn main() -> int { let mut values: array<First> = [] set values [Second { a: 2 }] return 0 }', 'declared nominal record type'),
          ('fn main() -> int { let values: array<First> = [Second { a: 2 }] return 0 }', 'declared nominal record type'),
          ('let values: array<First> = [Second { a: 2 }] fn main() -> int { return 0 }', 'declared nominal record type'),
          ('fn main() -> int { let holder: Holder = Holder { values: [Second { a: 2 }] } return 0 }', 'declared nominal record type'),
          ('fn values() -> array<First> { return [Second { a: 2 }] } shadow values { assert true } fn main() -> int { return 0 }', 'declared nominal record type'),
          ('fn take(values: array<First>) -> int { return 0 } shadow take { assert true } fn main() -> int { return (take [Second { a: 2 }]) }', 'declared nominal record type'),
          ('fn main() -> int { let wrong: array<Second> = [Second { a: 2 }] let values: array<First> = wrong return 0 }', 'declared nominal record type'),
          ('fn main() -> int { let values: array<First> = (array_push [] Second { a: 2 }) return 0 }', 'declared nominal record type'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
          root=Path(tmp);source=root/'bad.nano';output=root/'prior'
          for i,(program,diagnostic) in enumerate(programs):
            source.write_text(DECL+program+END)
            for compiler in ('nanoc_c','nano_virt'):
              with self.subTest(case=i,compiler=compiler):
                output.write_text('prior-output');p=self.compile(compiler,source,output)
                self.assertNotEqual(p.returncode,0)
                self.assertIn(diagnostic,p.stdout+p.stderr)
                self.assertEqual(output.read_text(),'prior-output')
    def test_matching_and_empty_contexts_execute(self):
        source_text=DECL+'''let initial: array<First> = [First { a: 7 }]
fn take(values: array<First>) -> int { return (array_length values) }
shadow take { assert (== (take []) 0) }
fn values() -> array<First> { return [First { a: 2 }] }
shadow values { assert (== (take (values)) 1) }
fn main() -> int {
 let filled: array<First> = (array_new 2 First { a: 8 })
 let sliced: array<First> = (array_slice filled 0 1)
 let pushed: array<First> = (array_push (array_push [] First { a: 9 }) First { a: 10 })
 assert (== (array_length sliced) 1)
 assert (== (array_length pushed) 2)
 assert (== (at pushed 1).a 10)
 let empty: array<First> = []
 let holder: Holder = Holder { values: initial }
 assert (== (take holder.values) 1)
 assert (== (take empty) 0)
 assert (== (at (values) 0).a 2)
 return 0
}'''+END
        with tempfile.TemporaryDirectory() as tmp:
          root=Path(tmp);source=root/'good.nano';source.write_text(source_text)
          for compiler in ('nanoc_c','nano_virt'):
            with self.subTest(compiler=compiler):
              output=root/compiler;p=self.compile(compiler,source,output)
              self.assertEqual(p.returncode,0,p.stdout+p.stderr)
              cmd=[output] if compiler=='nanoc_c' else [ROOT/'bin/nano_vm',output]
              p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30)
              self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def test_imported_record_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
          root=Path(tmp)
          (root/'left.nano').write_text('module Left\npub struct Entry { a: int }\npub fn entries() -> array<Entry> { return [Entry { a: 7 }] }\nshadow entries { assert (== (array_length (entries)) 1) }\npub fn take(values: array<Entry>) -> int { return (array_length values) }\nshadow take { assert (== (take []) 0) }\n')
          (root/'right.nano').write_text('module Right\npub struct Entry { a: int }\n')
          for wrong in (False,True):
            source=root/'main.nano';output=root/'result'
            source.write_text('module "left.nano" as left\nmodule "right.nano" as right\nfn main() -> int { let values: array<left.Entry> = (left.entries) assert (== (left.take values) 1) assert (== (left.take ['+('right' if wrong else 'left')+'.Entry { a: 3 }]) 1) return 0 }'+END)
            for compiler in ('nanoc_c','nano_virt'):
              with self.subTest(wrong=wrong,compiler=compiler):
                output.write_text('prior-output');p=self.compile(compiler,source,output)
                if wrong:
                  self.assertNotEqual(p.returncode,0)
                  self.assertIn('declared nominal record type',p.stdout+p.stderr)
                  self.assertEqual(output.read_text(),'prior-output')
                else:
                  self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                  cmd=[output] if compiler=='nanoc_c' else [ROOT/'bin/nano_vm',output]
                  p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30)
                  self.assertEqual(p.returncode,0,p.stdout+p.stderr)
if __name__=='__main__':unittest.main()
