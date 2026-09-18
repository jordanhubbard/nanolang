"""I preserve optional lexical names from both ordinary source frontends."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''fn compute(flag: bool) -> int {
    let value: int = 7
    if flag { let value: int = 11 assert (== value 11) }
    else { let value: int = 12 assert (== value 12) }
    let mut count: int = 0
    while (< count 2) { let step: int = (+ count 1) set count step }
    return (+ value count)
}
shadow compute { assert (== (compute true) 9) assert (== (compute false) 9) }
fn main() -> int { return (compute true) }
shadow main { assert (== (main) 9) }
'''
ARRAY = '''fn main() -> int {
    let values: array<int> = [1, 2]
    let answer: int = (array_length values)
    return answer
}
shadow main { assert (== (main) 2) }
'''

class LocalBindingMetadata(unittest.TestCase):
    def run_checked(self, args, expected=0):
        env = os.environ.copy()
        env.setdefault("NANOLANG_SHADOW_TIMEOUT_SECONDS", "60")
        p = subprocess.run(args, capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
        self.assertEqual(p.returncode, expected, p.stdout + p.stderr)
        return p.stdout

    def produce(self, source, expected, inspect):
        with tempfile.TemporaryDirectory(prefix="nano-local-names-") as name:
            tmp=Path(name); input_file=tmp/"case.nano"; input_file.write_text(source)
            for driver in ("nano_virt", "nanoc_stage1", "nanoc_stage2"):
                with self.subTest(driver=driver):
                    artifact=tmp/f"{driver}.nvm"
                    self.run_checked([ROOT/"bin"/driver,input_file,"--emit-nvm","-o",artifact])
                    # The C probe validates every interval, exact metadata/code
                    # preservation, original wire equality and canonical-cycle stability.
                    lines=self.run_checked([ROOT/"obj/test_local_bindings",artifact]).splitlines()
                    bindings=[(f,int(slot),int(begin),int(end),local) for f,slot,begin,end,local in (line.split() for line in lines)]
                    inspect(bindings)
                    self.run_checked([ROOT/"bin/nano_vm",artifact],expected)
                    native=tmp/f"{driver}.c"; binary=tmp/f"{driver}.out"
                    self.run_checked([ROOT/"bin/nvm2c",artifact,"-o",native])
                    self.run_checked(shlex.split(os.environ.get("CC","cc"))+["-std=c11","-Wall","-Wextra","-Werror",str(native),"-lm","-o",str(binary)])
                    self.run_checked([binary],expected)

    def test_parameters_shadowed_bindings_and_loop(self):
        def inspect(bindings):
            records=[b for b in bindings if b[0]=="compute"]
            self.assertEqual(sorted(b[4] for b in records),["count","flag","step","value","value","value"])
            flag=next(b for b in records if b[4]=="flag")
            self.assertEqual(flag[1:3],(0,0))
            values=sorted((b for b in records if b[4]=="value"),key=lambda b:b[2])
            self.assertEqual(len({b[1] for b in values}),3)
            self.assertLess(values[0][2],values[1][2])
            self.assertGreaterEqual(values[0][3],values[2][3])
            self.assertLessEqual(values[1][3],values[2][2])
            step=next(b for b in records if b[4]=="step")
            self.assertLess(step[2],step[3])
        self.produce(SOURCE,9,inspect)

    def test_non_scalar_name_remains_optional(self):
        def inspect(bindings):
            names=[b[4] for b in bindings if b[0]=="main"]
            self.assertEqual(names,["answer"])
        self.produce(ARRAY,2,inspect)

if __name__=="__main__": unittest.main()
