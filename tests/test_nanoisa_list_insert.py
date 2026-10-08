"""I execute generic list insertion through both producers and native translation."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ListInsert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nano-list-insert-"))
        print("I retain list insertion evidence at", cls.work, flush=True)
        cls.sequence = 0
        cls.driver = Path(os.environ.get("NANOLANG_INSERT_DRIVER", str(cls.work / "driver")))
        if "NANOLANG_INSERT_DRIVER" not in os.environ:
            source = cls.work / "driver.nano"
            source.write_text((ROOT / "tests/nanoisa/fixtures/program_closure_driver.nano.txt").read_text())
            cls.command(ROOT / "bin/nanoc_c", source, "-o", cls.driver)
        cls.cc = shlex.split(os.environ.get("CC", "clang"))

    @classmethod
    def command(cls, *args, success=True):
        cls.sequence += 1
        command = list(map(str, args))
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=900)
        (cls.work / f"command-{cls.sequence}.log").write_text(repr(command)+"\n"+p.stdout+"\n"+p.stderr)
        if success and p.returncode:
            raise AssertionError(f"{command}: {p.returncode}\n{p.stdout}\n{p.stderr}")
        return p

    def qualify(self, name, source, traps=False):
        path = self.work / (name + ".nano")
        path.write_text(source)
        for producer in ("seed", "selfhost"):
            with self.subTest(case=name, producer=producer):
                module = self.work / (name + "-" + producer + ".nvm")
                if producer == "seed":
                    self.command(ROOT / "bin/nano_virt", path, "--emit-nvm", "-o", module)
                else:
                    assembly = module.with_suffix(".nasm")
                    assembly.write_text(self.command(self.driver, path, "program").stdout)
                    self.command(ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
                self.command(ROOT / "bin/nano_vm", "--verify-only", module)
                vm = self.command(ROOT / "bin/nano_vm", module, success=not traps)
                native_c, native = module.with_suffix(".c"), module.with_suffix(".native")
                self.command(ROOT / "bin/nvm2c", module, "-o", native_c)
                self.command(*self.cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                             "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                             native_c, "-lm", "-o", native)
                result = self.command(native, success=not traps)
                if traps:
                    self.assertNotEqual(vm.returncode, 0)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("assert", vm.stderr.lower())
                    self.assertIn("native invariant", result.stderr)

    def test_positions_aliases_and_record_payloads(self):
        for element, first, second, accessor, decl in (
            ("int", "7", "9", "", ""),
            ("Item", 'Item { value: 7, text: "retained" }',
             'Item { value: 9, text: "new" }', ".value", "struct Item { value: int, text: string }\n")):
            source = decl + f"""fn main()->int {{
 let values:List<{element}> = (list_{element}_new)
 let alias:List<{element}> = values
 (list_{element}_insert values 0 {first})
 (list_{element}_insert values 1 {second})
 (list_{element}_insert values 1 {first})
 (list_{element}_insert values 0 {second})
 assert (== (list_{element}_length alias) 4)
 let a:{element} = (list_{element}_get alias 0)
 let b:{element} = (list_{element}_get alias 1)
 let c:{element} = (list_{element}_get alias 2)
 let d:{element} = (list_{element}_get alias 3)
 assert (== a{accessor} 9) assert (== b{accessor} 7)
 assert (== c{accessor} 7) assert (== d{accessor} 9)
 return 0
}}
shadow main {{ assert (== (main) 0) }}
"""
            if element == "Item":
                source = source.replace(" return 0", ' assert (== b.text "retained")\n return 0')
            self.qualify(element, source)

    def test_evaluation_order_once(self):
        self.qualify("order", """let mut trace:int = 0
fn receiver(values:List<int>)->List<int> { set trace (+ (* trace 10) 1) return values }
shadow receiver { let xs:List<int> = (list_int_new) assert (== (list_int_length (receiver xs)) 0) }
fn index()->int { set trace (+ (* trace 10) 2) return 0 }
shadow index { assert (== (index) 0) }
fn value(values:List<int>)->int { set trace (+ (* trace 10) 3) (list_int_push values 9) return 7 }
shadow value { let xs:List<int> = (list_int_new) assert (== (value xs) 7) }
fn main()->int {
 let xs:List<int> = (list_int_new)
 set trace 0
 (list_int_insert (receiver xs) (index) (value xs))
 assert (== trace 123)
 assert (== (list_int_length xs) 2)
 assert (== (list_int_get xs 0) 7)
 assert (== (list_int_get xs 1) 9)
 return 0
}
shadow main { assert (== (main) 0) }
""")

    def test_invalid_bounds_trap(self):
        for index in (-1, 1, 9223372036854775807):
            self.qualify("bound-"+str(index), f"""fn main()->int {{
 let xs:List<int> = (list_int_new)
 (list_int_insert xs {index} 7)
 return 0
}}
shadow main {{ assert true }}
""", traps=True)

    def test_unchecked_emitter_refuses_invalid_types(self):
        for call in ('(list_int_insert xs true 7)', '(list_int_insert xs 0 false)',
                     '(list_string_insert xs 0 "wrong")', '(list_int_insert xs 0)'):
            path = self.work / "invalid.nano"
            path.write_text('fn main()->int { let xs:List<int> = (list_int_new) ' + call + ' return 0 }')
            result = self.command(self.driver, path, "program", success=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")

if __name__ == "__main__":
    unittest.main()
