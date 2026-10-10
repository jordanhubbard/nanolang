"""I qualify managed list release without dropping either producer route."""
from tests import test_nanoisa_list_insert as insertion

class ListFree(insertion.ListInsert):
    def test_last_owner_reclaimed_before_return(self):
        source = self.work / "free-reclamation.nano"
        source.write_text('''fn main()->int {
 (print 0)
 let xs:List<int> = (list_int_new)
 (list_int_push xs 42)
 (print 1)
 let alias:List<int> = xs
 (list_int_free xs)
 (print 2)
 (list_int_free alias)
 (print 3)
 return 0
}
shadow main { assert true }
''')
        for producer in ("seed", "selfhost"):
            with self.subTest(producer=producer):
                module = self.work / ("free-reclamation-" + producer + ".nvm")
                if producer == "seed":
                    self.command(insertion.ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", module)
                else:
                    assembly = module.with_suffix(".nasm")
                    assembly.write_text(self.command(self.driver, source, "program").stdout)
                    self.command(insertion.ROOT / "bin/nanoisa", "asm", assembly, "-o", module)
                self.command(insertion.ROOT / "bin/nano_vm", "--verify-only", module)
                result = self.command(insertion.ROOT / "obj/list_free_observer", module)
                rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
                self.assertEqual([row[0] for row in rows], [0, 1, 2, 3])
                before, allocated, aliased, released = [row[1] for row in rows]
                self.assertGreater(allocated, before)
                self.assertEqual(aliased, allocated)
                self.assertEqual(released, before)
                # I keep the final owner in the control. Collection alone
                # must not make the reclamation assertion pass.
                retained = source.with_name("retained-owner.nano")
                retained.write_text(source.read_text().replace(" (list_int_free alias)", ""))
                control = module.with_name("retained-" + producer + ".nvm")
                if producer == "seed":
                    self.command(insertion.ROOT / "bin/nano_virt", retained, "--emit-nvm", "-o", control)
                else:
                    assembly = control.with_suffix(".nasm")
                    assembly.write_text(self.command(self.driver, retained, "program").stdout)
                    self.command(insertion.ROOT / "bin/nanoisa", "asm", assembly, "-o", control)
                self.command(insertion.ROOT / "bin/nano_vm", "--verify-only", control)
                result = self.command(insertion.ROOT / "obj/list_free_observer", control)
                counts = [int(line.split()[1]) for line in result.stdout.splitlines()]
                self.assertEqual(len(counts), 4)
                self.assertGreater(counts[-1], counts[0])
                self.assertEqual(counts[-1], counts[1])

    def test_record_alias_keeps_independent_owner(self):
        self.qualify("free-record-alias", """struct Item { value:int, text:string }
fn main()->int {
 let values:List<Item> = (list_Item_new)
 (list_Item_push values Item { value:7, text:(str_concat "retained" "-child") })
 let alias:List<Item> = values
 (list_Item_free values)
 assert (== (list_Item_length alias) 1)
 let item:Item = (list_Item_get alias 0)
 assert (== item.value 7)
 assert (== item.text "retained-child")
 (list_Item_free alias)
 assert (== item.text "retained-child")
 return 0
}
shadow main { assert true }
""")

    def test_named_roots_and_repeated_allocation(self):
        self.qualify("free-roots", """fn main()->int {
 let mut total:int = 0
 for i in (range 0 3000) {
  let names:List<string> = (list_string_new)
  (list_string_push names (str_concat "allocated-" (int_to_string i)))
  set total (+ total (list_string_length names))
  (list_string_free names)
  let numbers:List<int> = (list_int_new)
  (list_int_push numbers i)
  set total (+ total (list_int_get numbers 0))
  (list_int_free numbers)
 }
 assert (== total 4501500)
 return 0
}
shadow main { assert true }
""")

    def test_temporary_receiver_evaluated_once(self):
        self.qualify("free-temporary", """let mut calls:int = 0
fn make()->List<string> {
 set calls (+ calls 1)
 let names:List<string> = (list_string_new)
 (list_string_push names (str_concat "temporary-" (int_to_string calls)))
 return names
}
fn main()->int {
 (list_string_free (make))
 assert (== calls 1)
 return 0
}
shadow main { assert true }
""")

    def test_declared_function_retains_precedence(self):
        self.qualify("free-declared", """fn list_Custom_free(value:int)->int { return (+ value 1) }
fn main()->int { assert (== (list_Custom_free 9) 10) return 0 }
shadow main { assert true }
""")

    def test_unchecked_release_refusals(self):
        for call in ('(list_string_free xs 1)', '(list_int_free xs)', '(list_string_free 7)'):
            with self.subTest(call=call):
                source=self.work/'invalid-free.nano'
                source.write_text('fn main()->int { let xs:List<string> = (list_string_new) '+call+' return 0 }')
                result=self.command(self.driver,source,'program',success=False)
                self.assertEqual(result.returncode,1)
                self.assertEqual(result.stdout,'')
