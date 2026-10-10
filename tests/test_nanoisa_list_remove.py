"""I qualify list removal and pop without dropping insertion coverage."""
import unittest
from tests import test_nanoisa_list_insert as insertion

class ListRemoval(insertion.ListInsert):
    def test_remove_and_pop_preserve_aliases_and_values(self):
        self.qualify("remove-int", """fn main()->int {
 let xs:List<int> = (list_int_new)
 let alias:List<int> = xs
 (list_int_push xs 7) (list_int_push xs 9) (list_int_push xs 11)
 assert (== (list_int_remove xs 1) 9)
 assert (== (list_int_length alias) 2)
 assert (== (list_int_get alias 1) 11)
 assert (== (list_int_pop xs) 11)
 assert (== (list_int_pop alias) 7)
 assert (== (list_int_length xs) 0)
 return 0
}
shadow main { assert (== (main) 0) }
""")
        self.qualify("remove-record", """struct Item { value:int, text:string }
fn main()->int {
 let xs:List<Item> = (list_Item_new)
 let alias:List<Item> = xs
 (list_Item_push xs Item { value:1099511627776, text:"first" })
 (list_Item_push xs Item { value:9, text:"middle" })
 (list_Item_push xs Item { value:11, text:"last" })
 (list_Item_remove xs 1)
 assert (== (list_Item_length alias) 2)
 let item:Item = (list_Item_pop xs)
 assert (== item.value 11) assert (== item.text "last")
 assert (== (list_Item_length alias) 1)
 let first:Item = (list_Item_get alias 0)
 assert (== first.value 1099511627776) assert (== first.text "first")
 return 0
}
shadow main { assert (== (main) 0) }
""")

    def test_removal_without_pop(self):
        self.qualify("remove-only", """struct Item { value:int, text:string }
fn main()->int {
 let xs:List<Item> = (list_Item_new)
 let alias:List<Item> = xs
 (list_Item_push xs Item { value:7, text:"first" })
 (list_Item_push xs Item { value:9, text:"middle" })
 (list_Item_push xs Item { value:11, text:"last" })
 let indices:array<int> = [1]
 (list_Item_remove xs (at indices 0))
 let last:Item = (list_Item_get alias 1)
 assert (== last.value 11) assert (== last.text "last")
 (list_Item_remove xs 0)
 assert (== (list_Item_length alias) 1)
 (list_Item_remove alias 0)
 assert (== (list_Item_length xs) 0)
 let numbers:List<int> = (list_int_new)
 (list_int_push numbers 42)
 assert (== (list_int_remove numbers 0) 42)
 assert (== (list_int_length numbers) 0)
 return 0
}
shadow main { assert (== (main) 0) }
""")

    def test_removal_bounds_trap(self):
        for name, call in (("pop-empty", "(list_int_pop xs)"),
                           ("remove-empty", "(list_int_remove xs 0)"),
                           ("remove-negative", "(list_int_remove xs -1)"),
                           ("remove-huge", "(list_int_remove xs 9223372036854775807)")):
            self.qualify(name,'fn main()->int { let xs:List<int> = (list_int_new) '+call+
                         ' return 0 } shadow main { assert true }',traps=True)

    def test_unchecked_removal_operand_refusals(self):
        for call in ('(list_int_pop xs 1)', '(list_int_remove xs true)',
                     '(list_string_pop xs)', '(list_string_remove xs 0)'):
            path=self.work/'invalid-removal.nano'
            path.write_text('fn main()->int { let xs:List<int> = (list_int_new) '+call+' return 0 }')
            p=self.command(self.driver,path,'program',success=False)
            self.assertEqual(p.returncode,1)
            self.assertEqual(p.stdout,'')

if __name__=='__main__':
    unittest.main()
