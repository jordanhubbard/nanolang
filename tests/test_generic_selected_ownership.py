"""I transfer only the concretely selected generic payload's obligations."""
import unittest
from tests import test_affine_generic_identity as generic

PREFIX = '''resource struct Handle { fd: int }
union Result<T, E> { Ok { value: T }, Err { error: E } }
union Box<T> { Some { value: T }, None {} }
fn close_handle(owner: Handle) -> int { let Handle { fd } = owner return fd }
shadow close_handle { assert (== (close_handle Handle { fd: 7 }) 7) }
'''
CONSUME = '''fn consume(result: Result<Handle, string>) -> int {
 match result {
  Ok(payload) => { let Result.Ok { value } = payload return (close_handle value) }
  Err(payload) => { let Result.Err { error } = payload assert (== error "empty") return 0 }
 }
}
shadow consume {
 let owner: Handle = Handle { fd: 7 }
 let value: Result<Handle, string> = Result.Ok { value: owner }
 assert (== (consume value) 7)
 let empty: Result<Handle, string> = Result.Err { error: "empty" }
 assert (== (consume empty) 0)
}
'''
ENDING = 'fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n'
MAIN = '''fn main() -> int {
 let owner: Handle = Handle { fd: 9 }
 let value: Result<Handle, string> = Result.Ok { value: owner }
 return (- (consume value) 9)
}
shadow main { assert (== (main) 0) }
'''

class GenericSelectedOwnership(unittest.TestCase):
    def check(self, source, accepted):
        generic.GenericAffineIdentity().check(source, accepted)

    def test_resource_and_ordinary_selected_arms(self):
        self.check(PREFIX + CONSUME + MAIN, True)

    def test_ordinary_selected_arm_can_be_ignored(self):
        source = PREFIX + CONSUME.replace('let Result.Err { error } = payload assert (== error "empty") return 0', 'return 0') + MAIN
        self.check(source, True)

    def test_empty_arm_has_no_obligation(self):
        self.check(PREFIX + '''fn consume(value: Box<Handle>) -> int { match value {
 Some(payload) => { let Box.Some { value } = payload return (close_handle value) }
 None(payload) => { let Box.None {} = payload return 0 }
} }
shadow consume { let empty: Box<Handle> = Box.None {} assert (== (consume empty) 0) let owner: Handle = Handle { fd: 7 } let value: Box<Handle> = Box.Some { value: owner } assert (== (consume value) 7) }
''' + ENDING, True)

    def test_declared_return_and_alias_transfer(self):
        source = PREFIX + CONSUME + '''fn identity(value: Result<Handle,string>) -> Result<Handle,string> { let moved: Result<Handle,string> = value return moved }
shadow identity { let owner: Handle = Handle { fd: 7 } let value: Result<Handle,string> = Result.Ok { value: owner } assert (== (consume (identity value)) 7) }
''' + MAIN.replace('(consume value)', '(consume (identity value))')
        self.check(source, True)

    def test_declared_constructor_return(self):
        self.check(PREFIX + CONSUME + """fn wrap(owner: Handle) -> Result<Handle,string> { return Result.Ok { value: owner } }
shadow wrap { assert (== (consume (wrap Handle { fd: 7 })) 7) }
fn main() -> int { return (- (consume (wrap Handle { fd: 9 })) 9) }
shadow main { assert (== (main) 0) }
""", True)

    def test_declared_constructor_argument(self):
        self.check(PREFIX + CONSUME + """fn main() -> int { let owner: Handle = Handle { fd: 9 } return (- (consume Result.Ok { value: owner }) 9) }
shadow main { assert (== (main) 0) }
""", True)

    def test_conditional_constructor_return(self):
        self.check(PREFIX + CONSUME + """fn choose(owner: Handle, keep: bool) -> Result<Handle,string> {
 if keep { return Result.Ok { value: owner } } else { (close_handle owner) return Result.Err { error: "empty" } }
}
shadow choose { assert (== (consume (choose Handle { fd: 7 } true)) 7) assert (== (consume (choose Handle { fd: 8 } false)) 0) }
fn main() -> int { return (- (consume (choose Handle { fd: 9 } true)) 9) }
shadow main { assert (== (main) 0) }
""", True)

    def test_match_constructor_return(self):
        self.check(PREFIX + CONSUME + """fn rebuild(boxed: Box<Handle>) -> Result<Handle,string> {
 return match boxed {
  Some(payload) => { let Box.Some { value } = payload Result.Ok { value: value } }
  None(payload) => Result.Err { error: "empty" }
 }
}
shadow rebuild { let owner: Handle = Handle { fd: 7 } let boxed: Box<Handle> = Box.Some { value: owner } assert (== (consume (rebuild boxed)) 7) let empty: Box<Handle> = Box.None {} assert (== (consume (rebuild empty)) 0) }
""" + ENDING, True)

    def test_nested_generic_transfer(self):
        self.check(PREFIX + CONSUME + '''fn outer(boxed: Box<Result<Handle,string>>) -> int { match boxed {
 Some(payload) => { let Box.Some { value } = payload return (consume value) }
 None(payload) => { return 0 }
} }
shadow outer { let owner: Handle = Handle { fd: 7 } let result: Result<Handle,string> = Result.Ok { value: owner } let boxed: Box<Result<Handle,string>> = Box.Some { value: result } assert (== (outer boxed) 7) }
''' + ENDING, True)

    def test_branch_join_consumes_outer_owner(self):
        self.check(PREFIX + CONSUME + '''fn joined(value: Result<Handle,string>, other: Handle) -> int { let mut total: int = 0 match value {
 Ok(payload) => { let Result.Ok { value } = payload set total (+ (close_handle value) (close_handle other)) }
 Err(payload) => { let Result.Err { error } = payload set total (close_handle other) }
} return total }
shadow joined { let empty: Result<Handle,string> = Result.Err { error: "empty" } assert (== (joined empty Handle { fd: 7 }) 7) }
''' + ENDING, True)

    def test_drop_union_rejected(self):
        self.check(PREFIX + 'fn abandon(value: Result<Handle,string>) -> void { }\n' + ENDING, False)

    def test_drop_selected_field_rejected(self):
        self.check(PREFIX + CONSUME.replace('return (close_handle value)', 'return 0') + MAIN, False)

    def test_duplicate_consume_rejected(self):
        self.check(PREFIX + CONSUME + MAIN.replace('return (- (consume value) 9)', 'let first: int = (consume value) return (- (+ first (consume value)) 18)'), False)

    def test_match_then_reuse_rejected(self):
        self.check(PREFIX + CONSUME + '''fn reuse(value: Result<Handle,string>) -> int { match value {
 Ok(payload) => { let Result.Ok { value } = payload (close_handle value) }
 Err(payload) => { let Result.Err { error } = payload assert (== error "empty") }
} return (consume value) }
''' + ENDING, False)

    def test_partial_move_rejected(self):
        self.check(PREFIX + CONSUME.replace('let Result.Ok { value } = payload return (close_handle value)', 'return (close_handle payload.value)') + MAIN, False)

    def test_incomplete_match_rejected(self):
        self.check(PREFIX + 'fn consume(value: Result<Handle,string>) -> int { match value { Ok(payload) => { let Result.Ok { value } = payload return (close_handle value) } } }\n' + ENDING, False)

    def test_join_mismatch_rejected(self):
        self.check(PREFIX + '''fn joined(value: Result<Handle,string>, other: Handle) -> void { match value {
 Ok(payload) => { let Result.Ok { value } = payload (close_handle value) (close_handle other) }
 Err(payload) => { let Result.Err { error } = payload assert (== error "empty") }
} }
''' + ENDING, False)

    def test_guarded_generic_match_remains_rejected(self):
        from tests import test_selected_variant_ownership as selected
        source = PREFIX + CONSUME.replace('Ok(payload) =>', 'Ok(payload) if true =>') + MAIN
        selected.SelectedVariantOwnership().program(source, False, {
            'nanoc_c': 'exhaustive unguarded owned match',
            'nanoc_stage1': "Parse error.*unexpected token 'if'",
            'nanoc_stage2': "Parse error.*unexpected token 'if'",
        })

    def test_resource_collection_still_rejected(self):
        self.check(PREFIX + 'fn abandon(value: Box<array<Handle>>) -> void { }\n' + ENDING, False)

    def test_unresolved_tuple_still_rejected(self):
        self.check(PREFIX + 'union Bundle<T> { Some { value: (T,int) } }\nfn abandon(value: Bundle<Handle>) -> void { }\n' + ENDING, False)

if __name__ == '__main__':
    unittest.main()
