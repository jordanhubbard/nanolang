"""I exercise defensive witness proofs separately from executable admission."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nanoisa_reconstruction import Expr, INT, BOOL, Refusal
from nanoisa_owned_helper_locals import WitnessProof


def temporary(tag, name):
    return Expr(tag, 'temporary', name)


def integer(value):
    return Expr(INT, 'constant', value)


def equal(left, right):
    return Expr(BOOL, 'binary', '==', (left, right))


class HelperLocalWitness(unittest.TestCase):
    def test_literal_and_token_equality_only(self):
        proof = WitnessProof()
        a, b = temporary(INT, 'a'), temporary(INT, 'b')
        env = proof.mapping()
        env[a] = proof.token(INT)
        env[b] = env[a]
        self.assertEqual(proof.value(equal(a, b), env), ('literal', BOOL, True))
        env[b] = proof.token(INT)
        self.assertNotEqual(proof.value(equal(a, b), env), ('literal', BOOL, True))
        expr = Expr(INT, 'arithmetic', 'add', (integer(1), integer(1)))
        self.assertNotEqual(proof.value(expr, env), proof.value(expr, env))

    def test_exact_witness_literal_types_and_bounds(self):
        for expression in (integer(True), integer(1 << 63), integer(-(1 << 63)-1),
                           Expr(BOOL, 'constant', 1)):
            with self.subTest(expression=expression), self.assertRaises(Refusal):
                WitnessProof().literal(expression)
        for value in (-(1 << 63), (1 << 63)-1):
            self.assertEqual(WitnessProof().literal(integer(value)), ('literal', INT, value))

    def test_same_iteration_owner_capture(self):
        proof = WitnessProof()
        index = Expr(INT, 'local', 3)
        captured, read = temporary(INT, 'captured'), temporary(INT, 'read')
        owner = temporary(('owner', 0), 'owner')
        token = proof.token(INT)
        env = proof.mapping(((index, token),))
        body = [('let', captured, index), ('own_pack', owner, 0, (captured,)),
                ('own_store', 4, 0, owner), ('own_read', read, 4, 0),
                ('assert', equal(read, index)), ('own_unpack', 4, 0, (captured,)),
                ('return', integer(0))]
        self.assertTrue(proof.block(body, (env, {}, {}), {})[2])
        changed = body[:4] + [('store', 3, Expr(INT, 'arithmetic', 'add', (index, integer(1))))] + body[4:]
        with self.assertRaisesRegex(Refusal, 'assertion witness'):
            WitnessProof().block(changed, (env, {}, {}), {})

    def test_actual_shared_literal_required(self):
        read = temporary(INT, 'shared')
        body = [('multi_borrow_read', read, 1, 0), ('assert', equal(read, integer(4))),
                ('return', integer(0))]
        self.assertTrue(WitnessProof().block(body, ({}, {}, {}), {(1, 0): ('literal', INT, 4)})[2])
        for value in (None, ('literal', INT, 0), ('token', INT, 1)):
            with self.subTest(value=value), self.assertRaisesRegex(Refusal, 'assertion witness'):
                WitnessProof().block(body, ({}, {}, {}), {(1, 0): value})

    def test_loop_preinvalidates_every_nonrank_write(self):
        rank, other = Expr(INT, 'local', 3), Expr(INT, 'local', 4)
        observed = temporary(INT, 'observed')
        condition = Expr(BOOL, 'binary', '<', (rank, integer(2)))
        step = ('store', 3, Expr(INT, 'arithmetic', 'add', (rank, integer(1))))
        cases = [
            [('assert', equal(other, integer(0))), ('store', 4, integer(1)), step],
            [('multi_borrow_read', observed, 0, 0), ('assert', equal(observed, integer(0))),
             ('multi_borrow_write', 0, 0, integer(1)), step],
            [('multi_borrow_read', observed, 0, 0), ('assert', equal(observed, integer(0))),
             ('if', Expr(BOOL, 'constant', True), [('multi_borrow_write', 0, 0, integer(1))], []), step],
            [('assert', equal(other, integer(0))),
             ('if', Expr(BOOL, 'constant', True), [('store', 4, integer(1))], []), step],
        ]
        for body in cases:
            with self.subTest(body=body), self.assertRaisesRegex(Refusal, 'assertion witness'):
                WitnessProof().block([('while', condition, body), ('return', integer(0))],
                                     ({rank: ('literal', INT, 0), other: ('literal', INT, 0)}, {}, {}),
                                     {(0, 0): ('literal', INT, 0)})

    def test_immutable_preloop_snapshot_survives_invalidation(self):
        rank = Expr(INT, 'local', 3)
        before = temporary(INT, 'before')
        condition = Expr(BOOL, 'binary', '<', (rank, integer(2)))
        body = [('multi_borrow_read', before, 0, 0),
                ('while', condition, [
                    ('assert', equal(before, integer(4))),
                    ('multi_borrow_write', 0, 0, integer(5)),
                    ('store', 3, Expr(INT, 'arithmetic', 'add', (rank, integer(1))))]),
                ('return', integer(0))]
        self.assertTrue(WitnessProof().block(body, ({rank: ('literal', INT, 0)}, {}, {}),
                                            {(0, 0): ('literal', INT, 4)})[2])

    def test_helper_backedge_cannot_replace_same_layout_origin(self):
        from nanoisa_owned_origins import prove_owner_reassignment
        from nanoisa_reconstruction import Function
        first, second = temporary(('owner', 0), 'first'), temporary(('owner', 0), 'second')
        value = temporary(INT, 'field')
        original = [('own_pack', first, 0, (integer(4),)), ('own_store', 2, 0, first)]
        replacement = [('own_unpack', 2, 0, (value,)),
                       ('own_pack', second, 0, (integer(4),)), ('own_store', 2, 0, second)]
        body = original + [('while', Expr(BOOL, 'constant', True), replacement),
                           ('own_unpack', 2, 0, (value,)), ('return', integer(0))]
        fn = Function('helper', [('borrow', 0, 2), ('borrow', 0, 1)], INT, {}, [], body)
        with self.assertRaises(Refusal):
            prove_owner_reassignment(fn, helper_locals=True)
        # Borrowed formals never become owned merely because their layout matches.
        fn.body = [('own_move', first, 0), ('own_store', 2, 0, first),
                   ('own_unpack', 2, 0, (value,)), ('return', integer(0))]
        with self.assertRaises(Refusal):
            prove_owner_reassignment(fn, helper_locals=True)

    def test_field_generation_and_nominal_identity(self):
        proof = WitnessProof()
        first = proof.pack(0, [('literal', INT, 4)], ('helper', 0))
        second = proof.pack(0, [('literal', INT, 5)], ('helper', 0))
        self.assertNotEqual(first, second)
        self.assertNotEqual(proof.field(first, 0), proof.field(second, 0))
        with self.assertRaisesRegex(Refusal, 'nominal unpack'):
            proof.block([('own_unpack', 2, 1, (temporary(INT, 'field'),))], ({}, {2: first}, {}), {})

    def test_allocation_and_traversal_caps_precede_mutation(self):
        proof = WitnessProof()
        facts = proof.mapping()
        proof.allocated = 4096
        with self.assertRaisesRegex(Refusal, 'bindings'):
            facts['new'] = ('literal', INT, 0)
        self.assertNotIn('new', facts)
        proof.work = 65536
        with self.assertRaisesRegex(Refusal, 'traversal'):
            proof.token(INT)
        self.assertEqual(proof.tokens, 0)
        proof = WitnessProof()
        proof.tokens = 4096
        with self.assertRaisesRegex(Refusal, 'tokens'):
            proof.token(INT)

    def test_actual_binding_boundary_without_preseeded_counters(self):
        proof = WitnessProof()
        facts = proof.mapping((index, ('literal', INT, index)) for index in range(4096))
        self.assertEqual(len(facts), 4096)
        self.assertEqual(proof.allocated, 4096)
        with self.assertRaisesRegex(Refusal, 'bindings'):
            facts[4096] = ('literal', INT, 4096)
        self.assertEqual(len(facts), 4096)

    def test_merge_does_not_equate_unknown_or_distinct_facts(self):
        proof = WitnessProof()
        a = proof.token(INT)
        b = proof.token(INT)
        result = proof.intersect([{'x': a, 'same': a}, {'x': b, 'same': a}])
        self.assertNotIn('x', result)
        self.assertEqual(result['same'], a)


if __name__ == '__main__':
    unittest.main()
