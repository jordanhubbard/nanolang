"""I prove exact owner restoration separately from lexical declarations."""
from nanoisa_reconstruction import Expr, require


def prove_owner_reassignment(function, helper_locals=False):
    """I validate the new entry-only reassignment IR without changing it."""
    reassigned = set()
    serial = [0]
    pack_sites = set()

    def owned(expression):
        return isinstance(expression, Expr) and isinstance(expression.tag, tuple) and expression.tag[0] == 'owner'

    def contains(origin, marker):
        return marker in origin[2] or any(child is not None and contains(child, marker) for child in origin[4])

    def block(body, incoming, declarations, context=()):
        live, declared, operands = dict(incoming), dict(declarations), {}
        terminal = False

        def consume(value, layout):
            require(owned(value) and value.tag == ('owner', layout) and value in operands,
                    'require one exact unconsumed owner-origin operand')
            origin = operands.pop(value)
            require(origin[3] == layout, 'require exact origin nominal identity')
            return origin

        def produce(value, origin):
            require(value not in operands, 'refuse duplicate owner-origin operands')
            operands[value] = origin

        for stmt in body:
            require(not terminal, 'refuse unreachable owner-origin statements')
            kind = stmt[0]
            if kind == 'own_pack':
                _, target, layout, values = stmt
                require(target.value not in pack_sites, 'require unique syntactic pack-origin sites')
                pack_sites.add(target.value)
                children = tuple(consume(value, value.tag[1]) if owned(value) else None for value in values)
                produce(target, ('pack', target.value, context, layout, children))
            elif kind in ('own_store', 'own_assign'):
                _, slot, layout, value = stmt
                require(slot not in live, 'require a consumed owner-origin destination')
                if kind == 'own_store':
                    require(slot not in declared, 'require one lexical owner declaration')
                    declared[slot] = layout
                else:
                    require(declared.get(slot) == layout, 'require a dominating exact owner declaration')
                    reassigned.add(slot)
                live[slot] = consume(value, layout)
            elif kind == 'own_move':
                _, target, slot = stmt
                require(slot in live and target.tag == ('owner', live[slot][3]),
                        'require the exact live moved owner origin')
                produce(target, live.pop(slot))
            elif kind == 'own_unpack':
                _, slot, layout, values = stmt
                require(slot in live and live[slot][3] == layout, 'require a live nominal unpack origin')
                origin = live.pop(slot)
                require(len(values) == len(origin[4]), 'require complete origin child extraction')
                for target, child in zip(values, origin[4]):
                    require(owned(target) == (child is not None), 'require exact scalar and child origin positions')
                    if child is not None:
                        require(target.tag == ('owner', child[3]), 'require exact extracted child origin')
                        produce(target, child)
            elif kind == 'if':
                require(not operands, 'require empty owner operands at origin branches')
                left = block(stmt[2], live, declared, context)
                right = block(stmt[3], live, declared, context)
                continuing = [arm for arm in (left, right) if not arm[1]]
                terminal = not continuing
                if continuing:
                    require(all(arm[0] == continuing[0][0] for arm in continuing),
                            'require exact continuing owner-origin agreement')
                    require(all(live.get(slot) == origin for slot, origin in continuing[0][0].items()),
                            'require each surviving incoming branch owner origin')
                    live = continuing[0][0]
                else:
                    live = {}
                # Child declarations never become available in this lexical list.
            elif kind in ('while', 'while_observe'):
                require(not operands, 'require empty owner operands at origin loop entry')
                serial[0] += 1
                marker = serial[0]
                require(all(not contains(origin, marker) for origin in live.values()),
                        'refuse a loop-body pack origin as an incoming owner')
                if kind == 'while_observe':
                    pretest = block(stmt[2], live, declared, context)
                    require(not pretest[1] and pretest[0] == live,
                            'require ownership-preserving origin pretests')
                loop_body = stmt[2] if kind == 'while' else stmt[3]
                result = block(loop_body, live, declared, context + (marker,))
                if not result[1]:
                    require(all(not contains(origin, marker) for origin in result[0].values()),
                            'refuse loop-body owner origins escaping a backedge')
                    require(result[0] == live, 'require exact incoming owner-origin restoration')
                # I keep the incoming lifetime/declaration state on the false edge.
            elif kind in ('borrow_call', 'path_borrow_call', 'own_read', 'path_read'):
                require(stmt[2] in live, 'require a live reference root origin')
                require(not owned(stmt[1]), 'refuse an escaping reference-origin result')
            elif helper_locals and kind == 'multi_borrow_call':
                require(not operands and len(stmt[2]) == 2 and len(set(stmt[2])) == 2 and
                        all(slot in live for slot in stmt[2]) and not owned(stmt[1]),
                        'require two distinct live caller origins without escaping owners')
            elif kind == 'return':
                require(not live and not operands and not owned(stmt[1]),
                        'require complete successful owner-origin consumption')
                terminal = True
            else:
                require(kind in ('let', 'store', 'assert') or helper_locals and kind in ('multi_borrow_read', 'multi_borrow_write'), 'require bounded entry owner-origin operations')
                def scalar(value):
                    if isinstance(value, Expr):
                        return not owned(value) and all(scalar(arg) for arg in value.args)
                    return True
                require(all(scalar(value) for value in stmt[1:]), 'refuse owner operands in scalar origin operations')
        require(not operands, 'require consumed owner-origin operands at region exit')
        return live, terminal

    result = block(function.body, {}, {})
    require(result[1], 'require a terminal reassignment entry')
    return frozenset(reassigned)
