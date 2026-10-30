"""I prove the exact read-only eight-formal profile before copied rendering."""
from nanoisa_reconstruction import Expr, INT, BOOL, require


def prove_eight_shared(functions, layouts):
    require(len(functions) == 2 and type(functions[1].result) is int and functions[1].result == INT and
            len(functions[1].params) == 8 and all(
                isinstance(p, tuple) and len(p) == 3 and p[0] == 'borrow' and
                type(p[1]) is int and 0 <= p[1] < len(layouts) and type(p[2]) is int and p[2] == 1
                for p in functions[1].params), 'require eight exact shared nominal formals')
    work = 0

    def charge():
        nonlocal work
        work += 1
        require(work <= 4096, 'limit eight-shared proof traversal to 4096 visits')

    def scalar(node, helper=False):
        charge()
        require(isinstance(node, Expr) and type(node.tag) is int and node.tag in (INT, BOOL),
                'require exact scalar shared expressions')
        if node.kind == 'constant':
            require(not node.args and (node.tag == BOOL and type(node.value) is bool or
                    node.tag == INT and type(node.value) is int and -(1 << 63) <= node.value < (1 << 63)),
                    'require exact BOOL or signed64 literals')
        elif node.kind in ('local', 'temporary'):
            require(not node.args, 'require scalar leaf references')
        elif node.kind == 'arithmetic':
            require(node.value in ('add', 'mul') and node.tag == INT and len(node.args) == 2 and
                    all(arg.tag == INT for arg in node.args), 'require total shared ADD/MUL operands')
        else:
            require(not helper and node.kind in ('binary', 'not', 'bool_int', 'unsigned_compare'),
                    'require bounded shared caller scalar operations')
        for arg in node.args:
            scalar(arg, helper)

    values, live, operands, declared = {}, {}, {}, set()
    witness = None
    terminal = False
    calls = 0

    def literal(node):
        return node if node.kind == 'constant' else values.get(node)

    def consume(value, layout):
        require(isinstance(value, Expr) and value.tag == ('owner', layout) and value in operands,
                'require one exact unconsumed shared-caller operand')
        return operands.pop(value)

    for stmt in functions[0].body:
        charge()
        require(not terminal, 'refuse unreachable eight-shared caller statements')
        kind = stmt[0]
        if kind == 'let':
            scalar(stmt[2]); values[stmt[1]] = literal(stmt[2])
        elif kind == 'store':
            scalar(stmt[2]); values[Expr(stmt[2].tag, 'local', stmt[1])] = literal(stmt[2])
        elif kind == 'own_pack':
            _, target, layout, fields = stmt
            require(type(layout) is int and 0 <= layout < len(layouts) and
                    target.tag == ('owner', layout) and target not in operands and
                    len(fields) == len(layouts[layout]['fields']), 'require exact nominal construction')
            for value, tag in zip(fields, layouts[layout]['fields']):
                scalar(value)
                require(value.tag == tag and tag in (INT, BOOL), 'require exact scalar field tags')
            vector = tuple(literal(value) for value in fields)
            operands[target] = (layout, vector)
        elif kind == 'own_store':
            _, slot, layout, value = stmt
            require(slot not in live and slot not in declared, 'require one lexical owner declaration')
            live[slot] = consume(value, layout); declared.add(slot)
        elif kind == 'own_move':
            _, target, slot = stmt
            require(slot in live and target not in operands and target.tag == ('owner', live[slot][0]),
                    'require exact live owner transfer')
            operands[target] = live.pop(slot)
        elif kind == 'own_read':
            _, target, slot, field = stmt
            require(type(slot) is int and slot in live and type(field) is int and
                    0 <= field < len(layouts[live[slot][0]]['fields']) and
                    type(target.tag) is int and target.tag == layouts[live[slot][0]]['fields'][field],
                    'require an exact nonconsuming live-owner field observation')
            charge()
            values[target] = live[slot][1][field]
        elif kind == 'own_unpack':
            _, slot, layout, targets = stmt
            require(slot in live and live[slot][0] == layout and
                    len(targets) == len(layouts[layout]['fields']), 'require exact once-only final consumption')
            _, vector = live.pop(slot)
            for target, tag, value in zip(targets, layouts[layout]['fields'], vector):
                charge(); require(target.tag == tag, 'require exact destructured scalar identity')
                values[target] = value
        elif kind == 'multi_borrow_call':
            _, target, slots = stmt
            require(not operands and len(slots) == 8 and target.tag == INT,
                    'require complete nonescaping shared calls')
            snapshot = []
            for slot, param in zip(slots, functions[1].params):
                charge()
                require(slot in live and live[slot][0] == param[1],
                        'require every positional shared root live with exact nominal identity')
                snapshot.append(live[slot][1])
            if witness is None:
                for parameter, vector in enumerate(snapshot):
                    layout = functions[1].params[parameter][1]
                    require(len(vector) == len(layouts[layout]['fields']), 'require exact positional snapshot fields')
                    for value, tag in zip(vector, layouts[layout]['fields']):
                        charge()
                        require(value is not None and type(tag) is int and value.tag == tag,
                                'require actual typed literal first-call snapshots')
                        scalar(value, True)
                witness = tuple(snapshot)
            values[target] = None
            calls += 1
        elif kind == 'assert':
            scalar(stmt[1])
            require(witness is not None, 'require first-call snapshots before caller assertions')
        elif kind == 'return':
            scalar(stmt[1])
            require(not live and not operands, 'require final consumption of every caller owner')
            terminal = True
        else:
            require(False, 'require straight-line eight-shared caller transfers')
    require(terminal and calls and witness is not None, 'require complete actual shared calls and return')

    provenance = {}
    terminal = False
    for stmt in functions[1].body:
        charge()
        require(not terminal, 'refuse unreachable eight-shared helper statements')
        kind = stmt[0]
        if kind == 'multi_borrow_read':
            _, target, parameter, field = stmt
            require(type(parameter) is int and 0 <= parameter < 8 and type(field) is int and
                    0 <= field < len(witness[parameter]), 'require exact positional shared field reads')
            value = witness[parameter][field]
            require(value.tag == target.tag, 'require retained field type identity')
            provenance[target] = value if value.tag == BOOL else None
        elif kind == 'let':
            scalar(stmt[2], True); provenance[stmt[1]] = provenance.get(stmt[2])
        elif kind == 'store':
            require(type(stmt[1]) is int and stmt[1] >= 8, 'refuse overwriting shared formal slots')
            scalar(stmt[2], True)
            provenance[Expr(stmt[2].tag, 'local', stmt[1])] = provenance.get(stmt[2])
        elif kind == 'assert':
            scalar(stmt[1], True); value = provenance.get(stmt[1])
            require(value is not None and value.tag == BOOL and type(value.value) is bool and value.value,
                    'require a true exact first-call BOOL field assertion')
        elif kind == 'return':
            scalar(stmt[1], True)
            require(stmt[1].tag == INT, 'require exact shared INT result')
            terminal = True
        else:
            require(False, 'refuse owners, writes, calls or control in eight-shared helper')
    require(terminal, 'require complete shared helper return')
    return witness, {'work': work, 'calls': calls, 'owned_declarations': len(declared)}
