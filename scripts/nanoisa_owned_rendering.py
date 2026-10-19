"""I plan seven bounded Nano rendering forms without changing ownership IR."""
from collections import Counter
from copy import deepcopy
from nanoisa_reconstruction import Expr, INT, BOOL, require


def plan_reassignment_nano(function):
    occurrences, stores, reads = Counter(), Counter(), Counter()
    definitions, expected, covered = {}, set(), set()
    removed_slots, patterns = set(), Counter()

    def visit(node):
        if isinstance(node, Expr):
            occurrences[node] += 1
            if node.kind == 'local':
                reads[node.value] += 1
            for child in node.args:
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    def children(stmt, path):
        if stmt[0] == 'if':
            return [(stmt[2], path + (0,)), (stmt[3], path + (1,))]
        if stmt[0] == 'while':
            return [(stmt[2], path + (0,))]
        if stmt[0] == 'while_observe':
            return [(stmt[2], path + (0,)), (stmt[3], path + (1,))]
        return []

    def index(body, scope=()):
        for i, stmt in enumerate(body):
            path = scope + (i,)
            expected.add(path)
            if stmt[0] == 'store':
                stores[stmt[1]] += 1
            targets = (stmt[1],) if stmt[0] in ('let', 'own_read', 'path_read', 'borrow_read', 'borrow_call', 'path_borrow_call') else stmt[3] if stmt[0] == 'own_unpack' else ()
            for target in targets:
                if isinstance(target, Expr) and target.tag in (INT, BOOL):
                    require(target not in definitions, 'require unique scalar rendering definitions')
                    definitions[target] = (scope, i)
            for nested, child_scope in children(stmt, path):
                index(nested, child_scope)

    def dominates(value, scope, position):
        if value.kind == 'constant':
            return True
        if value.kind != 'temporary' or value not in definitions:
            return False
        parent, before = definitions[value]
        return scope[:len(parent)] == parent and before < (scope[len(parent)] if len(scope) > len(parent) else position)

    def total(value):
        return isinstance(value, Expr) and value.tag in (INT, BOOL) and (
            value.kind in ('constant', 'local', 'temporary') and not value.args or
            value.kind == 'not' and len(value.args) == 1 and total(value.args[0]) or
            value.kind == 'binary' and value.value in ('==', '!=', '<', '<=', '>', '>=', 'and', 'or') and
            len(value.args) == 2 and all(total(child) for child in value.args))

    def mentions(value, target):
        return value == target or isinstance(value, Expr) and any(mentions(child, target) for child in value.args)

    def substitute(value, target, replacement):
        if value == target:
            return replacement
        if isinstance(value, Expr):
            return Expr(value.tag, value.kind, value.value,
                        tuple(substitute(child, target, replacement) for child in value.args))
        return value

    def cover(scope, indices):
        for i in indices:
            path = scope + (i,)
            require(path in expected and path not in covered, 'require disjoint rendering statement coverage')
            covered.add(path)

    def plan(body, scope=()):
        result, rewritten, skipped = [], {}, set()
        # I remove only the exact two-field scalar staging bundle. Pack/store
        # fusion remains a separate disjoint statement pair below.
        for i in range(len(body) - 4):
            a, b, c, d, pack = body[i:i + 5]
            if not (a[0] == b[0] == 'store' and c[0] == d[0] == 'let' and pack[0] == 'own_pack'):
                continue
            if not (len(pack[3]) == 2 and tuple(stmt[1] for stmt in (c, d)) == pack[3] and
                    c[2].kind == d[2].kind == 'local' and [c[2].value, d[2].value] == [a[1], b[1]] and a[1] != b[1]):
                continue
            if not all(store[2].tag == load[1].tag == load[2].tag and store[2].tag in (INT, BOOL) and
                       stores[store[1]] == reads[store[1]] == 1 and occurrences[load[1]] == 2 and
                       dominates(store[2], scope, i + offset)
                       for offset, (store, load) in enumerate(((a, c), (b, d)))):
                continue
            require(not skipped.intersection(range(i, i + 4)), 'require disjoint constructor bundles')
            skipped.update(range(i, i + 4))
            removed_slots.update((a[1], b[1]))
            rewritten[i + 4] = (pack[0], pack[1], pack[2], (a[2], b[2]))
            patterns['scalar_constructor_bundle'] += 1
        i = 0
        while i < len(body):
            if i in skipped:
                cover(scope, [i]); i += 1; continue
            stmt = rewritten.get(i, body[i])
            nxt = body[i + 1] if i + 1 < len(body) and i + 1 not in skipped else None
            kind = stmt[0]
            if kind in ('own_pack', 'own_move') and nxt and nxt[0] in ('own_store', 'own_assign') and nxt[3] == stmt[1] and occurrences[stmt[1]] == 2:
                if kind == 'own_pack' and nxt[0] == 'own_store':
                    result.append(('own_pack_store', nxt[1], nxt[2], stmt[3]))
                    patterns['pack_store'] += 1
                    cover(scope, [i, i + 1]); i += 2; continue
                if kind == 'own_move' and stmt[2] != nxt[1]:
                    result.append(('own_move_store', nxt[1], nxt[2], stmt[2], nxt[0] == 'own_assign'))
                    patterns['move_' + nxt[0]] += 1
                    cover(scope, [i, i + 1]); i += 2; continue
            if kind == 'own_read' and nxt and nxt[0] == 'store' and nxt[2] == stmt[1] and occurrences[stmt[1]] == 2:
                result.append(('own_read_store', nxt[1], stmt[2], stmt[3]))
                patterns['read_store'] += 1
                cover(scope, [i, i + 1]); i += 2; continue
            if kind == 'let' and stmt[2].kind == 'local' and stmt[1].tag == stmt[2].tag and occurrences[stmt[1]] == 2 and nxt and nxt[0] == 'assert' and total(nxt[1]) and mentions(nxt[1], stmt[1]):
                result.append(('assert', substitute(nxt[1], stmt[1], stmt[2])))
                patterns['snapshot_assert'] += 1
                cover(scope, [i, i + 1]); i += 2; continue
            if kind == 'own_unpack':
                retained = tuple(target.tag not in (INT, BOOL) or occurrences[target] != 1 for target in stmt[3])
                if not all(retained):
                    stmt = ('own_unpack_pruned', *stmt[1:], retained)
                    patterns['unused_unpack_copies'] += retained.count(False)
            cover(scope, [i])
            nested = children(stmt, scope + (i,))
            if nested:
                lists = [plan(child, child_scope) for child, child_scope in nested]
                stmt = (kind, stmt[1], *lists)
            result.append(stmt)
            i += 1
        return result

    visit(function.body)
    index(function.body)
    output = deepcopy(function)
    output.body = plan(function.body)
    require(covered == expected, 'require complete original rendering statement coverage')
    def surviving(node):
        if isinstance(node, Expr):
            require(node.kind != 'local' or node.value not in removed_slots,
                    'refuse surviving removed scalar rendering slots')
            for child in node.args:
                surviving(child)
        elif isinstance(node, (list, tuple)):
            if node and node[0] == 'store':
                require(node[1] not in removed_slots, 'refuse surviving removed scalar stores')
            for child in node:
                surviving(child)
    surviving(output.body)
    output.locals = {slot: tag for slot, tag in output.locals.items() if slot not in removed_slots}
    return output, {'patterns': dict(patterns), 'removed_slots': sorted(removed_slots),
                    'covered_statements': len(covered), 'original_statements': len(expected)}
