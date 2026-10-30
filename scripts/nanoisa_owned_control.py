"""I prove bounded work for the new exact exclusive-control helper profile."""
from nanoisa_reconstruction import Expr, INT, BOOL, require

CONTROL_WORK_LIMIT = 4096
# I count both operands separately. This covers every lexical operation in the
# existing guarded ADD expansion, including both branches and snapshot lets.
CONTROL_ADD_WORK = 256


def prove_exclusive_helper_control(function, helper_locals=False):
    loops = []

    def expression(node):
        require(isinstance(node, Expr) and node.tag in (INT, BOOL),
                'require exact scalar exclusive-control expressions')
        if node.kind in ('constant', 'local', 'temporary'):
            require(not node.args, 'require scalar proof leaves')
            return 1
        if node.kind == 'arithmetic':
            require(node.value == 'add' and node.tag == INT and len(node.args) == 2,
                    'require only total guarded ADD in exclusive control')
            return CONTROL_ADD_WORK + sum(expression(child) for child in node.args)
        require(node.kind in ('binary', 'not', 'bool_int', 'unsigned_compare'),
                'require total exclusive-control scalar operations')
        if node.kind == 'binary':
            require(node.value in ('==', '!=', '<', '<=', '>', '>=', 'and', 'or'),
                    'require total exclusive-control comparison or Boolean operation')
        return 1 + sum(expression(child) for child in node.args)

    def value(node, env):
        if node.kind == 'constant':
            return ('constant', node.tag, node.value)
        if node.kind in ('local', 'temporary'):
            return env.get(node)
        if node.kind == 'arithmetic' and node.value == 'add':
            return ('add', *(value(child, env) for child in node.args))
        return None

    def local(slot):
        return Expr(INT, 'local', slot)

    def block(body, incoming, inside=False, rank=None, updates=0):
        env, cost, terminal = dict(incoming), 0, False
        for position, stmt in enumerate(body):
            require(not terminal, 'refuse unsupported unreachable exclusive-control statements')
            kind = stmt[0]
            if kind == 'let':
                cost += 1 + expression(stmt[2])
                env[stmt[1]] = value(stmt[2], env)
            elif kind == 'store':
                cost += 1 + expression(stmt[2])
                result = value(stmt[2], env)
                if rank is not None and stmt[1] == rank:
                    require(updates == 0 and result == ('add', ('rank', rank, 0), ('constant', INT, 1)),
                            'require exactly one monotone rank update on continuing paths')
                    updates = 1
                    result = ('rank', rank, 1)
                env[Expr(stmt[2].tag, 'local', stmt[1])] = result
            elif kind == 'borrow_read' or helper_locals and kind in ('multi_borrow_read', 'own_read', 'path_read'):
                cost += 1
                env[stmt[1]] = None
            elif kind == 'borrow_write':
                cost += 1 + expression(stmt[2])
            elif helper_locals and kind == 'multi_borrow_write':
                cost += 1 + expression(stmt[3])
            elif helper_locals and kind == 'assert':
                cost += 1 + expression(stmt[1])
            elif helper_locals and kind in ('own_pack', 'own_store', 'own_move', 'own_unpack'):
                # Ownership and assertion provenance are proved independently.
                cost += 1
                if kind == 'own_pack':
                    for operand in stmt[3]:
                        cost += 1 if isinstance(operand.tag, tuple) else expression(operand)
                elif kind == 'own_unpack':
                    cost += len(stmt[3])
                    for target in stmt[3]:
                        if not isinstance(target.tag, tuple):
                            env[target] = None
            elif kind == 'return':
                cost += 1 + expression(stmt[1])
                terminal = True
            elif kind == 'if':
                left = block(stmt[2], env, inside, rank, updates)
                right = block(stmt[3], env, inside, rank, updates)
                cost += 1 + expression(stmt[1]) + max(left[1], right[1])
                continuing = [arm for arm in (left, right) if not arm[2]]
                terminal = not continuing
                if continuing:
                    updates = continuing[0][3]
                    require(all(arm[3] == updates for arm in continuing),
                            'require equal rank updates on continuing branch joins')
                    env = {key: item for key, item in continuing[0][0].items()
                           if all(arm[0].get(key) == item for arm in continuing)}
            elif kind in ('while', 'while_observe'):
                require(not inside and rank is None, 'refuse nested exclusive helper loops')
                condition = stmt[1]
                pretest, loop_body = ([], stmt[2]) if kind == 'while' else (stmt[2], stmt[3])
                require(all(item[0] in ('let', 'store', 'borrow_read') for item in pretest),
                        'require only scalar snapshots in exclusive helper pretests')
                before = block(pretest, env, True)
                require(not before[2], 'require continuing exclusive helper condition snapshots')
                condition_cost = 1 + before[1] + expression(condition)
                trial = block(loop_body, before[0], True)
                if trial[2]:
                    cost += condition_cost + trial[1]
                    loops.append({'form': 'return-only', 'iterations': 1,
                                  'work': condition_cost + trial[1]})
                else:
                    require(not pretest and condition.kind == 'binary' and condition.value == '<' and
                            len(condition.args) == 2 and condition.args[0].kind == 'local' and
                            condition.args[0].tag == INT and condition.args[1].kind == 'constant' and
                            condition.args[1].tag == INT,
                            'require an exact initialized literal-bound rank loop')
                    slot, bound = condition.args[0].value, condition.args[1].value
                    initial = env.get(local(slot))
                    require(initial is not None and initial[:2] == ('constant', INT) and
                            type(initial[2]) is int and type(bound) is int and
                            -(1 << 63) <= initial[2] <= bound <= (1 << 63) - 1,
                            'require dominating exact rank initialization without overflow')
                    symbolic = dict(env)
                    symbolic[local(slot)] = ('rank', slot, 0)
                    checked = block(loop_body, symbolic, True, slot)
                    require(not checked[2] and checked[3] == 1,
                            'require one rank increment on every continuing backedge')
                    trips = bound - initial[2]
                    work = condition_cost + trips * (condition_cost + checked[1])
                    cost += work
                    loops.append({'form': 'rank', 'slot': slot, 'initial': initial[2],
                                  'bound': bound, 'iterations': trips, 'work': work})
                # I retain condition initialization and conservatively invalidate
                # every loop-body scalar write, including all branch arms.
                env = dict(before[0])
                def invalidate(statements):
                    for item in statements:
                        if item[0] == 'store':
                            env.pop(Expr(item[2].tag, 'local', item[1]), None)
                        elif item[0] in ('let', 'borrow_read') or helper_locals and item[0] in ('multi_borrow_read', 'own_read', 'path_read'):
                            env.pop(item[1], None)
                        elif item[0] == 'if':
                            invalidate(item[2]); invalidate(item[3])
                if not trial[2]:
                    invalidate(loop_body)
            else:
                require(False, 'require bounded exclusive helper control statements')
            require(cost <= CONTROL_WORK_LIMIT,
                    'limit exclusive helper proof work to 4096 logical operations')
        return env, cost, terminal, updates

    result = block(function.body, {})
    require(result[2], 'require every bounded exclusive helper path to return')
    return {'work': result[1], 'loops': loops, 'add_work': CONTROL_ADD_WORK,
            'limit': CONTROL_WORK_LIMIT}
