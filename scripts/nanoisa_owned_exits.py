"""I retain exact structured innermost loop exits before source rendering."""
from dataclasses import dataclass
from nanoisa_reconstruction import Expr, Function, INT, BOOL, SIMPLE, require


def needs_loop_exits(module):
    if module.get('ownership_version', 1) != 1 or len(module['functions']) != 2 or module['functions'][1].get('param_modes') != [2]:
        return False
    return has_structured_exits(module)


def has_structured_exits(module):
    for fn in module['functions']:
        code = fn['code']; positions = {ins['pc']: n for n, ins in enumerate(code)}
        groups = {}
        for n, ins in enumerate(code):
            if ins['op'] == 'JMP' and ins['target'] in positions and positions[ins['target']] <= n:
                groups.setdefault(positions[ins['target']], []).append(n)
        for header, backs in groups.items():
            branch = next((n for n in range(header, min(backs)) if code[n]['op'] in ('JMP_TRUE', 'JMP_FALSE')), None)
            if len(backs) > 1 or branch is not None and code[branch]['target'] != (code[max(backs) + 1]['pc'] if max(backs) + 1 < len(code) else fn['size']):
                return True
            if branch is not None and any(code[n]['op'] == 'JMP' and code[n]['target'] == code[branch]['target']
                                          for n in range(branch + 1, max(backs))):
                return True
    return False


@dataclass
class Edge:
    kind: str
    target: int | None
    initialized: set
    live: dict
    used: set


class ExitRegions:
    def __init__(self, analyzer):
        self.a = analyzer
        self.loops = {}
        groups = {}
        for n, ins in enumerate(analyzer.code):
            if ins['op'] == 'JMP' and analyzer.positions[ins['target']] <= n:
                groups.setdefault(analyzer.positions[ins['target']], []).append(n)
        for header, backs in groups.items():
            branch = next((n for n in range(header, min(backs)) if analyzer.code[n]['op'] in ('JMP_TRUE', 'JMP_FALSE')), None)
            require(branch is not None, 'require an exact loop-exit pretest')
            end = analyzer.positions[analyzer.code[branch]['target']]
            require(branch < min(backs) <= max(backs) < end, 'require all backedges inside the conditional loop exit')
            self.loops[header] = (branch, end)
        for header, (_, end) in self.loops.items():
            for other, (_, other_end) in self.loops.items():
                require(not (header < other < end < other_end), 'refuse crossing loop-exit intervals')
            for n, ins in enumerate(analyzer.code):
                if ins['op'] in ('JMP', 'JMP_TRUE', 'JMP_FALSE'):
                    target = analyzer.positions[ins['target']]
                    require(not (header < target < end and not header <= n < end),
                            'refuse external entries into loop-exit interiors')

    def edge(self, kind, target, initialized):
        return Edge(kind, target, set(initialized), dict(self.a.live), set(self.a.used))

    def merge(self, edges):
        require(edges, 'require a reachable structured edge merge')
        require(all(edge.live == edges[0].live for edge in edges), 'require exact live owners on loop-exit joins')
        return Edge('fallthrough', None, set.intersection(*(edge.initialized for edge in edges)),
                    dict(edges[0].live), set.union(*(edge.used for edge in edges)))

    def region(self, start, end, incoming, context=(), depth=0):
        from nanoisa_owned_reconstruction import OWNED_PRETEST, OWNED_WRAPPING_PRETEST, owned_type
        a = self.a
        require(depth <= 32, 'limit loop-exit region nesting to 32')
        initialized = set(incoming.initialized); a.live = dict(incoming.live); a.used = set(incoming.used)
        stack, body, exits = [], [], []
        i = start
        while i < end:
            if i in self.loops:
                require(not stack, 'require empty operands entering an exit-aware loop')
                branch, after = self.loops[i]
                require(after <= end, 'require a loop-exit interval inside its parent region')
                condition_stack, observations = [], []
                materialized = any(a.code[n]['op'] in OWNED_WRAPPING_PRETEST or a.code[n]['op'] in ('REGION_BEGIN', 'REF_GET') for n in range(i, branch))
                n = i
                while n < branch:
                    ins = a.code[n]
                    if ins['op'] == 'REGION_BEGIN':
                        a.touch(n)
                        require(all(x['op'] != 'CALL_REF' for x in a.code[n:n + a.reference_region_length(n)]),
                                'retain loop-exit pretests without reference calls')
                        a.observation(n, branch, condition_stack, observations, allow_call=False)
                        n += a.reference_region_length(n); continue
                    if a.helper and ins['op'] == 'REF_GET':
                        a.touch(n); a.helper_read(ins, condition_stack, observations); n += 1; continue
                    require(ins['op'] in OWNED_PRETEST | OWNED_WRAPPING_PRETEST, 'require bounded loop-exit pretest operations')
                    if ins['op'] == 'LOAD_LOCAL':require(ins['arg'] in a.types, 'require scalar loop-exit pretest locals')
                    a.simple(n, condition_stack, initialized, observations, pure=not materialized)
                    n += 1
                condition = a.pop(condition_stack, BOOL)
                require(not condition_stack, 'require exactly one loop-exit condition')
                if a.code[branch]['op'] == 'JMP_TRUE':condition = Expr(BOOL, 'not', None, (condition,))
                a.touch(branch)
                entry = self.edge('fallthrough', None, initialized)
                nested, outcomes = self.region(branch + 1, after, entry, context + ((i, after),), depth + 1)
                false_and_break = [entry]
                for edge in outcomes:
                    if edge.kind == 'return':exits.append(edge)
                    elif edge.kind == 'break':
                        require(edge.target == i and edge.live == entry.live, 'require innermost exact-owner break exits')
                        false_and_break.append(edge)
                    else:
                        require(edge.kind == 'continue' and edge.target == i and edge.live == entry.live,
                                'require innermost exact-owner backedges')
                merged = self.merge(false_and_break)
                a.live, a.used, initialized = merged.live, set.union(entry.used, *(edge.used for edge in outcomes)), merged.initialized
                body.append(('while_observe', condition, observations, nested) if materialized else ('while', condition, nested))
                i = after; continue
            ins = a.code[i]; op, arg, pc = ins['op'], ins['arg'], ins['pc']
            if op in ('JMP_TRUE', 'JMP_FALSE'):
                a.touch(i); condition = a.pop(stack, BOOL)
                require(not stack, 'require empty branch operands in loop-exit regions')
                target = a.positions[ins['target']]
                require(i < target <= end, 'require forward loop-exit branch boundaries')
                if op == 'JMP_TRUE':condition = Expr(BOOL, 'not', None, (condition,))
                separator = target - 1
                transfer = context and a.code[separator]['op'] == 'JMP' and a.positions[a.code[separator]['target']] in context[-1]
                has_else = separator > i and a.code[separator]['op'] == 'JMP' and not transfer
                join = a.positions[a.code[separator]['target']] if has_else else target
                require(target <= join <= end, 'require noncrossing loop-exit branch joins')
                entry = self.edge('fallthrough', None, initialized)
                left, le = self.region(i + 1, separator if has_else else target, entry, context, depth + 1)
                if has_else:
                    a.touch(separator)
                    right, re = self.region(target, join, entry, context, depth + 1)
                else:right, re = [], [entry]
                body.append(('if', condition, left, right))
                continuing = [edge for edge in le + re if edge.kind == 'fallthrough']
                exits.extend(edge for edge in le + re if edge.kind != 'fallthrough')
                if not continuing:
                    require(join == end, 'refuse unaccounted instructions after abrupt branch exits')
                    return body, exits
                merged = self.merge(continuing)
                require(all(entry.live.get(slot) == layout for slot, layout in merged.live.items()),
                        'refuse lexical owner escape at loop-exit branches')
                a.live, a.used, initialized = merged.live, set.union(*(edge.used for edge in le + re)), merged.initialized
                i = join; continue
            if op == 'JMP':
                a.touch(i); target = a.positions[ins['target']]
                require(not stack and i == end - 1, 'require complete abrupt-transfer regions without operands')
                if context and target in context[-1]:
                    header, after = context[-1]
                    kind = 'continue' if target == header else 'break'
                    body.append((kind,)); exits.append(self.edge(kind, header, initialized))
                else:
                    require(target == end, 'refuse unstructured or outer-loop transfers')
                    exits.append(self.edge('fallthrough', None, initialized))
                return body, exits
            if op == 'RET':
                a.touch(i)
                require(i == end - 1 and not a.live, 'require complete cleanup before loop-exit return')
                result = a.pop(stack, a.result)
                require(not stack, 'require one exact scalar loop-exit return')
                body.append(('return', result)); exits.append(self.edge('return', None, initialized))
                return body, exits
            # I reuse the original checked linear instruction routine, including
            # exact owner stack suffix isolation and complete reference regions.
            i = a.exit_step(i, end, stack, initialized, body)
        require(not stack, 'require empty operands at loop-exit region boundaries')
        exits.append(self.edge('fallthrough', None, initialized))
        return body, exits

    def run(self):
        a = self.a
        body, edges = self.region(0, len(a.code), self.edge('fallthrough', None, set(range(len(a.params)))))
        require(edges and all(edge.kind == 'return' for edge in edges), 'require all exit-aware function paths to return')
        require(len(a.seen) == len(a.code), 'require complete loop-exit instruction coverage')
        consuming = getattr(a, 'consuming_exits', False)
        parameters = [('own_parameter', i, t[1]) for i, t in enumerate(a.params)
                      if consuming and isinstance(t, tuple) and t[0] == 'owner']
        function = Function(a.fn['name'], a.params, a.result, a.types, a.fn['names'], parameters + body)
        function.loop_exits = not consuming
        function.consuming_exits = consuming
        return function


def prove_exit_origins(function, consuming_functions=None, layouts=None):
    """I prove original entry origins on all structured outcomes without editing IR."""
    sites, serial = set(), [0]
    def owned(value):return isinstance(value, Expr) and isinstance(value.tag, tuple) and value.tag[0] == 'owner'
    def created(origin, marker):return marker in origin[2] or any(child is not None and created(child, marker) for child in origin[4])
    def block(body, incoming, declarations, context=()):
        live, declared, operands, abrupt = dict(incoming), dict(declarations), {}, []
        def consume(value, layout):
            require(owned(value) and value.tag == ('owner', layout) and value in operands, 'require one exact loop-exit owner operand')
            return operands.pop(value)
        def produce(value, origin):
            require(value not in operands and value.tag == ('owner', origin[3]), 'require exact unique loop-exit owner outputs')
            operands[value] = origin
        for position, stmt in enumerate(body):
            kind = stmt[0]
            if kind == 'own_parameter':
                _, slot, layout = stmt
                require(consuming_functions is not None and layouts is not None and
                        type(slot) is int and 0 <= slot < len(function.params) and
                        function.params[slot] == ('owner', layout) and slot not in live and slot not in declared and
                        not context, 'require exact consuming parameter origins')
                declared[slot] = layout
                live[slot] = ('parameter', slot, (), layout, tuple(None for _ in layouts[layout]['fields']))
            elif kind == 'let' and isinstance(stmt[2], Expr) and stmt[2].kind == 'call':
                call = stmt[2]
                require(consuming_functions is not None and type(call.value) is int and
                        0 < call.value < len(consuming_functions), 'require an exact consuming callee')
                callee = consuming_functions[call.value]
                require(call.tag == INT and stmt[1].tag == INT and callee.result == INT and
                        len(call.args) == len(callee.params), 'require exact consuming call signature')
                for value, tag in zip(call.args, callee.params):
                    require(isinstance(value, Expr) and value.tag == tag, 'require typed consuming call arguments')
                    if owned(value):consume(value, tag[1])
                    else:
                        require(tag in (INT, BOOL) and value.kind in ('constant', 'local', 'temporary') and not value.args,
                                'require materialized scalar consuming arguments')
            elif kind == 'own_pack':
                _, target, layout, values = stmt
                require(target.value not in sites, 'require unique loop-exit pack sites');sites.add(target.value)
                children = tuple(consume(v, v.tag[1]) if owned(v) else None for v in values)
                produce(target, ('pack', target.value, context, layout, children))
            elif kind == 'own_store':
                _, slot, layout, value = stmt
                require(slot not in live and slot not in declared, 'require a fresh lexical loop-exit owner declaration')
                declared[slot] = layout;live[slot] = consume(value, layout)
            elif kind == 'own_move':
                _, target, slot = stmt
                require(slot in live, 'require a live loop-exit moved origin');produce(target, live.pop(slot))
            elif kind == 'own_unpack':
                _, slot, layout, values = stmt
                require(slot in live and live[slot][3] == layout, 'require exact loop-exit destructive origin')
                origin = live.pop(slot)
                require(len(values) == len(origin[4]), 'require complete loop-exit unpack positions')
                for target, child in zip(values, origin[4]):
                    require(owned(target) == (child is not None), 'require exact loop-exit child positions')
                    if child is not None:produce(target, child)
            elif kind == 'if':
                require(not operands, 'require empty branch owner operands')
                outcomes = block(stmt[2], live, declared, context) + block(stmt[3], live, declared, context)
                continuing = [state for name, state in outcomes if name == 'fallthrough']
                abrupt.extend((name, state) for name, state in outcomes if name != 'fallthrough')
                if not continuing:
                    require(position == len(body) - 1, 'refuse unreachable origin statements after abrupt branches')
                    return abrupt
                require(all(state == continuing[0] for state in continuing) and
                        all(live.get(slot) == origin for slot, origin in continuing[0].items()),
                        'require exact surviving branch origins')
                live = continuing[0]
            elif kind in ('while', 'while_observe'):
                require(not operands, 'require empty loop-entry owner operands')
                serial[0] += 1;marker = serial[0]
                if kind == 'while_observe':
                    pre = block(stmt[2], live, declared, context)
                    require(pre == [('fallthrough', live)], 'require ownership-preserving loop-exit snapshots')
                nested = stmt[2] if kind == 'while' else stmt[3]
                outcomes = block(nested, live, declared, context + (marker,))
                for name, state in outcomes:
                    if name == 'return':abrupt.append((name, state));continue
                    require(name in ('break', 'continue') and state == live and
                            all(not created(origin, marker) for origin in state.values()),
                            'require exact incoming origins at every loop break and backedge')
                # Only incoming declarations and the zero/final-false state survive.
            elif kind in ('return', 'break', 'continue'):
                require(not operands and position == len(body) - 1, 'require complete abrupt origin boundaries')
                if kind == 'return':require(not live and not owned(stmt[1]), 'require complete return owner cleanup')
                else:require(context, 'require an enclosing origin loop')
                return abrupt + [(kind, dict(live))]
            elif kind in ('own_read', 'borrow_call'):
                require(stmt[2] in live and not owned(stmt[1]), 'require a live nonescaping loop-exit reference root')
            else:
                require(kind in ('let', 'store', 'assert'), 'retain bounded loop-exit entry ownership operations')
                def scalar(node):return not isinstance(node, Expr) or not owned(node) and (consuming_functions is None or node.kind != 'call') and all(scalar(child) for child in node.args)
                require(all(scalar(node) for node in stmt[1:]), 'refuse owners in loop-exit scalar operands')
        require(not operands, 'require consumed origin operands at lexical exit')
        return abrupt + [('fallthrough', live)]
    require(all(name == 'return' for name, _ in block(function.body, {}, {})), 'require only successful origin returns')


def prove_exit_helper(function):
    """I bound helper work while distinguishing loop exits from function return."""
    from nanoisa_owned_control import CONTROL_ADD_WORK, CONTROL_WORK_LIMIT
    loops = []
    def expression(node):
        require(isinstance(node, Expr) and node.tag in (INT, BOOL), 'require exact scalar loop-exit helper values')
        if node.kind in ('constant', 'local', 'temporary'):
            require(not node.args, 'require scalar helper leaves');return 1
        if node.kind == 'arithmetic':
            require(node.value == 'add' and node.tag == INT and len(node.args) == 2, 'retain ADD-only loop-exit helpers')
            return CONTROL_ADD_WORK + sum(expression(child) for child in node.args)
        require(node.kind in ('binary', 'not', 'bool_int', 'unsigned_compare') and
                (node.kind != 'binary' or node.value in ('==', '!=', '<', '<=', '>', '>=', 'and', 'or')),
                'require total loop-exit helper expressions')
        return 1 + sum(expression(child) for child in node.args)
    def value(node, env):
        if node.kind == 'constant':return ('constant', node.tag, node.value)
        if node.kind in ('local', 'temporary'):return env.get(node)
        if node.kind == 'arithmetic' and node.value == 'add':return ('add', *(value(child, env) for child in node.args))
        return None
    def merge(states):
        result = []
        for kind in ('fallthrough', 'break', 'continue', 'return'):
            group = [state for state in states if state[0] == kind]
            if group:
                updates = group[0][3]
                if kind in ('fallthrough', 'continue'):
                    if any(state[3] != updates for state in group):updates = -1
                env = {key: val for key, val in group[0][1].items() if all(state[1].get(key) == val for state in group)}
                result.append((kind, env, max(state[2] for state in group), updates))
        return result
    def block(body, incoming, inside=False, rank=None, initial_updates=0):
        env, cost, updates, abrupt = dict(incoming), 0, initial_updates, []
        for position, stmt in enumerate(body):
            kind = stmt[0]
            if kind == 'let':cost += 1 + expression(stmt[2]);env[stmt[1]] = value(stmt[2], env)
            elif kind == 'store':
                cost += 1 + expression(stmt[2]);result = value(stmt[2], env)
                if rank is not None and stmt[1] == rank:
                    if updates == 0 and result == ('add', ('rank', rank, 0), ('constant', INT, 1)):
                        updates = 1;result = ('rank', rank, 1)
                    else:updates = -1
                env[Expr(stmt[2].tag, 'local', stmt[1])] = result
            elif kind == 'borrow_read':cost += 1;env[stmt[1]] = None
            elif kind == 'borrow_write':cost += 1 + expression(stmt[2])
            elif kind in ('return', 'break', 'continue'):
                require(position == len(body) - 1 and (inside or kind == 'return'), 'require bounded helper abrupt edges')
                cost += 1 + (expression(stmt[1]) if kind == 'return' else 0)
                require(cost <= CONTROL_WORK_LIMIT, 'limit loop-exit helper proof work to 4096 logical operations')
                return merge(abrupt + [(kind, env, cost, updates)])
            elif kind == 'if':
                prefix = cost + 1 + expression(stmt[1])
                outcomes = block(stmt[2], env, inside, rank, updates) + block(stmt[3], env, inside, rank, updates)
                outcomes = merge([(name, state, prefix + work, changed) for name, state, work, changed in outcomes])
                abrupt += [state for state in outcomes if state[0] != 'fallthrough']
                ongoing = [state for state in outcomes if state[0] == 'fallthrough']
                if not ongoing:
                    require(position == len(body) - 1, 'refuse unreachable helper control after abrupt branches')
                    require(max(state[2] for state in abrupt) <= CONTROL_WORK_LIMIT, 'limit loop-exit helper proof work to 4096 logical operations')
                    return merge(abrupt)
                _, env, cost, updates = ongoing[0]
            elif kind in ('while', 'while_observe'):
                require(not inside and rank is None, 'refuse nested loop-exit helper loops')
                pretest, nested = ([], stmt[2]) if kind == 'while' else (stmt[2], stmt[3])
                require(all(x[0] in ('let', 'store', 'borrow_read') for x in pretest), 'require scalar helper pretest snapshots')
                before = block(pretest, env, True)
                require(len(before) == 1 and before[0][0] == 'fallthrough', 'require continuing helper pretest snapshots')
                before_env, pre_cost = before[0][1], before[0][2]
                cond_cost = pre_cost + 1 + expression(stmt[1])
                trial = block(nested, before_env, True)
                continued = [x for x in trial if x[0] in ('continue', 'fallthrough')]
                if continued:
                    condition = stmt[1]
                    require(not pretest and condition.kind == 'binary' and condition.value == '<' and
                            len(condition.args) == 2 and condition.args[0].kind == 'local' and condition.args[0].tag == INT,
                            'require exact initialized strict helper rank pretests')
                    slot = condition.args[0].value;initial = env.get(Expr(INT, 'local', slot));bound = value(condition.args[1], env)
                    require(initial is not None and bound is not None and initial[:2] == bound[:2] == ('constant', INT) and
                            type(initial[2]) is int and type(bound[2]) is int and -(1 << 63) <= initial[2] <= bound[2] <= (1 << 63) - 1,
                            'require dominating exact rank and bound without overflow')
                    symbolic = dict(env);symbolic[Expr(INT, 'local', slot)] = ('rank', slot, 0)
                    checked = block(nested, symbolic, True, slot)
                    backedges = [x for x in checked if x[0] in ('continue', 'fallthrough')]
                    require(backedges and all(x[3] == 1 and value(condition.args[1], x[1]) == bound for x in backedges),
                            'require one rank increment and unchanged bound on every continuing edge')
                    trips = bound[2] - initial[2]
                    work = cond_cost + trips * (cond_cost + max(x[2] for x in checked))
                    loops.append({'form': 'rank-exits', 'slot': slot, 'initial': initial[2], 'bound': bound[2], 'iterations': trips, 'work': work})
                else:
                    require(all(x[0] in ('break', 'return') for x in trial), 'require finite abrupt helper loop outcomes')
                    work = cond_cost + max(x[2] for x in trial)
                    loops.append({'form': 'break-return-only', 'iterations': 1, 'work': work})
                cost += work
                # I count after-loop work even when a prior iteration may return.
                # This conservatively overcounts, but never drops break-path writes.
                env = dict(before_env)
                def invalidate(statements):
                    for item in statements:
                        if item[0] == 'store':env.pop(Expr(item[2].tag, 'local', item[1]), None)
                        elif item[0] in ('let', 'borrow_read'):env.pop(item[1], None)
                        elif item[0] == 'if':invalidate(item[2]);invalidate(item[3])
                invalidate(nested)
            else:require(False, 'require bounded loop-exit helper statements')
            require(cost <= CONTROL_WORK_LIMIT and all(x[2] <= CONTROL_WORK_LIMIT for x in abrupt),
                    'limit loop-exit helper proof work to 4096 logical operations')
        return merge(abrupt + [('fallthrough', env, cost, updates)])
    states = block(function.body, {})
    require(states and all(x[0] == 'return' for x in states), 'require complete loop-exit helper return coverage')
    return {'work': max(x[2] for x in states), 'loops': loops, 'add_work': CONTROL_ADD_WORK, 'limit': CONTROL_WORK_LIMIT}
