"""I prove finite scalar witnesses without executing borrowed helper programs."""
from nanoisa_reconstruction import Expr, INT, BOOL, require

TOKEN_LIMIT = 4096
BINDING_LIMIT = 4096
TRAVERSAL_LIMIT = 65536
DEPTH_LIMIT = 32


class Facts(dict):
    def __init__(self, proof):
        super().__init__()
        self.proof = proof

    def __setitem__(self, key, value):
        self.proof.charge()
        if key not in self:
            self.proof.reserve(1)
        super().__setitem__(key, value)


class WitnessProof:
    def __init__(self):
        self.work = 0
        self.tokens = 0
        self.saved = []
        self.nodes = {}
        self.serial = 0
        self.allocated = 0

    def charge(self, n=1):
        require(n >= 0 and self.work + n <= TRAVERSAL_LIMIT,
                'limit helper witness traversal to 65536 operations')
        self.work += n

    def token(self, tag):
        self.charge()
        require(self.tokens < TOKEN_LIMIT, 'limit helper witness tokens to 4096')
        self.tokens += 1
        return ('token', tag, self.tokens)

    def reserve(self, n):
        # Cumulative allocation is deliberately stricter than peak live storage.
        require(n >= 0 and self.allocated + n <= BINDING_LIMIT,
                'limit retained helper witness bindings to 4096')
        self.allocated += n

    def mapping(self, source=()):
        result = Facts(self)
        for key, value in (source.items() if isinstance(source, dict) else source):
            self.charge()
            result[key] = value
        return result

    def bindings(self, states):
        self.charge(len(states))

    def clone(self, states):
        self.reserve(len(states))
        self.saved.append(states)
        return tuple(self.mapping(state) for state in states)

    def literal(self, expr):
        require(expr.tag in (INT, BOOL), 'require exact scalar witness literals')
        require(type(expr.value) is (int if expr.tag == INT else bool),
                'require exact witness literal value types')
        require(expr.tag != INT or -(1 << 63) <= expr.value < (1 << 63),
                'require exact signed64 witness literal bounds')
        return ('literal', expr.tag, expr.value)

    def value(self, expr, env, depth=0):
        self.charge()
        require(depth <= DEPTH_LIMIT and isinstance(expr, Expr) and expr.tag in (INT, BOOL),
                'require bounded typed scalar witness expressions')
        if expr.kind == 'constant':
            return self.literal(expr)
        if expr.kind in ('local', 'temporary'):
            return env.get(expr)
        require(len(expr.args) <= 2, 'limit scalar witness operand staging to two')
        args = [self.value(arg, env, depth + 1) for arg in expr.args]
        if expr.kind == 'not' and len(args) == 1 and args[0] is not None and args[0][:2] == ('literal', BOOL):
            return ('literal', BOOL, not args[0][2])
        if expr.kind == 'binary' and expr.value == '==' and len(args) == 2:
            a, b = args
            if a is not None and b is not None and a[1] == b[1]:
                if a == b:
                    return ('literal', BOOL, True)
                if a[0] == b[0] == 'literal':
                    return ('literal', BOOL, False)
        # Arithmetic and other comparisons remain distinct opaque definitions.
        return self.token(expr.tag)

    def pack(self, layout, fields, context):
        self.charge(len(fields) + 1)
        origin = self.token(('owner', layout))[2]
        self.reserve(len(fields) + 1)
        self.nodes[origin] = (layout, context, tuple(fields))
        return origin

    def field(self, origin, field):
        self.charge()
        require(origin in self.nodes and type(field) is int and 0 <= field < len(self.nodes[origin][2]),
                'require an exact live witness owner field')
        return self.nodes[origin][2][field]

    def prefix(self, function):
        env, owners, operands = self.mapping(), self.mapping(), self.mapping()
        for stmt in function.body:
            self.charge()
            kind = stmt[0]
            if kind == 'let':
                env[stmt[1]] = self.value(stmt[2], env)
            elif kind == 'store':
                env[Expr(stmt[2].tag, 'local', stmt[1])] = self.value(stmt[2], env)
            elif kind == 'own_pack':
                require(1 <= len(stmt[3]) <= 16, 'limit caller witness field staging to sixteen')
                fields = tuple(self.value(v, env) for v in stmt[3])
                require(all(v is not None and v[0] == 'literal' for v in fields),
                        'require exact actual literal caller witness fields')
                operands[stmt[1]] = self.pack(stmt[2], fields, ('caller',))
            elif kind == 'own_store':
                require(stmt[1] not in owners and stmt[3] in operands,
                        'require exact first-call owner witness transfer')
                owners[stmt[1]] = operands.pop(stmt[3])
            elif kind == 'own_move':
                require(stmt[2] in owners, 'require a live caller witness origin')
                operands[stmt[1]] = owners.pop(stmt[2])
            elif kind == 'multi_borrow_call':
                require(len(stmt[2]) == 2 and len(set(stmt[2])) == 2 and not operands,
                        'require two distinct complete literal caller witnesses')
                require(all(slot in owners for slot in stmt[2]), 'require live literal caller witness roots')
                return tuple(tuple(Expr(value[1], 'constant', value[2])
                                   for value in self.nodes[owners[slot]][2]) for slot in stmt[2])
            else:
                require(False, 'require a straight-line actual literal helper-owner witness prefix')
            self.bindings((env, owners, operands))
        require(False, 'require an actual first helper-owner call')

    def block(self, body, states, borrowed, depth=0, context=('helper',)):
        require(depth <= DEPTH_LIMIT, 'limit helper witness nesting to 32')
        env, owners, operands = self.clone(states)
        terminal = False
        try:
            for stmt in body:
                self.charge()
                require(not terminal, 'refuse unreachable helper witness statements')
                kind = stmt[0]
                if kind == 'let':
                    env[stmt[1]] = self.value(stmt[2], env)
                elif kind == 'store':
                    env[Expr(stmt[2].tag, 'local', stmt[1])] = self.value(stmt[2], env)
                elif kind == 'multi_borrow_read':
                    self.charge()
                    env[stmt[1]] = borrowed.get((stmt[2], stmt[3]))
                    if env[stmt[1]] is None:
                        env[stmt[1]] = self.token(stmt[1].tag)
                elif kind == 'multi_borrow_write':
                    require(stmt[1] == 0, 'retain exclusive formal write authority')
                    self.value(stmt[3], env)
                    borrowed[(stmt[1], stmt[2])] = None
                elif kind == 'own_pack':
                    require(1 <= len(stmt[3]) <= 16, 'limit helper witness field staging to sixteen')
                    fields = []
                    for value in stmt[3]:
                        self.charge()
                        if isinstance(value.tag, tuple):
                            require(value in operands, 'require an exact consumed child witness')
                            fields.append(('child', operands.pop(value)))
                        else:
                            fields.append(self.value(value, env))
                    operands[stmt[1]] = self.pack(stmt[2], fields, context)
                elif kind == 'own_store':
                    require(stmt[1] not in owners and stmt[3] in operands, 'require one local witness origin transfer')
                    owners[stmt[1]] = operands.pop(stmt[3])
                elif kind == 'own_move':
                    require(stmt[2] in owners, 'require a live helper witness generation')
                    operands[stmt[1]] = owners.pop(stmt[2])
                elif kind == 'own_unpack':
                    require(stmt[1] in owners, 'require a live unpack witness origin')
                    origin = owners.pop(stmt[1])
                    require(self.nodes[origin][0] == stmt[2] and len(self.nodes[origin][2]) == len(stmt[3]),
                            'require exact nominal unpack witness fields')
                    for index, target in enumerate(stmt[3]):
                        fact = self.field(origin, index)
                        if isinstance(target.tag, tuple):
                            require(fact is not None and fact[0] == 'child', 'require exact child witness origin')
                            operands[target] = fact[1]
                        else:
                            env[target] = fact
                elif kind in ('own_read', 'path_read'):
                    require(stmt[2] in owners, 'require a live observed helper witness owner')
                    origin = owners[stmt[2]]
                    if kind == 'path_read':
                        for field in stmt[3]:
                            fact = self.field(origin, field)
                            require(fact is not None and fact[0] == 'child', 'require exact witness child traversal')
                            origin = fact[1]
                    env[stmt[1]] = self.field(origin, stmt[4] if kind == 'path_read' else stmt[3])
                elif kind == 'assert':
                    require(self.value(stmt[1], env) == ('literal', BOOL, True),
                            'require a proved actual helper-owner assertion witness')
                elif kind == 'if':
                    self.value(stmt[1], env)
                    require(not operands, 'require empty witness operands at branch')
                    left = self.block(stmt[2], (env, owners, operands), self.mapping(borrowed), depth + 1, context)
                    right = self.block(stmt[3], (env, owners, operands), self.mapping(borrowed), depth + 1, context)
                    continuing = [arm for arm in (left, right) if not arm[2]]
                    terminal = not continuing
                    if continuing:
                        require(all(arm[0][1] == owners and not arm[0][2] for arm in continuing),
                                'require restored branch witness owners')
                        env = self.intersect([arm[0][0] for arm in continuing])
                        borrowed = self.intersect([arm[1] for arm in continuing])
                    else:
                        owners = {}
                elif kind == 'while':
                    require(not operands, 'require empty witness operands at loop')
                    self.value(stmt[1], env)
                    # The separate rank proof checks exact initialization and work.
                    require(stmt[1].kind == 'binary' and stmt[1].value == '<' and
                            stmt[1].args[0].kind == 'local', 'require the proved rank witness loop')
                    rank = stmt[1].args[0]
                    iteration = self.mapping(env)
                    self.charge(len(env))
                    iteration_borrowed = self.mapping(borrowed)
                    def invalidate(statements, nesting=0):
                        require(nesting <= DEPTH_LIMIT, 'limit loop witness write traversal depth')
                        for item in statements:
                            self.charge()
                            tag = item[0]
                            if tag == 'store':
                                iteration.pop(Expr(item[2].tag, 'local', item[1]), None)
                            elif tag in ('let', 'multi_borrow_read', 'own_read', 'path_read'):
                                iteration.pop(item[1], None)
                            elif tag == 'own_unpack':
                                for target in item[3]:
                                    self.charge()
                                    iteration.pop(target, None)
                            elif tag == 'multi_borrow_write':
                                iteration_borrowed.pop((item[1], item[2]), None)
                            elif tag == 'if':
                                invalidate(item[2], nesting + 1)
                                invalidate(item[3], nesting + 1)
                            elif tag == 'while':
                                invalidate(item[2], nesting + 1)
                    invalidate(stmt[2])
                    iteration[rank] = self.token(INT)
                    self.serial += 1
                    result = self.block(stmt[2], (iteration, owners, {}), iteration_borrowed, depth + 1,
                                        context + (self.serial,))
                    require(not result[2] and result[0][1] == owners and not result[0][2],
                            'require exact invariant loop witness owners')
                    # Only identical facts also present on the zero-trip edge survive.
                    env = self.intersect([env, result[0][0]])
                    borrowed = self.intersect([borrowed, result[1]])
                elif kind == 'return':
                    self.value(stmt[1], env)
                    require(not owners and not operands, 'require successful helper witness consumption')
                    terminal = True
                else:
                    require(False, 'require bounded helper-owner witness statements')
                self.bindings((env, owners, operands, borrowed))
            return (env, owners, operands), borrowed, terminal
        finally:
            self.saved.pop()

    def intersect(self, maps):
        result = self.mapping()
        for key, value in maps[0].items():
            self.charge(len(maps))
            if all(mapping.get(key) == value for mapping in maps):
                result[key] = value
        return result


def prove_helper_owners(functions):
    from nanoisa_owned_origins import prove_owner_reassignment
    from nanoisa_owned_control import prove_exclusive_helper_control
    require(len(functions) == 2 and [p[2] for p in functions[1].params] == [2, 1],
            'require exact helper-local exclusive/shared parameters')
    # The origin proof consumes no borrowed formal and never changes the IR.
    prove_owner_reassignment(functions[0], helper_locals=True)
    prove_owner_reassignment(functions[1], helper_locals=True)
    control = prove_exclusive_helper_control(functions[1], helper_locals=True)
    proof = WitnessProof()
    witness = proof.prefix(functions[0])
    borrowed = proof.mapping(((parameter, field), proof.literal(value))
                             for parameter, fields in enumerate(witness) for field, value in enumerate(fields))
    result = proof.block(functions[1].body, ({}, {}, {}), borrowed)
    require(result[2], 'require a successful terminal helper witness')
    control['witness'] = {'tokens': proof.tokens, 'traversal': proof.work, 'allocated_bindings': proof.allocated,
                          'token_limit': TOKEN_LIMIT, 'binding_limit': BINDING_LIMIT,
                          'traversal_limit': TRAVERSAL_LIMIT}
    return witness, control
