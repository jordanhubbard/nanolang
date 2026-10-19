"""I retain ordered local resource transfers in one bounded owned region tree."""
from nanoisa_reconstruction import Analyze, Emit, Expr, Function, INT, BOOL, SIMPLE, require


VOID = 0

# I inline only nontrapping scalar pretests without duplicated/discarded operands.
OWNED_PRETEST = {
    "NOP", "PUSH_I64", "PUSH_BOOL", "LOAD_LOCAL",
    "EQ", "NE", "LT", "LE", "GT", "GE",
    "I64_EQ", "I64_NE", "I64_LT_S", "I64_LE_S", "I64_GT_S", "I64_GE_S",
    "I64_LT_U", "I64_LE_U", "I64_GT_U", "I64_GE_U",
    "BOOL_AND", "BOOL_OR", "BOOL_NOT", "AND", "OR", "NOT", "CAST_BOOL", "CAST_INT",
}


# I materialize only the source-qualified exact-INT wrapping operations.
OWNED_WRAPPING_PRETEST = {"ADD", "SUB", "MUL"}


def owned_type(layout):
    return ('owner', layout)


class OwnedAnalyze(Analyze):
    def __init__(self, module, index=0):
        self.module = module
        require(module.get('profile') == 'owned-local-v1' and module['entry'] == 0 and
                0 < len(module['functions']) <= 8, 'require bounded verified resource functions')
        self.fn = module['functions'][index]
        self.index = index
        self.layouts = module['layouts']
        require(0 < len(self.layouts) <= 16, 'require bounded resource layouts')
        self.fields = []
        self.depths = []
        for i, layout in enumerate(self.layouts):
            require(layout['identity'] == i and layout['resource'] is True and
                    0 < len(layout['fields']) <= 16 and
                    all(t in (INT, BOOL, 8) for t in layout['fields']),
                    'require exact resource identity and bounded fields')
            fields = layout.get('field_types', [[t, 4294967295] for t in layout['fields']])
            require(isinstance(fields, list) and
                    all(isinstance(pair, list) and len(pair) == 2 and
                        all(type(value) is int for value in pair) for pair in fields) and
                    len(fields) == len(layout['fields']) and
                    all(pair[0] == tag for pair, tag in zip(fields, layout['fields'])),
                    'require complete exact resource field descriptors')
            depth = 1
            for tag, child in fields:
                if tag == 8:
                    require(0 <= child < i, 'require exact prior resource children')
                    depth = max(depth, self.depths[child] + 1)
                    require(depth <= 3, 'limit consuming resource layouts to depth three')
                else:
                    require(tag in (INT, BOOL) and child == 4294967295,
                            'require exact scalar fields without child layout identities')
            self.depths.append(depth)
            self.fields.append([owned_type(child) if tag == 8 else tag for tag, child in fields])
        self.declarations = self.fn['local_types']
        require(len(self.declarations) == self.fn['locals'] <= 128,
                'require complete local declarations')
        self.types = {i: t for i, (t, _) in enumerate(self.declarations) if t in (INT, BOOL)}

        def declared(pair):
            tag, layout = pair
            require(tag in (INT, BOOL) and layout == 4294967295 or
                    tag == 8 and 0 <= layout < len(self.layouts),
                    'require exact scalar or nominal resource declarations')
            return owned_type(layout) if tag == 8 else tag
        self.params = [declared(t) for t in self.fn['params']]
        self.modes = self.fn.get('param_modes', [0] * len(self.params))
        require(len(self.modes) == len(self.params) and all(m in (0, 1, 2) for m in self.modes),
                'require exact value or borrowed parameter modes')
        self.borrowed = any(any(f.get('param_modes', [])) for f in module['functions'])
        if self.borrowed:
            require(max(self.depths) <= 2, 'retain borrowed modules at depth two')
            require(len(module['functions']) == 2 and
                    module['functions'][0].get('param_modes', []) == [] and
                    module['functions'][1].get('param_modes') in ([1], [2]),
                    'require the separate entry-to-borrowed-helper profile')
        self.exclusive = self.borrowed and module['functions'][1].get('param_modes') == [2]
        self.helper = self.borrowed and index == 1
        if self.helper:
            require(len(self.params) == 1 and isinstance(self.params[0], tuple),
                    'require one exact borrowed resource parameter')
            require(all(t in (INT, BOOL) for t in self.fields[self.params[0][1]]),
                    'require an exact scalar-leaf borrowed helper parameter')
            self.params[0] = ('borrow', self.params[0][1], self.modes[0])
        self.result = VOID if self.fn['result_type'] == [VOID, 4294967295] else declared(self.fn['result_type'])
        require(self.fn.get('result_count') == (0 if self.result == VOID else 1),
                'require the exact owned helper result count')
        require(self.fn['result'] == self.fn['result_type'][0], 'require matching result declarations')
        require(len(self.params) <= 8 and len(self.params) <= len(self.declarations) and
                all(declared(self.declarations[i]) == (owned_type(t[1]) if self.modes[i] else t) for i, t in enumerate(self.params)),
                'require exact initial parameter slots')
        require(index != 0 or not self.params and self.result == INT,
                'require a scalar entry without parameters')
        self.live = {i: t[1] for i, t in enumerate(self.params) if isinstance(t, tuple) and t[0] == 'owner'}
        self.used = set(self.live)
        self.seen = set()
        self.calls = set()
        self.code = self.fn['code']
        if self.helper:
            require(self.result in (INT, BOOL) and
                    all(ins['op'] in OWNED_PRETEST | {'STORE_LOCAL', 'REF_GET', 'RET'} | ({'ADD', 'REF_SET'} if self.exclusive else set()) for ins in self.code),
                    'require a straight-line scalar borrowed helper')
        require(0 < len(self.code) <= 4096, 'require bounded function instructions')
        self.positions = {ins['pc']: n for n, ins in enumerate(self.code)}
        require(len(self.positions) == len(self.code), 'require unique instruction boundaries')
        self.positions[self.fn['size']] = len(self.code)
        interiors = {n + offset for n, ins in enumerate(self.code)
                     if ins['op'] == 'REGION_BEGIN' for offset in (1, 2, 3)}
        self.backedges = {}
        for n, ins in enumerate(self.code):
            if ins['op'] in ('JMP', 'JMP_TRUE', 'JMP_FALSE'):
                require(ins.get('target') in self.positions, 'require exact owned branch boundaries')
                target = self.positions[ins['target']]
                require(target not in interiors, 'require branches outside ephemeral read interiors')
                if target <= n:
                    require(ins['op'] == 'JMP' and target not in self.backedges,
                            'require one unconditional backedge per owned loop header')
                    self.backedges[target] = n

    def owner_local(self, slot):
        require(slot in self.live, 'require a live exact local owner')
        return self.live[slot]

    def observation(self, i, end, stack, body, allow_call=True):
        require(i + 3 < end, 'require a complete ephemeral read within its owned region')
        for offset in (1, 2, 3):
            self.touch(i + offset)
        borrow, read, region_end_instruction = self.code[i + 1:i + 4]
        require(borrow['op'] in ('BORROW_LOCAL_SHARED', 'BORROW_LOCAL_EXCLUSIVE') and
                read['op'] in ('REF_GET', 'CALL_REF') and
                region_end_instruction['op'] == 'REGION_END' and len(borrow['args']) == 2 and
                len(read['args']) == 2,
                'require one complete ephemeral reference operation')
        slot = borrow['args'][1]
        layout = self.owner_local(slot)
        if read['op'] == 'CALL_REF':
            require(allow_call, 'refuse calls in owned observation pretests')
            require(borrow['op'] == ('BORROW_LOCAL_EXCLUSIVE' if self.exclusive else 'BORROW_LOCAL_SHARED'),
                    'require the exact helper reference mode')
            require(self.borrowed and self.index == 0 and read['args'] == [1, borrow['args'][0]] and
                    self.module['functions'][1]['params'] == [[8, layout]],
                    'require exact caller owner and borrowed helper identity')
            target = Expr(self.module['functions'][1]['result'], 'temporary', read['pc'])
            body.append(('borrow_call', target, slot))
            self.calls.add(1)
        else:
            require(borrow['op'] == 'BORROW_LOCAL_SHARED', 'require shared ephemeral field reads')
            field = read['args'][1]
            require(borrow['args'][0] == read['args'][0] and
                    0 <= field < len(self.fields[layout]),
                    'require a field within the exact owner layout')
            require(all(t in (INT, BOOL) for t in self.fields[layout]),
                    'require a scalar-leaf owner for ephemeral field observation')
            require(self.fields[layout][field] in (INT, BOOL), 'require a scalar shared field')
            target = Expr(self.fields[layout][field], 'temporary', read['pc'])
            body.append(('own_read', target, slot, field))
        stack.append(target)

    def region(self, start, end, initialized, live, used, depth=0):
        require(depth <= 32, 'limit owned region nesting to 32')
        initialized = set(initialized)
        self.live, self.used = dict(live), set(used)
        stack = []
        body = []
        i = start
        while i < end:
            if i in self.backedges:
                require(not stack, 'require an empty owned loop-entry stack')
                back = self.backedges[i]
                require(i < back < end, 'require an owned loop within its enclosing region')
                branch = next((n for n in range(i, back)
                               if self.code[n]['op'] in ('JMP_TRUE', 'JMP_FALSE')), None)
                require(branch is not None, 'require an owned loop pretest')
                require(self.positions[self.code[branch]['target']] == back + 1,
                        'require an owned pretest exit immediately after its backedge')
                condition_stack, condition_statements = [], []
                materialized = any(self.code[n]['op'] == 'REGION_BEGIN' or
                                   self.code[n]['op'] in OWNED_WRAPPING_PRETEST for n in range(i, branch))
                n = i
                while n < branch:
                    condition_instruction = self.code[n]
                    if materialized and condition_instruction['op'] == 'REGION_BEGIN':
                        self.touch(n)
                        self.observation(n, branch, condition_stack, condition_statements, allow_call=False)
                        n += 4
                        continue
                    require(condition_instruction['op'] in OWNED_PRETEST | OWNED_WRAPPING_PRETEST,
                            'require a nontrapping pure scalar owned pretest')
                    if condition_instruction['op'] == 'LOAD_LOCAL':
                        require(condition_instruction['arg'] in self.types,
                                'require a scalar owned pretest local')
                    self.simple(n, condition_stack, initialized, condition_statements, pure=not materialized)
                    require(all(value.tag in (INT, BOOL) for value in condition_stack),
                            'require exact scalar owned pretest values')
                    n += 1
                condition = self.pop(condition_stack, BOOL)
                require(not condition_stack, 'require one owned loop condition')
                if self.code[branch]['op'] == 'JMP_TRUE':
                    condition = Expr(BOOL, 'not', None, (condition,))
                self.touch(branch)
                entry_live, entry_used = dict(self.live), set(self.used)
                loop_body, _, loop_live, loop_used, loop_returns = self.region(
                    branch + 1, back, initialized, entry_live, entry_used, depth + 1)
                require(loop_returns or loop_live == entry_live,
                        'require exact carried owner identities at the backedge')
                self.touch(back)
                body.append(('while_observe', condition, condition_statements, loop_body) if materialized else
                            ('while', condition, loop_body))
                # I retain the zero-iteration state. Body-local declarations execute
                # inside while on each iteration; I analyze their syntax only once.
                self.live, self.used = entry_live, entry_used | loop_used
                i = back + 1
                continue
            ins = self.code[i]
            op, arg, pc = ins['op'], ins['arg'], ins['pc']
            if op in ('JMP_TRUE', 'JMP_FALSE'):
                self.touch(i)
                condition = self.pop(stack, BOOL)
                require(not stack, 'require empty operand stacks at owned branches')
                target = self.positions[ins['target']]
                require(i < target <= end, 'require a forward owned region boundary')
                if op == 'JMP_TRUE':
                    condition = Expr(BOOL, 'not', None, (condition,))
                separator = target - 1
                has_else = separator > i and self.code[separator]['op'] == 'JMP'
                join = self.positions[self.code[separator]['target']] if has_else else target
                require(target <= join <= end, 'require a non-crossing owned diamond')
                entry_live, entry_used = dict(self.live), set(self.used)
                left, li, ll, lu, lr = self.region(i + 1, separator if has_else else target,
                                                  initialized, entry_live, entry_used, depth + 1)
                if has_else:
                    self.touch(separator)
                    right, ri, rl, ru, rr = self.region(target, join, initialized,
                                                       entry_live, entry_used, depth + 1)
                else:
                    right, ri, rl, ru, rr = [], set(initialized), entry_live, entry_used, False
                body.append(('if', condition, left, right))
                self.used = lu | ru
                if lr and rr:
                    require(join == end, 'do not discard a terminal owned continuation')
                    return body, initialized, {}, self.used, True
                if lr:
                    initialized, self.live = ri, rl
                elif rr:
                    initialized, self.live = li, ll
                else:
                    require(ll == rl, 'require exact live-owner agreement at joins')
                    initialized, self.live = li & ri, ll
                require(all(entry_live.get(slot) == layout for slot, layout in self.live.items()),
                        'refuse branch-local owners escaping a structured region')
                i = join
                continue
            require(op != 'JMP', 'refuse unstructured owned jumps')
            if op not in SIMPLE or op == 'CALL':
                self.touch(i)
            if op == 'OWN_PACK':
                require(0 <= arg < len(self.layouts), 'require a retained nominal layout')
                fields = self.fields[arg]
                values = tuple(reversed([self.pop(stack, t) for t in reversed(fields)]))
                target = Expr(owned_type(arg), 'temporary', pc)
                body.append(('own_pack', target, arg, values))
                stack.append(target)
            elif op == 'OWN_STORE_LOCAL':
                require(0 <= arg < len(self.declarations) and arg not in self.used,
                        'require a fresh declared owner destination')
                tag, layout = self.declarations[arg]
                require(tag == 8, 'require an owned record local')
                value = self.pop(stack, owned_type(layout))
                body.append(('own_store', arg, layout, value))
                self.live[arg] = layout
                self.used.add(arg)
            elif op == 'OWN_MOVE_LOCAL':
                layout = self.owner_local(arg)
                target = Expr(owned_type(layout), 'temporary', pc)
                body.append(('own_move', target, arg))
                del self.live[arg]
                stack.append(target)
            elif op == 'OWN_UNPACK_LOCAL':
                layout = self.owner_local(arg)
                values = tuple(Expr(t, 'temporary', f'{pc}_{f}')
                               for f, t in enumerate(self.fields[layout]))
                body.append(('own_unpack', arg, layout, values))
                del self.live[arg]
                stack.extend(values)
            elif op == 'REF_SET':
                require(self.helper and self.exclusive and len(ins['args']) == 2 and arg == 0,
                        'require the exact exclusive helper reference')
                field, layout = ins['args'][1], self.params[0][1]
                require(0 <= field < len(self.fields[layout]), 'require an exact exclusive field')
                value = self.pop(stack, self.fields[layout][field])
                body.append(('borrow_write', field, value))
            elif op == 'REF_GET':
                require(self.helper and len(ins['args']) == 2 and arg == 0,
                        'require the exact borrowed helper parameter reference')
                field, layout = ins['args'][1], self.params[0][1]
                require(0 <= field < len(self.fields[layout]), 'require an exact borrowed parameter field')
                require(self.fields[layout][field] in (INT, BOOL), 'require a scalar shared field')
                target = Expr(self.fields[layout][field], 'temporary', pc)
                body.append(('borrow_read', target, field))
                stack.append(target)
            elif op == 'REGION_BEGIN':
                self.observation(i, end, stack, body)
                i += 3
            elif op == 'CALL':
                require(not self.borrowed, 'refuse mixed consuming and borrowed calls')
                require(0 < arg < len(self.module['functions']), 'require a direct non-entry callee')
                callee = self.module['functions'][arg]
                params = [owned_type(t[1]) if t[0] == 8 else t[0] for t in callee['params']]
                values = tuple(reversed([self.pop(stack, t) for t in reversed(params)]))
                tag, layout = callee['result_type']
                result_type = owned_type(layout) if tag == 8 else tag
                call = Expr(result_type, 'call', arg, values)
                if result_type == VOID:
                    body.append(('call_void', call))
                else:
                    target = Expr(result_type, 'temporary', pc)
                    body.append(('let', target, call))
                    stack.append(target)
                self.calls.add(arg)
            elif op == 'ASSERT':
                body.append(('assert', self.pop(stack, BOOL)))
            elif op == 'RET':
                require(i == end - 1 and not self.live,
                        'require all owners consumed before terminal return')
                if self.result == VOID:
                    require(not stack, 'require an empty operand stack at VOID return')
                    body.append(('return_void',))
                else:
                    result = self.pop(stack, self.result)
                    require(not stack, 'require an empty operand stack after exact return')
                    body.append(('return', result))
                return body, initialized, self.live, self.used, True
            else:
                require(op in SIMPLE and op not in ('CALL', 'PUSH_U8', 'PUSH_F64', 'F64_FROM_BITS', 'F64_TO_BITS', 'F64_NEG'),
                        'refuse unsupported resource-route instruction ' + op)
                # I isolate the suffix above the last staged owner before scalar evaluation.
                split = max((n + 1 for n, value in enumerate(stack)
                             if value.tag not in (INT, BOOL)), default=0)
                prefix, scalar = stack[:split], stack[split:]
                if op in ('LOAD_LOCAL', 'STORE_LOCAL'):
                    require(arg in self.types, 'require a declared scalar local')
                self.simple(i, scalar, initialized, body)
                require(all(value.tag in (INT, BOOL) for value in scalar),
                        'require exact scalar resource-route values')
                stack = prefix + scalar
            i += 1
        require(not stack, 'require an empty operand stack at owned region exit')
        return body, initialized, self.live, self.used, False

    def run(self):
        parameters = [('own_parameter', i, layout) for i, layout in self.live.items()]
        body, _, _, _, returns = self.region(0, len(self.code), set(range(len(self.params))),
                                             self.live, self.used)
        require(returns, 'require every owned function path to return')
        require(len(self.seen) == len(self.code), 'require complete owned instruction coverage')
        return Function(self.fn['name'], self.params, self.result, self.types, self.fn['names'], parameters + body)


class OwnedEmit(Emit):
    def __init__(self, functions, language, layouts):
        super().__init__(functions, language)
        self.layouts = layouts
        self.exclusive = any(isinstance(t, tuple) and t[0] == 'borrow' and len(t) == 3 and t[2] == 2
                             for fn in functions for t in fn.params)
        self.inline_add = self.exclusive and language == 'nano'
        if self.inline_add:
            def validate_inline(node, direct=False):
                if isinstance(node, Expr):
                    require(node.kind != 'arithmetic' or node.value != 'add' or direct,
                            'require ADD at an explicit inline statement boundary')
                    for child in node.args:
                        validate_inline(child)
                elif isinstance(node, (list, tuple)):
                    if node and node[0] == 'let':
                        validate_inline(node[1])
                        validate_inline(node[2], True)
                    else:
                        for child in node:
                            validate_inline(child)
            for function in functions:
                validate_inline(function.body)
        self.fields = [[owned_type(child) if tag == 8 else tag
                        for tag, child in layout.get('field_types', [[t, 4294967295] for t in layout['fields']])]
                       for layout in layouts]

    def nano_local_cost(self, function):
        # My emitted subset allocates named lets, constructor staging and pattern slots.
        def statements_cost(body):
            total = 0
            for stmt in body:
                kind = stmt[0]
                if kind == 'let':
                    value = stmt[2]
                    total += 7 if value.kind == 'arithmetic' and value.value == 'add' else 1
                elif kind in ('borrow_read', 'borrow_call', 'own_store', 'own_move', 'own_read'):
                    total += 1
                elif kind == 'own_pack':
                    total += 1 + len(stmt[3])
                elif kind == 'own_unpack':
                    # Child-bearing patterns stage every field before named projections.
                    staged = any(isinstance(target.tag, tuple) for target in stmt[3])
                    total += 1 + (3 if staged else 2) * len(stmt[3])
                elif kind == 'if':
                    total += statements_cost(stmt[2]) + statements_cost(stmt[3])
                elif kind == 'while':
                    total += statements_cost(stmt[2])
                elif kind == 'while_observe':
                    total += statements_cost(stmt[2]) + statements_cost(stmt[3])
                else:
                    require(kind in ('store', 'borrow_write', 'assert', 'return'),
                            'require counted exclusive Nano statement storage')
            return total
        return len(function.locals) + len(function.params) + statements_cost(function.body)

    def name(self, index):
        return 'main' if self.language == 'nano' and index == 0 else super().name(index)

    def type(self, tag):
        if tag == VOID:
            return 'void'
        if isinstance(tag, tuple) and tag[0] == 'borrow':
            mutable = len(tag) == 3 and tag[2] == 2
            return (('' if mutable else 'const ') + f'NlrOwned{tag[1]} *' if self.language == 'c' else
                    ('&mut ' if mutable else '&') + f'NlrOwned{tag[1]}')
        if isinstance(tag, tuple) and tag[0] == 'owner':
            return f'NlrOwned{tag[1]}'
        return super().type(tag)

    def inline_add_statement(self, target, value, indent):
        name = self.expression(target)
        a, b = name + '_a', name + '_b'
        low, high, boundary = name + '_low', name + '_high', name + '_boundary'
        self.line(f'let {a}: int = {self.expression(value.args[0])}', indent)
        self.line(f'let {b}: int = {self.expression(value.args[1])}', indent)
        self.line(f'let {low}: int = (- -9223372036854775807 1)', indent)
        self.line(f'let {high}: int = 9223372036854775807', indent)
        self.line(f'let mut {name}: int = 0', indent)
        # I use exactly the helper formulas; only result assignment replaces return.
        from nanoisa_reconstruction import NANO_INTEGER_HELPERS
        helper = NANO_INTEGER_HELPERS['add']
        positive = 'return (+ low (- (- a boundary) 1))'
        negative = 'return (+ high (+ (- a boundary) 1))'
        require(positive in helper and negative in helper and 'return (+ a b)' in helper,
                'require unchanged guarded ADD formulas')
        def formula(text):
            import re
            names = {'a': a, 'b': b, 'low': low, 'high': high, 'boundary': boundary}
            return re.sub(r'\b(a|b|low|high|boundary)\b', lambda m: names[m[0]], text[7:])
        self.line(f'if (> {b} 0) {{', indent)
        self.line(f'let {boundary}: int = (- {high} {b})', indent + 1)
        self.line(f'if (> {a} {boundary}) {{ set {name} {formula(positive)} }} else {{ set {name} (+ {a} {b}) }}', indent + 1)
        self.line(f'}} else {{ if (< {b} 0) {{', indent)
        self.line(f'let {boundary}: int = (- {low} {b})', indent + 1)
        self.line(f'if (< {a} {boundary}) {{ set {name} {formula(negative)} }} else {{ set {name} (+ {a} {b}) }}', indent + 1)
        self.line(f'}} else {{ set {name} (+ {a} {b}) }} }}', indent)

    def statements(self, statements, indent=1):
        c = self.language == 'c'
        for stmt in statements:
            kind = stmt[0]
            if kind == 'let' and self.inline_add and stmt[2].kind == 'arithmetic' and stmt[2].value == 'add':
                self.inline_add_statement(stmt[1], stmt[2], indent)
            elif kind == 'borrow_write':
                _, field, value = stmt
                self.line(f'nlr_a0->f{field} = {self.expression(value)};' if c else
                          f'set nlr_a0.f{field} {self.expression(value)}', indent)
            elif kind == 'while_observe':
                _, condition, observations, loop_body = stmt
                self.line('while (true) {' if c else 'while true {', indent)
                self.statements(observations, indent + 1)
                value = self.expression(condition)
                self.line(f'if (!({value})) {{ break; }}' if c else
                          f'if (not {value}) {{ break }}', indent + 1)
                self.statements(loop_body, indent + 1)
                self.line('}', indent)
            elif kind in ('borrow_read', 'borrow_call'):
                _, target, operand = stmt
                if kind == 'borrow_read':
                    value = f'nlr_a0->f{operand}' if c else f'nlr_a0.f{operand}'
                else:
                    value = f'{self.name(1)}(&{self.local(operand)})' if c else f'({self.name(1)} {"&mut " if self.exclusive else "&"}{self.local(operand)})'
                self.line(f'{self.type(target.tag)} {self.expression(target)} = {value};' if c else
                          f'let {self.expression(target)}: {self.type(target.tag)} = {value}', indent)
                if c:
                    self.line(f'(void){self.expression(target)};', indent)
            elif kind == 'call_void':
                self.line(self.expression(stmt[1]) + (';' if c else ''), indent)
            elif kind == 'return_void':
                self.line('return;' if c else 'return', indent)
            elif kind == 'assert':
                expression = self.expression(stmt[1])
                self.line(f'if (!({expression})) {{ fputs("Assertion failed\\n", stderr); abort(); }}' if c else
                          f'assert {expression}', indent)
            elif kind == 'own_parameter':
                _, slot, layout = stmt
                name, typ = self.local(slot), self.type(owned_type(layout))
                self.line(f'{typ} {name} = nlr_a{slot};' if c else
                          f'let {name}: {typ} = nlr_a{slot}', indent)
            elif kind == 'own_pack':
                _, target, layout, values = stmt
                fields = ', '.join((f'.f{i} = ' if c else f'f{i}: ') + self.expression(v)
                                   for i, v in enumerate(values))
                name, typ = self.expression(target), self.type(target.tag)
                self.line(f'{typ} {name} = ({typ}){{{fields}}};' if c else
                          f'let {name}: {typ} = {typ} {{ {fields} }}', indent)
            elif kind in ('own_store', 'own_move'):
                if kind == 'own_store':
                    _, slot, layout, value = stmt
                    name, typ, value = self.local(slot), f'NlrOwned{layout}', self.expression(value)
                else:
                    _, target, slot = stmt
                    name, typ, value = self.expression(target), self.type(target.tag), self.local(slot)
                self.line(f'{typ} {name} = {value};' if c else f'let {"mut " if self.exclusive else ""}{name}: {typ} = {value}', indent)
            elif kind == 'own_read':
                _, target, slot, field = stmt
                name, typ, value = self.expression(target), self.type(target.tag), f'{self.local(slot)}.f{field}'
                self.line(f'{typ} {name} = {value};' if c else f'let {name}: {typ} = {value}', indent)
                if c:
                    self.line(f'(void){name};', indent)
            elif kind == 'own_unpack':
                _, slot, layout, targets = stmt
                if not c:
                    fields = ', '.join(f'f{i}' for i in range(len(targets)))
                    self.line(f'let NlrOwned{layout} {{ {fields} }} = {self.local(slot)}', indent)
                for i, target in enumerate(targets):
                    name, typ = self.expression(target), self.type(target.tag)
                    self.line(f'{typ} {name} = {self.local(slot)}.f{i};' if c else
                              f'let {name}: {typ} = f{i}', indent)
                    if c:
                        self.line(f'(void){name};', indent)
            else:
                super().statements([stmt], indent)

    def run(self, entry):
        c = self.language == 'c'
        if self.inline_add:
            require(all(self.nano_local_cost(fn) <= 128 for fn in self.functions),
                    'limit generated exclusive Nano functions to 128 local slots')
        declarations = ['#include <stdint.h>\n#include <stdbool.h>\n#include <stdio.h>\n#include <stdlib.h>'] if c else []
        for i, layout in enumerate(self.layouts):
            fields = (' '.join(f'{self.type(t)} f{j};' for j, t in enumerate(self.fields[i])) if c else
                      ', '.join(f'f{j}: {self.type(t)}' for j, t in enumerate(self.fields[i])))
            declarations.append(f'typedef struct {{ {fields} }} NlrOwned{i};' if c else
                                f'resource struct NlrOwned{i} {{ {fields} }}')
        body = super().run(entry).replace(
            '# I reconstruct executable scalar regions; original shadows are not retained.',
            '# I reconstruct ordered resource transfers; original shadows are not retained.')
        if not c:
            # I emit the retained entry directly; an extra wrapper exceeds the producer budget.
            wrapper = 'fn main() -> int {\n    return (main)\n}\n'
            require(body.endswith(wrapper), 'require the exact generated entry wrapper')
            body = body[:-len(wrapper)]
            require(sum(line.startswith('fn ') for line in body.splitlines()) <= 7,
                    'reserve one of eight producer functions for shadow execution')
        borrowed = (len(self.functions) == 2 and self.functions[1].params and
                    isinstance(self.functions[1].params[0], tuple) and
                    self.functions[1].params[0][0] == 'borrow')
        if not c and borrowed:
            require(sum(line.startswith('fn ') for line in body.splitlines()) == 2,
                    'require the existing two-function borrowed Nano profile')
            layout = self.functions[1].params[0][1]
            fields = self.fields[layout]
            values = ['0' if tag == INT else 'false' for tag in fields]
            entries = ', '.join(f'f{i}: {value}' for i, value in enumerate(values))
            names = ', '.join(f'f{i}' for i in range(len(fields)))
            if self.exclusive:
                body += f'\nshadow {self.name(1)} {{\n'
                for suffix in ('first', 'second'):
                    body += f'    let mut owner_{suffix}: NlrOwned{layout} = NlrOwned{layout} {{ {entries} }}\n'
                    body += f'    let result_{suffix}: {self.type(self.functions[1].result)} = ({self.name(1)} &mut owner_{suffix})\n'
                    body += f'    let NlrOwned{layout} {{ {names} }} = owner_{suffix}\n'
                    for i, tag in enumerate(fields):
                        body += f'    let field_{suffix}_{i}: {self.type(tag)} = f{i}\n'
                body += '    assert (== result_first result_second)\n'
                for i in range(len(fields)):
                    body += f'    assert (== field_first_{i} field_second_{i})\n'
                body += '}\n'
            else:
                body += f'\nshadow {self.name(1)} {{\n    let owner: NlrOwned{layout} = NlrOwned{layout} {{ {entries} }}\n'
                body += f'    let observed: {self.type(self.functions[1].result)} = ({self.name(1)} &owner)\n'
                body += f'    assert (== observed ({self.name(1)} &owner))\n'
                body += f'    let NlrOwned{layout} {{ {names} }} = owner\n'
                for i, value in enumerate(values):
                    body += f'    assert (== f{i} {value})\n'
                body += '}\n'
        return '\n'.join(declarations) + '\n' + body


def reconstruct_owned(module, language):
    require(sum(len(fn['code']) for fn in module['functions']) <= 4096,
            'limit the complete owned module to 4096 instructions')
    analyses = [OwnedAnalyze(module, i) for i in range(len(module['functions']))]
    functions = [analysis.run() for analysis in analyses]
    visiting, visited = set(), set()
    def visit(index):
        require(index not in visiting, 'require acyclic direct resource calls')
        if index in visited:
            return
        visiting.add(index)
        for target in analyses[index].calls:
            visit(target)
        visiting.remove(index)
        visited.add(index)
    for index in range(len(functions)):
        visit(index)
    return OwnedEmit(functions, language, module['layouts']).run(module['entry'])
