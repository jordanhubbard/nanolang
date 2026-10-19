"""I retain ordered local resource transfers in one bounded owned region tree."""
from nanoisa_reconstruction import Analyze, Emit, Expr, Function, INT, BOOL, SIMPLE, require


# I inline only nontrapping scalar pretests without duplicated/discarded operands.
OWNED_PRETEST = {
    "NOP", "PUSH_I64", "PUSH_BOOL", "LOAD_LOCAL",
    "EQ", "NE", "LT", "LE", "GT", "GE",
    "I64_EQ", "I64_NE", "I64_LT_S", "I64_LE_S", "I64_GT_S", "I64_GE_S",
    "I64_LT_U", "I64_LE_U", "I64_GT_U", "I64_GE_U",
    "BOOL_AND", "BOOL_OR", "BOOL_NOT", "AND", "OR", "NOT", "CAST_BOOL", "CAST_INT",
}


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
        for i, layout in enumerate(self.layouts):
            require(layout['identity'] == i and layout['resource'] is True and
                    0 < len(layout['fields']) <= 16 and
                    all(t in (INT, BOOL) for t in layout['fields']),
                    'require exact resource identity and scalar fields')
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
        self.result = declared(self.fn['result_type'])
        require(self.fn['result'] == self.fn['result_type'][0], 'require matching result declarations')
        require(len(self.params) <= 8 and len(self.params) <= len(self.declarations) and
                all(declared(self.declarations[i]) == t for i, t in enumerate(self.params)),
                'require exact initial parameter slots')
        require(index != 0 or not self.params and self.result == INT,
                'require a scalar entry without parameters')
        self.live = {i: t[1] for i, t in enumerate(self.params) if isinstance(t, tuple)}
        self.used = set(self.live)
        self.seen = set()
        self.calls = set()
        self.code = self.fn['code']
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
                condition_stack = []
                for n in range(i, branch):
                    condition_instruction = self.code[n]
                    require(condition_instruction['op'] in OWNED_PRETEST,
                            'require a nontrapping pure scalar owned pretest')
                    if condition_instruction['op'] == 'LOAD_LOCAL':
                        require(condition_instruction['arg'] in self.types,
                                'require a scalar owned pretest local')
                    self.simple(n, condition_stack, initialized, [], pure=True)
                    require(all(value.tag in (INT, BOOL) for value in condition_stack),
                            'require exact scalar owned pretest values')
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
                body.append(('while', condition, loop_body))
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
                fields = self.layouts[arg]['fields']
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
                               for f, t in enumerate(self.layouts[layout]['fields']))
                body.append(('own_unpack', arg, layout, values))
                del self.live[arg]
                stack.extend(values)
            elif op == 'REGION_BEGIN':
                require(i + 3 < end, 'require a complete ephemeral read within its owned region')
                for offset in (1, 2, 3):
                    self.touch(i + offset)
                borrow, read, region_end_instruction = self.code[i + 1:i + 4]
                require(borrow['op'] == 'BORROW_LOCAL_SHARED' and read['op'] == 'REF_GET' and
                        region_end_instruction['op'] == 'REGION_END' and len(borrow['args']) == 2 and
                        len(read['args']) == 2 and borrow['args'][0] == read['args'][0],
                        'require one matching ephemeral shared scalar-field read')
                slot, field = borrow['args'][1], read['args'][1]
                layout = self.owner_local(slot)
                require(0 <= field < len(self.layouts[layout]['fields']),
                        'require a field within the exact owner layout')
                target = Expr(self.layouts[layout]['fields'][field], 'temporary', read['pc'])
                body.append(('own_read', target, slot, field))
                stack.append(target)
                i += 3
            elif op == 'CALL':
                require(0 < arg < len(self.module['functions']), 'require a direct non-entry callee')
                callee = self.module['functions'][arg]
                params = [owned_type(t[1]) if t[0] == 8 else t[0] for t in callee['params']]
                values = tuple(reversed([self.pop(stack, t) for t in reversed(params)]))
                tag, layout = callee['result_type']
                result_type = owned_type(layout) if tag == 8 else tag
                target = Expr(result_type, 'temporary', pc)
                body.append(('let', target, Expr(result_type, 'call', arg, values)))
                stack.append(target)
                self.calls.add(arg)
            elif op == 'RET':
                require(i == end - 1 and not self.live,
                        'require all owners consumed before terminal return')
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

    def name(self, index):
        return 'main' if self.language == 'nano' and index == 0 else super().name(index)

    def type(self, tag):
        if isinstance(tag, tuple) and tag[0] == 'owner':
            return f'NlrOwned{tag[1]}'
        return super().type(tag)

    def statements(self, statements, indent=1):
        c = self.language == 'c'
        for stmt in statements:
            kind = stmt[0]
            if kind == 'own_parameter':
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
                self.line(f'{typ} {name} = {value};' if c else f'let {name}: {typ} = {value}', indent)
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
        declarations = ['#include <stdint.h>\n#include <stdbool.h>'] if c else []
        for i, layout in enumerate(self.layouts):
            fields = (' '.join(f'{self.type(t)} f{j};' for j, t in enumerate(layout['fields'])) if c else
                      ', '.join(f'f{j}: {self.type(t)}' for j, t in enumerate(layout['fields'])))
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
