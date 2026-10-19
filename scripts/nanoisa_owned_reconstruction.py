"""I retain ordered local resource transfers in one bounded owned region tree."""
from nanoisa_reconstruction import Analyze, Emit, Expr, Function, INT, BOOL, SIMPLE, require


def owned_type(layout):
    return ('owner', layout)


class OwnedAnalyze(Analyze):
    def __init__(self, module):
        self.module = module
        require(module.get('profile') == 'owned-local-v1' and module['entry'] == 0 and
                len(module['functions']) == 1, 'require one verified local resource entry')
        self.fn = module['functions'][0]
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
        self.live = {}
        self.used = set()
        self.seen = set()
        self.calls = set()
        self.code = self.fn['code']

    def owner_local(self, slot):
        require(slot in self.live, 'require a live exact local owner')
        return self.live[slot]

    def run(self):
        stack, body, initialized = [], [], set()
        i = 0
        while i < len(self.code):
            ins = self.code[i]
            op, arg, pc = ins['op'], ins['arg'], ins['pc']
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
                require(i + 3 < len(self.code), 'require a complete ephemeral read region')
                borrow, read, end = self.code[i + 1:i + 4]
                require(borrow['op'] == 'BORROW_LOCAL_SHARED' and read['op'] == 'REF_GET' and
                        end['op'] == 'REGION_END' and len(borrow['args']) == 2 and
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
            elif op == 'RET':
                require(i == len(self.code) - 1 and not self.live,
                        'require all owners consumed before terminal return')
                result = self.pop(stack, INT)
                require(not stack, 'require an empty operand stack after scalar return')
                body.append(('return', result))
            else:
                require(op in SIMPLE and op not in ('CALL', 'PUSH_U8', 'PUSH_F64', 'F64_FROM_BITS', 'F64_TO_BITS', 'F64_NEG'),
                        'refuse unsupported resource-route instruction ' + op)
                require(all(value.tag in (INT, BOOL) for value in stack),
                        'keep owned tokens out of scalar stack operations')
                if op in ('LOAD_LOCAL', 'STORE_LOCAL'):
                    require(arg in self.types, 'require a declared scalar local')
                self.simple(i, stack, initialized, body)
                require(all(value.tag in (INT, BOOL) for value in stack),
                        'require exact scalar resource-route values')
            i += 1
        require(body and body[-1][0] == 'return', 'require an explicit terminal return')
        return [Function('owned_main', [], INT, self.types, self.fn['names'], body)]


class OwnedEmit(Emit):
    def __init__(self, functions, language, layouts):
        super().__init__(functions, language)
        self.layouts = layouts

    def type(self, tag):
        if isinstance(tag, tuple) and tag[0] == 'owner':
            return f'NlrOwned{tag[1]}'
        return super().type(tag)

    def statements(self, statements, indent=1):
        c = self.language == 'c'
        for stmt in statements:
            kind = stmt[0]
            if kind == 'own_pack':
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
        return '\n'.join(declarations) + '\n' + body


def reconstruct_owned(module, language):
    return OwnedEmit(OwnedAnalyze(module).run(), language, module['layouts']).run(module['entry'])
